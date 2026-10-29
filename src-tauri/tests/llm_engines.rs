//! Both notes engines against a fake server on loopback: what each sends, and
//! how each reply is read. No network, no models.
//!
//!   cargo test --test llm_engines

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use distil_lib::llm::{self, Engine};
use serde_json::{json, Value};

type Requests = Arc<Mutex<Vec<(String, Value)>>>;

/// Serves canned JSON by "METHOD /path" and records each request body.
fn fake_server(routes: Vec<(&'static str, Value)>) -> (String, Requests) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let requests: Requests = Arc::default();
    let seen = requests.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let mut length = 0;
            loop {
                let mut header = String::new();
                reader.read_line(&mut header).unwrap();
                if header.trim().is_empty() {
                    break;
                }
                if let Some((name, value)) = header.split_once(':') {
                    if name.eq_ignore_ascii_case("content-length") {
                        length = value.trim().parse().unwrap();
                    }
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();

            let route = request_line
                .split_whitespace()
                .take(2)
                .collect::<Vec<_>>()
                .join(" ");
            seen.lock().unwrap().push((
                route.clone(),
                serde_json::from_slice(&body).unwrap_or(Value::Null),
            ));
            let (status, reply) = match routes.iter().find(|(r, _)| *r == route) {
                Some((_, reply)) => ("200 OK", reply.to_string()),
                None => ("404 Not Found", "{}".to_string()),
            };
            let _ = write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                reply.len()
            );
        }
    });
    (url, requests)
}

fn last_body(requests: &Requests, route: &str) -> Value {
    requests
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(r, _)| r == route)
        .map(|(_, body)| body.clone())
        .expect("request was made")
}

#[test]
fn ollama_sends_presets_and_reads_timings() {
    let (url, requests) = fake_server(vec![(
        "POST /api/generate",
        json!({
            "response": "reasoning the template opened</think>## TL;DR\nShipped.",
            "done_reason": "stop",
            "load_duration": 2_000_000_000u64,
            "prompt_eval_count": 1000,
            "prompt_eval_duration": 500_000_000u64,
            "eval_count": 100,
            "eval_duration": 1_000_000_000u64,
        }),
    )]);

    let generation = llm::generate(Engine::Ollama, &url, "lfm3:1.7b", "PROMPT", false).unwrap();
    assert_eq!(generation.text, "## TL;DR\nShipped.");
    assert_eq!(generation.stats.prompt_tokens, Some(1000));
    assert_eq!(generation.stats.load_ms, Some(2000.0));
    assert_eq!(generation.stats.prompt_ms, Some(500.0));
    assert_eq!(generation.stats.output_ms, Some(1000.0));
    assert!(!generation.stats.truncated);

    let body = last_body(&requests, "POST /api/generate");
    assert_eq!(body["prompt"], "PROMPT");
    assert_eq!(body["think"], false);
    assert_eq!(body["options"]["temperature"], 0.1);
    assert_eq!(body["options"]["top_k"], 50);
    assert_eq!(body["options"]["repeat_penalty"], 1.1);
    assert_eq!(body["options"]["num_ctx"], 16_384);
    assert_eq!(body["options"]["num_predict"], 2_048);
    assert!(body.get("system").is_none());

    llm::generate(
        Engine::Ollama,
        &url,
        "hf.co/LiquidAI/LFM2-2.6B-Transcript-GGUF:Q4_K_M",
        "PROMPT",
        false,
    )
    .unwrap();
    let body = last_body(&requests, "POST /api/generate");
    assert!(body["system"]
        .as_str()
        .unwrap()
        .starts_with("You are an expert meeting analyst"));
    assert_eq!(body["options"]["temperature"], 0.3);
    assert!(body["options"].get("top_k").is_none());
}

#[test]
fn openai_sends_chat_and_reads_usage() {
    let (base, requests) = fake_server(vec![(
        "POST /v1/chat/completions",
        json!({
            "choices": [{
                "message": { "role": "assistant", "content": "<think>hmm</think>\n## TL;DR\nCut off" },
                "finish_reason": "length",
            }],
            "usage": { "prompt_tokens": 900, "completion_tokens": 8192 },
            "timings": { "prompt_ms": 300.0, "predicted_ms": 4000.0 },
        }),
    )]);
    let url = format!("{base}/v1/");

    let generation = llm::generate(
        Engine::OpenAi,
        &url,
        "LiquidAI/LFM3-1.7B-MLX-4bit",
        "PROMPT",
        true,
    )
    .unwrap();
    assert_eq!(generation.text, "## TL;DR\nCut off");
    assert_eq!(generation.stats.prompt_tokens, Some(900));
    assert_eq!(generation.stats.output_tokens, Some(8192));
    assert_eq!(generation.stats.prompt_ms, Some(300.0));
    assert_eq!(generation.stats.output_ms, Some(4000.0));
    assert!(generation.stats.truncated);

    let body = last_body(&requests, "POST /v1/chat/completions");
    assert_eq!(body["model"], "LiquidAI/LFM3-1.7B-MLX-4bit");
    assert_eq!(
        body["messages"],
        json!([{ "role": "user", "content": "PROMPT" }])
    );
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
    assert_eq!(body["max_tokens"], 8_192);
    assert_eq!(body["top_k"], 50);
    assert_eq!(body["repeat_penalty"], 1.1);
    assert_eq!(body["repetition_penalty"], 1.1);

    llm::generate(
        Engine::OpenAi,
        &url,
        "mlx-community/LFM2-2.6B-Transcript-4bit",
        "PROMPT",
        false,
    )
    .unwrap();
    let body = last_body(&requests, "POST /v1/chat/completions");
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(
        body["messages"][1],
        json!({ "role": "user", "content": "PROMPT" })
    );
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
}

#[test]
fn model_listing_and_auto_pick() {
    let (base, _) = fake_server(vec![
        (
            "GET /api/tags",
            json!({ "models": [
                { "name": "qwen3.5:4b", "details": { "parameter_size": "4.7B" } },
                { "name": "lfm3:1.7b", "details": { "parameter_size": "1.7B" } },
                { "name": "gpt-oss:20b-cloud", "details": { "parameter_size": "20B" } },
            ]}),
        ),
        (
            "GET /v1/models",
            json!({ "data": [{ "id": "a" }, { "id": "b" }] }),
        ),
        ("GET /solo/models", json!({ "data": [{ "id": "only" }] })),
    ]);

    let local = llm::list_models(Engine::Ollama, &base, false).unwrap();
    assert_eq!(local.len(), 2);
    assert_eq!(
        llm::list_models(Engine::Ollama, &base, true).unwrap().len(),
        3
    );
    assert_eq!(
        llm::resolve_model(Engine::Ollama, &base, None).unwrap(),
        "qwen3.5:4b"
    );
    assert_eq!(
        llm::resolve_model(Engine::Ollama, &base, Some("lfm3:1.7b")).unwrap(),
        "lfm3:1.7b"
    );

    let several = format!("{base}/v1");
    let listed = llm::list_models(Engine::OpenAi, &several, false).unwrap();
    assert_eq!(
        listed.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert!(listed
        .iter()
        .all(|m| !m.is_cloud && m.parameter_size.is_none()));
    assert!(llm::resolve_model(Engine::OpenAi, &several, None).is_err());
    assert_eq!(
        llm::resolve_model(Engine::OpenAi, &format!("{base}/solo"), None).unwrap(),
        "only"
    );
}
