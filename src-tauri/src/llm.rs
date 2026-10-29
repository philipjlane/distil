//! The notes model, whichever server runs it.
//!
//! Ollama keeps its native API: it alone gives a reliable thinking toggle, a
//! settable context window and model sizes for auto-pick. Everything else
//! (oMLX, mlx_lm.server, llama-server, LM Studio) speaks the OpenAI
//! chat-completions API, so one adapter covers them all.

use serde::{Deserialize, Serialize};

use crate::ollama;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    #[default]
    Ollama,
    /// Any OpenAI-compatible server; its base URL includes the `/v1` suffix.
    OpenAi,
}

pub struct Model {
    pub name: String,
    /// Billions of parameters, when the engine reports it (Ollama only).
    pub parameter_size: Option<f64>,
    /// Runs off this Mac: an Ollama `*-cloud` tag, or any model on an
    /// OpenAI-compatible server that isn't on loopback. Transcripts sent to
    /// it leave the Mac, so it is surfaced only behind the
    /// allow_cloud_models opt-in and never auto-picked.
    pub is_cloud: bool,
}

/// Sampling and system prompt per model family, taken from the model cards.
#[derive(Debug, PartialEq)]
pub struct Preset {
    pub system: Option<&'static str>,
    pub temperature: f64,
    pub top_k: Option<u32>,
    pub repeat_penalty: Option<f64>,
}

/// Liquid's recommended system prompt for LFM2-2.6B-Transcript.
const MEETING_ANALYST: &str = "You are an expert meeting analyst. Analyze the transcript \
carefully and provide clear, accurate information based on the content.";

pub fn preset_for(model: &str) -> Preset {
    let name = model.to_ascii_lowercase();
    if name.contains("lfm3") {
        Preset {
            system: None,
            temperature: 0.1,
            top_k: Some(50),
            repeat_penalty: Some(1.1),
        }
    } else if name.contains("lfm2") && name.contains("transcript") {
        Preset {
            system: Some(MEETING_ANALYST),
            temperature: 0.3,
            top_k: None,
            repeat_penalty: None,
        }
    } else {
        Preset {
            system: None,
            temperature: 0.3,
            top_k: None,
            repeat_penalty: None,
        }
    }
}

/// Notes run to ~300 words; the cap only stops a small model that loops.
/// Thinking needs room for the reasoning as well as the answer.
pub fn max_output_tokens(think: bool) -> u32 {
    if think {
        8_192
    } else {
        2_048
    }
}

pub struct Generation {
    pub text: String,
    pub stats: Stats,
}

/// What the server reported about the run; engines fill in what they can.
#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub prompt_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub load_ms: Option<f64>,
    pub prompt_ms: Option<f64>,
    pub output_ms: Option<f64>,
    /// Generation stopped at the token cap rather than finishing.
    pub truncated: bool,
}

pub fn list_models(
    engine: Engine,
    base_url: &str,
    include_cloud: bool,
) -> Result<Vec<Model>, String> {
    match engine {
        Engine::Ollama => ollama::list_models(base_url, include_cloud),
        Engine::OpenAi => {
            let models = openai_list_models(base_url)?;
            Ok(models
                .into_iter()
                .filter(|m| include_cloud || !m.is_cloud)
                .collect())
        }
    }
}

/// The user's pick, else auto. Ollama auto-picks its largest local model; an
/// OpenAI-compatible server reports no sizes, so auto only works when it
/// serves exactly one model.
pub fn resolve_model(
    engine: Engine,
    base_url: &str,
    selected: Option<&str>,
) -> Result<String, String> {
    if let Some(model) = selected {
        return Ok(model.to_string());
    }
    match engine {
        Engine::Ollama => ollama::auto_pick(base_url),
        Engine::OpenAi => match openai_list_models(base_url)?.as_slice() {
            [] => Err(format!("{base_url} isn't serving any models")),
            [only] if !only.is_cloud => Ok(only.name.clone()),
            [_] => Err(format!(
                "{base_url} isn't on this Mac, so its model is never picked automatically; choose it in Settings"
            )),
            _ => Err(format!("{base_url} serves several models; pick one in Settings")),
        },
    }
}

/// Refuse to send a transcript off the Mac unless the user opted in.
pub fn check_privacy(
    engine: Engine,
    base_url: &str,
    model: &str,
    allow_cloud: bool,
) -> Result<(), String> {
    if allow_cloud {
        return Ok(());
    }
    match engine {
        Engine::Ollama if model.ends_with("-cloud") => Err(format!(
            "{model} is a cloud model but cloud models are disabled — enable them in Settings or pick a local model"
        )),
        Engine::OpenAi if !is_loopback(base_url) => Err(format!(
            "{base_url} isn't on this Mac and cloud models are disabled; enable them in Settings or use a local server"
        )),
        _ => Ok(()),
    }
}

pub fn generate(
    engine: Engine,
    base_url: &str,
    model: &str,
    prompt: &str,
    think: bool,
) -> Result<Generation, String> {
    let preset = preset_for(model);
    let mut generation = match engine {
        Engine::Ollama => ollama::generate(base_url, model, prompt, &preset, think)?,
        Engine::OpenAi => openai_generate(base_url, model, prompt, &preset, think)?,
    };
    generation.text = ollama::strip_think_blocks(&generation.text)
        .trim()
        .to_string();
    Ok(generation)
}

/// localhost, 127.0.0.0/8 or ::1: the only hosts that keep a transcript on
/// this Mac.
pub fn is_loopback(url: &str) -> bool {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host_port = authority.rsplit('@').next().unwrap_or("");
    let host = if let Some(bracketed) = host_port.strip_prefix('[') {
        bracketed.split(']').next().unwrap_or("")
    } else {
        host_port.split(':').next().unwrap_or("")
    };
    let host = host.to_ascii_lowercase();
    host == "localhost"
        || host == "::1"
        || host
            .parse::<std::net::Ipv4Addr>()
            .is_ok_and(|ip| ip.is_loopback())
}

fn openai_list_models(base_url: &str) -> Result<Vec<Model>, String> {
    #[derive(Deserialize)]
    struct Listing {
        #[serde(default)]
        data: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        id: String,
    }

    let base_url = base_url.trim_end_matches('/');
    let mut response = ureq::get(format!("{base_url}/models"))
        .call()
        .map_err(|e| format!("No model server is reachable at {base_url}. Is it running? ({e})"))?;
    let listing: Listing = response.body_mut().read_json().map_err(|e| e.to_string())?;
    let remote = !is_loopback(base_url);
    Ok(listing
        .data
        .into_iter()
        .map(|entry| Model {
            name: entry.id,
            parameter_size: None,
            is_cloud: remote,
        })
        .collect())
}

fn openai_generate(
    base_url: &str,
    model: &str,
    prompt: &str,
    preset: &Preset,
    think: bool,
) -> Result<Generation, String> {
    #[derive(Deserialize)]
    struct Completion {
        #[serde(default)]
        choices: Vec<Choice>,
        usage: Option<Usage>,
        /// llama-server's own timing block; other servers omit it.
        timings: Option<Timings>,
    }
    #[derive(Deserialize)]
    struct Choice {
        message: Message,
        finish_reason: Option<String>,
    }
    #[derive(Deserialize)]
    struct Message {
        content: Option<String>,
    }
    #[derive(Deserialize)]
    struct Usage {
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
    }
    #[derive(Deserialize)]
    struct Timings {
        prompt_ms: Option<f64>,
        predicted_ms: Option<f64>,
    }

    let mut messages = Vec::new();
    if let Some(system) = preset.system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    messages.push(serde_json::json!({ "role": "user", "content": prompt }));
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "temperature": preset.temperature,
        "max_tokens": max_output_tokens(think),
        // Honoured by llama-server, mlx-lm and vLLM chat templates; ignored
        // elsewhere, which is why think blocks are still stripped afterwards.
        "chat_template_kwargs": { "enable_thinking": think },
    });
    if let Some(top_k) = preset.top_k {
        body["top_k"] = top_k.into();
    }
    if let Some(penalty) = preset.repeat_penalty {
        // llama-server and LM Studio read one name, mlx-lm and vLLM the other.
        body["repeat_penalty"] = penalty.into();
        body["repetition_penalty"] = penalty.into();
    }

    let base_url = base_url.trim_end_matches('/');
    let mut response = ureq::post(format!("{base_url}/chat/completions"))
        .send_json(body)
        .map_err(|e| format!("Generation with {model} at {base_url} failed: {e}"))?;
    let completion: Completion = response.body_mut().read_json().map_err(|e| e.to_string())?;
    let choice = completion
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| format!("{model} returned no choices"))?;
    Ok(Generation {
        text: choice.message.content.unwrap_or_default(),
        stats: Stats {
            prompt_tokens: completion.usage.as_ref().and_then(|u| u.prompt_tokens),
            output_tokens: completion.usage.as_ref().and_then(|u| u.completion_tokens),
            load_ms: None,
            prompt_ms: completion.timings.as_ref().and_then(|t| t.prompt_ms),
            output_ms: completion.timings.as_ref().and_then(|t| t.predicted_ms),
            truncated: choice.finish_reason.as_deref() == Some("length"),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_serialises_as_lowercase() {
        assert_eq!(
            serde_json::to_string(&Engine::OpenAi).unwrap(),
            "\"openai\""
        );
        assert_eq!(
            serde_json::from_str::<Engine>("\"ollama\"").unwrap(),
            Engine::Ollama
        );
    }

    #[test]
    fn presets_follow_the_model_cards() {
        assert_eq!(preset_for("lfm3:1.7b").temperature, 0.1);
        assert_eq!(preset_for("LiquidAI/LFM3-1.7B-MLX-4bit").top_k, Some(50));
        let transcript = preset_for("hf.co/LiquidAI/LFM2-2.6B-Transcript-GGUF:Q4_K_M");
        assert_eq!(transcript.system, Some(MEETING_ANALYST));
        assert_eq!(
            preset_for("qwen3.5:4b"),
            Preset {
                system: None,
                temperature: 0.3,
                top_k: None,
                repeat_penalty: None,
            }
        );
        assert_eq!(preset_for("lfm2.5:1.2b").system, None);
    }

    #[test]
    fn loopback_hosts_only() {
        assert!(is_loopback("http://localhost:8080/v1"));
        assert!(is_loopback("http://127.0.0.1:11434"));
        assert!(is_loopback("http://127.1.2.3/v1"));
        assert!(is_loopback("http://[::1]:8080/v1"));
        assert!(is_loopback("HTTP://LOCALHOST/v1"));
        assert!(!is_loopback("http://192.168.1.20:8080/v1"));
        assert!(!is_loopback("https://api.example.com/v1"));
        assert!(!is_loopback("http://localhost.example.com/v1"));
        assert!(!is_loopback("http://localhost@example.com/v1"));
        assert!(!is_loopback("http://0.0.0.0:8080/v1"));
    }

    #[test]
    fn privacy_gate() {
        assert!(check_privacy(
            Engine::Ollama,
            "http://localhost:11434",
            "qwen3.5:4b",
            false
        )
        .is_ok());
        assert!(check_privacy(
            Engine::Ollama,
            "http://localhost:11434",
            "gpt-oss:20b-cloud",
            false
        )
        .is_err());
        assert!(check_privacy(
            Engine::Ollama,
            "http://localhost:11434",
            "gpt-oss:20b-cloud",
            true
        )
        .is_ok());
        assert!(check_privacy(Engine::OpenAi, "http://localhost:8080/v1", "any", false).is_ok());
        assert!(check_privacy(Engine::OpenAi, "https://api.example.com/v1", "any", false).is_err());
        assert!(check_privacy(Engine::OpenAi, "https://api.example.com/v1", "any", true).is_ok());
    }
}
