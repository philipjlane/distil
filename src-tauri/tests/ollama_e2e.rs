//! Distillation test against a real local Ollama.
//!
//! Requires Ollama running with at least one local (non-cloud) model:
//!
//!   cargo test --test ollama_e2e -- --ignored --nocapture

const URL: &str = "http://localhost:11434";

#[test]
#[ignore = "requires a running Ollama with a local model"]
fn distils_sample_transcript_with_local_ollama() {
    let transcript = "# Transcript\n\n\
        **Me** [00:00]\nLet's plan the offsite. I'll book the venue by Friday.\n\n\
        **Them** [00:05]\nGreat. We decided on Brighton for the second week of September. \
        Can you also send the invite list to Sam?\n\n\
        **Me** [00:14]\nWill do. One open question — do we have budget sign-off yet?\n";

    let model = distil_lib::ollama::auto_pick(URL).expect("ollama running with a local model");
    println!("auto-picked model: {model}");

    let notes = distil_lib::llm::generate(
        distil_lib::llm::Engine::Ollama,
        URL,
        &model,
        &distil_lib::ollama::build_notes_prompt(
            distil_lib::ollama::DEFAULT_NOTES_PROMPT,
            transcript,
        ),
        false,
    )
    .expect("generation succeeds")
    .text;
    println!("--- notes ---\n{notes}");

    assert!(notes.len() > 40, "notes suspiciously short: {notes:?}");
    assert!(!notes.contains("<think>"), "think block leaked into notes");
}
