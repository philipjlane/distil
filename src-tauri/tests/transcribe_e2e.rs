//! End-to-end transcription test with synthetic speech.
//!
//! Generates two WAVs with macOS `say`, downloads the tiny English whisper
//! model (~43 MB), transcribes both, and checks the merged Me/Them transcript.
//! Ignored by default because it hits the network:
//!
//!   cargo test --test transcribe_e2e -- --ignored

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn synth_wav(dir: &Path, name: &str, text: &str) -> PathBuf {
    let aiff = dir.join(format!("{name}.aiff"));
    let wav = dir.join(format!("{name}.wav"));
    let say = Command::new("say")
        .args(["-o", aiff.to_str().unwrap(), text])
        .status()
        .expect("run say");
    assert!(say.success(), "say failed");
    let convert = Command::new("afconvert")
        .args([
            "-f", "WAVE",
            "-d", "LEI16@16000",
            "-c", "1",
            aiff.to_str().unwrap(),
            wav.to_str().unwrap(),
        ])
        .status()
        .expect("run afconvert");
    assert!(convert.success(), "afconvert failed");
    wav
}

#[test]
#[ignore = "downloads a whisper model and runs TTS; run explicitly"]
fn synthetic_two_stream_transcription() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&tmp).unwrap();

    let me_wav = synth_wav(&tmp, "me", "Let's review the quarterly budget numbers today.");
    let them_wav = synth_wav(&tmp, "them", "The marketing team needs three new laptops.");

    let model = distil_lib::models::download_model(&tmp.join("models"), "tiny.en-q8_0", |_, _| {})
        .expect("model download");

    let me = distil_lib::transcribe::transcribe(&model, &me_wav).expect("transcribe me");
    let them = distil_lib::transcribe::transcribe(&model, &them_wav).expect("transcribe them");

    let me_text = me.iter().map(|s| s.text.to_lowercase()).collect::<Vec<_>>().join(" ");
    let them_text = them.iter().map(|s| s.text.to_lowercase()).collect::<Vec<_>>().join(" ");
    assert!(me_text.contains("quarterly budget"), "me transcript was: {me_text:?}");
    assert!(them_text.contains("laptops"), "them transcript was: {them_text:?}");

    let merged = distil_lib::transcribe::merge_transcript(me, them);
    assert!(merged.contains("**Me**"), "merged transcript missing Me block:\n{merged}");
    assert!(merged.contains("**Them**"), "merged transcript missing Them block:\n{merged}");

    // Empty input stays graceful.
    let empty = distil_lib::transcribe::merge_transcript(Vec::new(), Vec::new());
    assert!(empty.contains("no speech detected"));
}
