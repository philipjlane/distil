//! End-to-end transcription test with synthetic speech.
//!
//! Generates two WAVs with macOS `say`, downloads the tiny English whisper
//! model (~43 MB), transcribes both, and checks the merged Me/Them transcript.
//! A second test does the same with the voice-activity model (~0.6 MB), padding
//! the speech with silence, and a third uses Parakeet (~670 MB).
//! Ignored by default because they hit the network:
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

    // One transcriber for both streams, as the app does.
    let mut transcriber =
        distil_lib::transcribe::Transcriber::whisper(&model, None).expect("load whisper");
    let me = transcriber.transcribe(&me_wav).expect("transcribe me");
    let them = transcriber.transcribe(&them_wav).expect("transcribe them");

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

const SPEECH_AT_MS: i64 = 5_000;

/// `wav` with five seconds of silence in front and eight behind, so the
/// speech starts at `SPEECH_AT_MS`.
fn pad_with_silence(dir: &Path, name: &str, wav: &Path) -> PathBuf {
    let speech: Vec<i16> = hound::WavReader::open(wav)
        .unwrap()
        .samples::<i16>()
        .map(Result::unwrap)
        .collect();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let padded = dir.join(format!("{name}.wav"));
    let mut writer = hound::WavWriter::create(&padded, spec).unwrap();
    let silence = |seconds: usize| std::iter::repeat_n(0i16, 16_000 * seconds);
    for sample in silence(5).chain(speech).chain(silence(8)) {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    padded
}

fn joined_text(segments: &[distil_lib::transcribe::Segment]) -> String {
    segments.iter().map(|s| s.text.to_lowercase()).collect::<Vec<_>>().join(" ")
}

#[test]
#[ignore = "downloads whisper and VAD models and runs TTS; run explicitly"]
fn whisper_skipping_silence_keeps_the_words_and_the_timeline() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&tmp).unwrap();
    let speech = synth_wav(&tmp, "vad-speech", "Let's review the quarterly budget numbers today.");
    let padded = pad_with_silence(&tmp, "vad-padded", &speech);

    let models = tmp.join("models");
    let model = distil_lib::models::download_model(&models, "tiny.en-q8_0", |_, _| {})
        .expect("model download");
    let vad = distil_lib::models::download_vad_model(&models, |_, _| {}).expect("VAD download");

    let mut transcriber =
        distil_lib::transcribe::Transcriber::whisper(&model, Some(&vad)).expect("load whisper");
    let segments = transcriber.transcribe(&padded).expect("transcribe");

    let text = joined_text(&segments);
    assert!(text.contains("quarterly budget"), "transcript was: {text:?}");
    // Timestamps are on the original recording, not the silence-free copy.
    let first = segments.first().expect("a segment");
    assert!(
        (first.start_ms - SPEECH_AT_MS).abs() < 1_000,
        "first segment starts at {} ms, speech starts at {SPEECH_AT_MS} ms",
        first.start_ms
    );

    // Pure silence has nothing to transcribe.
    let silent = tmp.join("vad-silent.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&silent, spec).unwrap();
    for _ in 0..16_000 * 10 {
        writer.write_sample(0i16).unwrap();
    }
    writer.finalize().unwrap();
    assert!(transcriber.transcribe(&silent).expect("transcribe silence").is_empty());
}

#[test]
#[ignore = "downloads Parakeet (~670 MB) and the VAD model and runs TTS; run explicitly"]
fn parakeet_transcribes_speech_amid_silence() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&tmp).unwrap();
    let speech = synth_wav(
        &tmp,
        "parakeet-speech",
        "Let's review the quarterly budget numbers today.",
    );
    let padded = pad_with_silence(&tmp, "parakeet-padded", &speech);

    let models = tmp.join("models");
    let vad = distil_lib::models::download_vad_model(&models, |_, _| {}).expect("VAD download");
    let parakeet =
        distil_lib::models::download_parakeet(&models, |_, _| {}).expect("Parakeet download");

    let mut transcriber =
        distil_lib::transcribe::Transcriber::parakeet(&parakeet, &vad).expect("load Parakeet");
    let segments = transcriber.transcribe(&padded).expect("transcribe");

    let text = joined_text(&segments);
    assert!(text.contains("quarterly budget"), "transcript was: {text:?}");
    let first = segments.first().expect("a segment");
    assert!(
        (first.start_ms - SPEECH_AT_MS).abs() < 1_000,
        "first segment starts at {} ms, speech starts at {SPEECH_AT_MS} ms",
        first.start_ms
    );
}
