//! Diarization test with two synthetic voices.
//!
//! Generates speech with two different macOS `say` voices, concatenates them
//! into one 16 kHz WAV (as if two remote participants spoke in turn), downloads
//! the diarization models (~45 MB), and checks that two speakers come back.
//!
//!   cargo test --test diarize_e2e -- --ignored --nocapture

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;

fn synth_wav(dir: &Path, name: &str, voice: &str, text: &str) -> PathBuf {
    let aiff = dir.join(format!("{name}.aiff"));
    let wav = dir.join(format!("{name}.wav"));
    let say = Command::new("say")
        .args(["-v", voice, "-o", aiff.to_str().unwrap(), text])
        .status()
        .expect("run say");
    assert!(say.success(), "say failed for voice {voice}");
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

fn read_samples(path: &Path) -> Vec<i16> {
    hound::WavReader::open(path)
        .unwrap()
        .samples::<i16>()
        .map(Result::unwrap)
        .collect()
}

#[test]
#[ignore = "downloads diarization models and runs TTS; run explicitly"]
fn two_voices_cluster_into_two_speakers() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&tmp).unwrap();

    let voice_a = synth_wav(
        &tmp,
        "voice-a",
        "Daniel",
        "Right, let's get started. The budget review is the main item today, and I would like to walk through the numbers carefully before we make any decisions at all.",
    );
    let voice_b = synth_wav(
        &tmp,
        "voice-b",
        "Samantha",
        "Thanks Daniel. Before you dive in, marketing has one blocker to raise about the launch date, because the vendor slipped their delivery by another two weeks.",
    );

    // Stitch: A, a second of silence, B — one mixed "system stream".
    let mut samples = read_samples(&voice_a);
    samples.extend(std::iter::repeat(0i16).take(16_000));
    samples.extend(read_samples(&voice_b));
    let combined = tmp.join("combined.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&combined, spec).unwrap();
    for s in &samples {
        writer.write_sample(*s).unwrap();
    }
    writer.finalize().unwrap();

    let models_dir = tmp.join("models");
    let (seg_name, seg_url) = distil_lib::diarize::SEGMENTATION_MODEL;
    let (emb_name, emb_url) = distil_lib::diarize::EMBEDDING_MODEL;
    let seg = models_dir.join(seg_name);
    let emb = models_dir.join(emb_name);
    distil_lib::models::download_file(seg_url, &seg, |_, _| {}).expect("segmentation model");
    distil_lib::models::download_file(emb_url, &emb, |_, _| {}).expect("embedding model");

    let spans = distil_lib::diarize::diarize_wav(&seg, &emb, &combined).expect("diarization");
    let speakers: std::collections::BTreeSet<usize> =
        spans.iter().map(|s| s.speaker).collect();
    for span in &spans {
        println!(
            "speaker {} — {:.1}s to {:.1}s",
            span.speaker,
            span.start_ms as f64 / 1000.0,
            span.end_ms as f64 / 1000.0
        );
    }
    assert!(
        speakers.len() >= 2,
        "expected two clustered speakers, got {speakers:?}"
    );

    // Labeling: one whisper-ish segment in each half should get distinct labels.
    let half = (samples.len() / 2 * 1000 / 16_000) as i64;
    let them = vec![
        distil_lib::transcribe::Segment { start_ms: 1_000, end_ms: 3_000, text: "a".into() },
        distil_lib::transcribe::Segment {
            start_ms: half + 1_000,
            end_ms: half + 3_000,
            text: "b".into(),
        },
    ];
    let labeled = distil_lib::diarize::label_them_segments(them, &spans);
    assert_ne!(labeled[0].0, labeled[1].0, "halves should be different speakers");
}
