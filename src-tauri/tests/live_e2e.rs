//! End-to-end test for the live transcript.
//!
//! Synthesises a minute of speech with macOS `say`, feeds it into a PCM file in
//! small chunks the way the capture sidecar does while recording, and drives the
//! preview through it on a simulated clock with the real tiny English whisper
//! model (~43 MB, downloaded on first run). Checks that text appears, that
//! committed text only ever grows and never repeats, that the watermark never
//! moves backwards, and that stopping commits the last words.
//! Ignored by default because it hits the network:
//!
//!   cargo test --test live_e2e -- --ignored

#![cfg(target_os = "macos")]

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use distil_lib::live::{self, Line, LiveEvent, Preview, Stream, WhisperDecoder};

const SENTENCES: [&str; 14] = [
    "Let's review the quarterly budget numbers today.",
    "The marketing team needs three new laptops.",
    "Revenue grew by twelve percent compared with last year.",
    "We should hire two more engineers before the summer.",
    "The customer survey results arrive on Friday.",
    "Please send the updated forecast to the finance group.",
    "Our biggest risk is the delay in the supplier contract.",
    "Let's schedule a follow up meeting for next Tuesday morning.",
    "The new office lease needs approval from the board.",
    "Support tickets dropped by a third after the last release.",
    "We still have to decide who owns the onboarding project.",
    "Travel costs were higher than planned in the second quarter.",
    "The training budget will stay the same as last year.",
    "Finally, remember to submit your expense reports.",
];

/// Speech as 16 kHz mono int16 samples.
fn synth(dir: &Path, name: &str, text: &str) -> Vec<i16> {
    let aiff = dir.join(format!("{name}.aiff"));
    let wav = dir.join(format!("{name}.wav"));
    let say = Command::new("say")
        .args(["-o", aiff.to_str().unwrap(), text])
        .status()
        .expect("run say");
    assert!(say.success(), "say failed");
    let convert = Command::new("afconvert")
        .args([
            "-f",
            "WAVE",
            "-d",
            "LEI16@16000",
            "-c",
            "1",
            aiff.to_str().unwrap(),
            wav.to_str().unwrap(),
        ])
        .status()
        .expect("run afconvert");
    assert!(convert.success(), "afconvert failed");
    hound::WavReader::open(&wav)
        .unwrap()
        .samples::<i16>()
        .map(Result::unwrap)
        .collect()
}

/// Append samples to a PCM file, as the sidecar does.
fn append(path: &Path, samples: &[i16]) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    file.write_all(&bytes).unwrap();
}

fn joined(lines: &[Line]) -> String {
    lines
        .iter()
        .map(|l| l.text.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "downloads a whisper model and runs TTS; run explicitly"]
fn live_preview_follows_a_growing_recording() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("live-e2e");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let speech = synth(&tmp, "speech", &SENTENCES.join(" "));
    let seconds = speech.len() / 16_000;
    assert!(
        seconds >= 30,
        "only {seconds} s of speech; the window needs to slide"
    );

    let model = distil_lib::models::download_model(&tmp.join("models"), live::MODEL, |_, _| {})
        .expect("model download");
    let mut preview = Preview::new(WhisperDecoder::load(&model).expect("load whisper"), &tmp);

    let mic = tmp.join(live::MIC_PCM);
    let system = tmp.join(live::SYSTEM_PCM);
    append(&system, &[]); // exists but stays empty, like a call with nobody else talking

    // Each event with the seconds of audio fed when it was sent; the final
    // flush is marked `AT_STOP`.
    const AT_STOP: usize = usize::MAX;
    let mut events: Vec<(usize, LiveEvent)> = Vec::new();
    // Capture hands over audio a second at a time; the preview ticks every third.
    for (index, second) in speech.chunks(16_000).enumerate() {
        // Write each second in two pieces, the first ending mid-sample, to
        // exercise the reader's handling of a partial trailing byte.
        let bytes: Vec<u8> = second.iter().flat_map(|s| s.to_le_bytes()).collect();
        let cut = bytes.len() / 2 + 1;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&mic)
            .unwrap();
        file.write_all(&bytes[..cut]).unwrap();
        file.write_all(&bytes[cut..]).unwrap();
        let fed = index + 1; // seconds of audio written so far
        if index % 3 == 2 {
            let pass = preview.tick();
            assert!(pass.error.is_none(), "tick failed: {:?}", pass.error);
            events.extend(pass.events.into_iter().map(|event| (fed, event)));
        }
    }
    let last = preview.flush();
    assert!(last.error.is_none(), "final pass failed: {:?}", last.error);
    events.extend(last.events.into_iter().map(|event| (AT_STOP, event)));

    // Only the stream with sound in it speaks.
    assert!(!events.is_empty(), "the preview never said anything");
    assert!(
        events.iter().all(|(_, e)| e.stream == Stream::Me),
        "silence was transcribed"
    );

    // Committed text only ever grows: in order, without repeats, and the
    // watermark (the end of the newest committed line) never moves back.
    let committed: Vec<(usize, Line)> = events
        .iter()
        .flat_map(|(at, e)| e.committed.iter().map(move |l| (*at, l.clone())))
        .collect();
    assert!(!committed.is_empty(), "nothing was ever committed");
    let mut seen = HashSet::new();
    let mut watermark = 0;
    for (at, line) in &committed {
        assert!(
            seen.insert((line.start_ms, line.end_ms, line.text.clone())),
            "re-sent: {line:?}"
        );
        assert!(
            line.end_ms > watermark,
            "the watermark moved back at {at} s: {line:?}"
        );
        watermark = line.end_ms;
    }
    // Until the stop, a line commits only once it is behind the horizon.
    for (at, line) in committed.iter().filter(|(at, _)| *at != AT_STOP) {
        assert!(
            line.end_ms + live::COMMIT_HORIZON_MS <= *at as i64 * 1_000,
            "committed too early at {at} s: {line:?}"
        );
    }

    // Everything said ends up committed, including the words spoken just
    // before the stop, and nothing is left provisional.
    let text = joined(&committed.iter().map(|(_, l)| l.clone()).collect::<Vec<_>>());
    assert!(
        text.contains("quarterly budget"),
        "transcript was: {text:?}"
    );
    assert!(text.contains("laptops"), "transcript was: {text:?}");
    assert!(
        text.contains("expense"),
        "the last words were lost: {text:?}"
    );
    let (_, final_event) = events.last().unwrap();
    assert!(
        final_event.provisional.is_empty(),
        "provisional text left after the flush"
    );
}

#[test]
#[ignore = "downloads a whisper model; run explicitly"]
fn a_silent_recording_stays_silent() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("live-e2e-silent");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let model = distil_lib::models::download_model(&tmp.join("models"), live::MODEL, |_, _| {})
        .expect("model download");
    let mut preview = Preview::new(WhisperDecoder::load(&model).expect("load whisper"), &tmp);
    for _ in 0..10 {
        append(&tmp.join(live::MIC_PCM), &vec![0i16; 3 * 16_000]);
        append(&tmp.join(live::SYSTEM_PCM), &vec![0i16; 3 * 16_000]);
        let pass = preview.tick();
        assert!(pass.events.is_empty(), "silence produced {:?}", pass.events);
    }
    assert!(preview.flush().events.is_empty());
}
