use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sherpa_rs::transducer::{TransducerConfig, TransducerRecognizer};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::vad;

/// Which model transcribes meetings (the `transcriber` setting).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TranscriberKind {
    #[default]
    Whisper,
    /// NVIDIA Parakeet TDT 0.6B v3, run by sherpa-onnx. Fast, but only for
    /// 25 European languages.
    Parakeet,
}

/// Folder under the models dir holding the Parakeet files.
pub const PARAKEET_DIR: &str = "parakeet-tdt-0.6b-v3-int8";
/// Where the Parakeet files are published (int8 build, CC-BY-4.0, NVIDIA).
pub const PARAKEET_URL: &str =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main";
/// The files Parakeet needs, with their sizes in bytes.
pub const PARAKEET_FILES: [(&str, u64); 4] = [
    ("encoder.int8.onnx", 652_184_281),
    ("decoder.int8.onnx", 11_845_275),
    ("joiner.int8.onnx", 6_355_277),
    ("tokens.txt", 93_939),
];

pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

pub fn read_wav_mono_f32(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        return Err(format!(
            "{}: expected 16 kHz mono, got {} Hz {} ch",
            path.display(),
            spec.sample_rate,
            spec.channels
        ));
    }
    match spec.sample_format {
        hound::SampleFormat::Int => Ok(reader
            .samples::<i16>()
            .filter_map(Result::ok)
            .map(|s| f32::from(s) / 32768.0)
            .collect()),
        hound::SampleFormat::Float => {
            Ok(reader.samples::<f32>().filter_map(Result::ok).collect())
        }
    }
}

/// A transcription model, loaded once and reused for every stream of a
/// meeting. Loading is the slow part (seconds, and a gigabyte or so of RAM),
/// so keep one alive across both streams and drop it before loading anything
/// else: 8 GB Macs only have room for one model at a time.
pub struct Transcriber {
    model: Model,
}

enum Model {
    Whisper {
        context: WhisperContext,
        /// Present: transcribe only the speech in each stream.
        vad_model: Option<PathBuf>,
    },
    Parakeet {
        recognizer: TransducerRecognizer,
        vad_model: PathBuf,
    },
}

impl Transcriber {
    /// Load a whisper.cpp model. With a VAD model, whisper only sees the
    /// stretches of each stream that have speech in them.
    pub fn whisper(model: &Path, vad_model: Option<&Path>) -> Result<Self, String> {
        if let Some(vad_model) = vad_model {
            vad::check_model(vad_model)?;
        }
        let context = WhisperContext::new_with_params(
            model.to_str().ok_or("model path is not valid utf-8")?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| format!("failed to load whisper model: {e}"))?;
        Ok(Self {
            model: Model::Whisper {
                context,
                vad_model: vad_model.map(Path::to_path_buf),
            },
        })
    }

    /// Load Parakeet from the folder holding its files (see `PARAKEET_FILES`).
    /// It can't take a whole meeting at once, so the VAD model is required:
    /// each stretch of speech is transcribed separately.
    pub fn parakeet(model_dir: &Path, vad_model: &Path) -> Result<Self, String> {
        vad::check_model(vad_model)?;
        // sherpa-onnx aborts the process on a missing file, so look first.
        for (file_name, _) in PARAKEET_FILES {
            if !model_dir.join(file_name).is_file() {
                return Err(format!(
                    "Parakeet model file missing: {}",
                    model_dir.join(file_name).display()
                ));
            }
        }
        let path = |file_name: &str| model_dir.join(file_name).to_string_lossy().into_owned();
        let recognizer = TransducerRecognizer::new(TransducerConfig {
            encoder: path("encoder.int8.onnx"),
            decoder: path("decoder.int8.onnx"),
            joiner: path("joiner.int8.onnx"),
            tokens: path("tokens.txt"),
            // sherpa-rs defaults to the plain "transducer" type, which is wrong for NeMo.
            model_type: "nemo_transducer".into(),
            decoding_method: "greedy_search".into(),
            num_threads: 4,
            sample_rate: 16_000,
            feature_dim: 128,
            provider: None,
            ..Default::default()
        })
        .map_err(|e| format!("failed to load Parakeet: {e}"))?;
        Ok(Self {
            model: Model::Parakeet {
                recognizer,
                vad_model: vad_model.to_path_buf(),
            },
        })
    }

    /// Transcribe one 16 kHz mono WAV into timestamped segments, on the
    /// recording's own timeline. Blocking and heavy; call from a blocking
    /// task, one at a time.
    pub fn transcribe(&mut self, wav: &Path) -> Result<Vec<Segment>, String> {
        let samples = read_wav_mono_f32(wav)?;
        if samples.len() < 16_000 {
            // Under a second of audio — treat as silence rather than erroring.
            return Ok(Vec::new());
        }
        match &mut self.model {
            Model::Whisper { context, vad_model } => {
                whisper_segments(context, vad_model.as_deref(), &samples)
            }
            Model::Parakeet { recognizer, vad_model } => {
                parakeet_segments(recognizer, vad_model, &samples)
            }
        }
    }
}

fn whisper_segments(
    context: &WhisperContext,
    vad_model: Option<&Path>,
    samples: &[f32],
) -> Result<Vec<Segment>, String> {
    // whisper only gets the speech: pack the spans into one buffer, then map
    // each segment's times back onto the recording. (`enable_vad` in the
    // params would do this inside whisper.cpp, but whisper-rs never reaches
    // that code.)
    let packed = match vad_model {
        Some(vad_model) => {
            let spans = vad::speech_spans_joined(vad_model, samples)?;
            if spans.is_empty() {
                return Ok(Vec::new());
            }
            Some(vad::Packed::pack(samples, &spans))
        }
        None => None,
    };
    let audio = packed.as_ref().map_or(samples, |p| p.samples.as_slice());

    let mut state = context.create_state().map_err(|e| e.to_string())?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("auto"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    state
        .full(params, audio)
        .map_err(|e| format!("transcription failed: {e}"))?;

    let mut segments = Vec::with_capacity(state.full_n_segments() as usize);
    for segment in state.as_iter() {
        let text = segment.to_str_lossy().map_err(|e| e.to_string())?;
        let text = strip_dialogue_dash(text.trim()).to_string();
        if text.is_empty() || is_noise_annotation(&text) {
            continue;
        }
        // whisper timestamps are in centiseconds
        let (start_ms, end_ms) = (segment.start_timestamp() * 10, segment.end_timestamp() * 10);
        let (start_ms, end_ms) = match &packed {
            Some(packed) => packed.segment_to_original_ms(start_ms, end_ms),
            None => (start_ms, end_ms),
        };
        segments.push(Segment { start_ms, end_ms, text });
    }
    Ok(segments)
}

fn parakeet_segments(
    recognizer: &mut TransducerRecognizer,
    vad_model: &Path,
    samples: &[f32],
) -> Result<Vec<Segment>, String> {
    // Parakeet returns text without timestamps, so each speech span becomes
    // one segment with the span's own times.
    let mut segments = Vec::new();
    for span in vad::speech_spans(vad_model, samples)? {
        let text = recognizer
            .transcribe(16_000, &samples[span.clone()])
            .trim()
            .to_string();
        if text.is_empty() || is_noise_annotation(&text) {
            continue;
        }
        segments.push(Segment {
            start_ms: (span.start * 1000 / vad::SAMPLE_RATE) as i64,
            end_ms: (span.end * 1000 / vad::SAMPLE_RATE) as i64,
            text,
        });
    }
    Ok(segments)
}

/// Whisper writes subtitle-style dialogue dashes ("- Yes.") at some speaker
/// turns, more often on packed speech. The transcript labels speakers itself,
/// so the dash is noise.
fn strip_dialogue_dash(text: &str) -> &str {
    match text.strip_prefix(['-', '\u{2013}', '\u{2014}']) {
        Some(rest) if rest.starts_with(' ') => rest.trim_start(),
        _ => text,
    }
}

/// Whisper marks non-speech as `*music*`, `[Applause]`, `(laughs)`, `♪ … ♪`.
fn is_noise_annotation(text: &str) -> bool {
    let bracketed = [('*', '*'), ('[', ']'), ('(', ')')]
        .iter()
        .any(|(open, close)| text.starts_with(*open) && text.ends_with(*close));
    bracketed || clean_text(text).is_empty()
}

/// Lowercase, alphanumeric + single spaces — comparison form for dedup.
fn clean_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_space = true;
    for c in text.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim_end().to_string()
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        current[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitution = prev[j] + usize::from(ca != cb);
            current[j + 1] = substitution.min(prev[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut prev, &mut current);
    }
    prev[b.len()]
}

fn similarity(a: &str, b: &str) -> f64 {
    let longest = a.chars().count().max(b.chars().count());
    if longest == 0 {
        return 1.0;
    }
    1.0 - levenshtein(a, b) as f64 / longest as f64
}

/// Timestamp slack when matching across streams: acoustic delay is milliseconds
/// but whisper's segment timestamps jitter by seconds.
const BLEED_WINDOW_MS: i64 = 6_000;

fn overlaps(a: &Segment, b: &Segment, slack_ms: i64) -> bool {
    a.start_ms < b.end_ms + slack_ms && b.start_ms < a.end_ms + slack_ms
}

/// Drop mic segments that are echoes of the system stream — speaker audio the
/// microphone picked up acoustically. Bleed only travels speakers → mic, so
/// only the Me stream is filtered. A mic segment is bleed when its text is
/// contained in the stitched text of time-overlapping system segments (whisper
/// chunks the same audio differently per stream), or is a close Levenshtein
/// match of a single overlapping segment.
pub fn drop_mic_bleed(me: Vec<Segment>, them: &[Segment]) -> Vec<Segment> {
    me.into_iter()
        .filter(|m| !is_bleed_of(m, them))
        .collect()
}

fn is_bleed_of(me: &Segment, them: &[Segment]) -> bool {
    let me_clean = clean_text(&me.text);
    if me_clean.len() < 4 {
        return false; // too short to judge; keep it
    }
    let window: Vec<&Segment> = them
        .iter()
        .filter(|t| overlaps(me, t, BLEED_WINDOW_MS))
        .collect();
    if window.is_empty() {
        return false;
    }
    if me_clean.len() >= 12 {
        let stitched = window
            .iter()
            .map(|t| clean_text(&t.text))
            .collect::<Vec<_>>()
            .join(" ");
        if stitched.contains(&me_clean) {
            return true;
        }
    }
    window
        .iter()
        .any(|t| similarity(&me_clean, &clean_text(&t.text)) >= 0.75)
}

fn mmss(ms: i64) -> String {
    let total_seconds = ms / 1000;
    format!("{:02}:{:02}", total_seconds / 60, total_seconds % 60)
}

/// Interleave labeled segments by start time and coalesce consecutive
/// segments with the same label into one block.
pub fn merge_labeled(mut all: Vec<(String, Segment)>) -> String {
    all.sort_by_key(|(_, s)| s.start_ms);

    let mut out = String::from("# Transcript\n");
    if all.is_empty() {
        out.push_str("\n_(no speech detected)_\n");
        return out;
    }
    let mut current_speaker = "";
    for (speaker, segment) in &all {
        if speaker != current_speaker {
            current_speaker = speaker;
            out.push_str(&format!("\n**{}** [{}]\n", speaker, mmss(segment.start_ms)));
        }
        out.push_str(&segment.text);
        out.push('\n');
    }
    out
}

/// Two-label convenience used when diarization is off (or found one voice).
pub fn merge_transcript(me: Vec<Segment>, them: Vec<Segment>) -> String {
    let mut all: Vec<(String, Segment)> = Vec::with_capacity(me.len() + them.len());
    all.extend(me.into_iter().map(|s| ("Me".to_string(), s)));
    all.extend(them.into_iter().map(|s| ("Them".to_string(), s)));
    merge_labeled(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(text: &str, start_ms: i64, end_ms: i64) -> Segment {
        Segment { start_ms, end_ms, text: text.into() }
    }

    /// The system-stream segments from the first real-world recording that
    /// exposed the speaker-bleed problem.
    fn youtube_them() -> Vec<Segment> {
        vec![
            seg("You know, people think this is politics of envy.", 23_000, 25_500),
            seg("It's politics of f***ing poverty.", 27_000, 29_000),
            seg(
                "If you allow the people who own all of the assets to never have to sell an asset, then your",
                29_000,
                34_000,
            ),
            seg("kids will never own an asset.", 34_000, 35_500),
        ]
    }

    #[test]
    fn bleed_exact_duplicate_is_dropped() {
        let me = vec![seg("You know, people think this is politics of envy.", 25_000, 27_500)];
        assert!(drop_mic_bleed(me, &youtube_them()).is_empty());
    }

    #[test]
    fn bleed_with_different_chunking_is_dropped() {
        // The mic stream chunked the same audio differently: one segment is a
        // prefix of a system segment, another spans two system segments.
        let me = vec![
            seg("If you allow the people who own all of the assets", 29_000, 32_000),
            seg(
                "to never have to sell an asset, then your kids will never own an asset.",
                32_000,
                35_500,
            ),
        ];
        assert!(drop_mic_bleed(me, &youtube_them()).is_empty());
    }

    #[test]
    fn genuine_speech_overlapping_in_time_is_kept() {
        let me = vec![seg("Let's push the deadline to Friday.", 33_000, 35_000)];
        assert_eq!(drop_mic_bleed(me, &youtube_them()).len(), 1);
    }

    #[test]
    fn speech_outside_any_window_is_kept() {
        let me = vec![seg("Yeah", 10_000, 11_000)];
        assert_eq!(drop_mic_bleed(me, &youtube_them()).len(), 1);
    }

    #[test]
    fn noise_annotations_are_detected() {
        assert!(is_noise_annotation("*Police music*"));
        assert!(is_noise_annotation("[Music]"));
        assert!(is_noise_annotation("(laughs)"));
        assert!(is_noise_annotation("♪ ♪ ♪"));
        assert!(!is_noise_annotation("Politics of envy."));
    }

    #[test]
    fn dialogue_dashes_are_stripped() {
        assert_eq!(strip_dialogue_dash("- Yes, ship it."), "Yes, ship it.");
        assert_eq!(strip_dialogue_dash("\u{2013} Agreed."), "Agreed.");
        assert_eq!(strip_dialogue_dash("-5 degrees outside."), "-5 degrees outside.");
        assert_eq!(strip_dialogue_dash("Mid - sentence dash."), "Mid - sentence dash.");
    }

    #[test]
    fn censored_text_cleans_consistently() {
        assert_eq!(clean_text("It's politics of f***ing poverty."), "it s politics of f ing poverty");
    }

    #[test]
    fn merge_orders_and_coalesces() {
        let me = vec![seg("Morning!", 0, 1_000), seg("All good.", 2_000, 3_000)];
        let them = vec![seg("Hi Phil.", 1_200, 1_900)];
        let merged = merge_transcript(me, them);
        let me_block = merged.find("**Me** [00:00]").expect("first Me block");
        let them_block = merged.find("**Them** [00:01]").expect("Them block");
        let second_me = merged.find("**Me** [00:02]").expect("second Me block");
        assert!(me_block < them_block && them_block < second_me, "wrong order:\n{merged}");
    }
}
