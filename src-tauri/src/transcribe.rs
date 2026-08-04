use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

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

/// Transcribe one 16 kHz mono WAV into timestamped segments.
/// Blocking and heavy; call from a blocking task, one at a time.
pub fn transcribe(model: &Path, wav: &Path) -> Result<Vec<Segment>, String> {
    let samples = read_wav_mono_f32(wav)?;
    if samples.len() < 16_000 {
        // Under a second of audio — treat as silence rather than erroring.
        return Ok(Vec::new());
    }

    let ctx = WhisperContext::new_with_params(
        model.to_str().ok_or("model path is not valid utf-8")?,
        WhisperContextParameters::default(),
    )
    .map_err(|e| format!("failed to load whisper model: {e}"))?;
    let mut state = ctx.create_state().map_err(|e| e.to_string())?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("auto"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    state
        .full(params, &samples)
        .map_err(|e| format!("transcription failed: {e}"))?;

    let mut segments = Vec::with_capacity(state.full_n_segments() as usize);
    for segment in state.as_iter() {
        let text = segment
            .to_str_lossy()
            .map_err(|e| e.to_string())?
            .trim()
            .to_string();
        if text.is_empty() || is_noise_annotation(&text) {
            continue;
        }
        segments.push(Segment {
            // whisper timestamps are in centiseconds
            start_ms: segment.start_timestamp() * 10,
            end_ms: segment.end_timestamp() * 10,
            text,
        });
    }
    Ok(segments)
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
