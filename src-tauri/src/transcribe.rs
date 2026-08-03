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
        if text.is_empty() {
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

fn mmss(ms: i64) -> String {
    let total_seconds = ms / 1000;
    format!("{:02}:{:02}", total_seconds / 60, total_seconds % 60)
}

/// Interleave the two single-speaker transcripts by start time and coalesce
/// consecutive segments from the same speaker into one block.
pub fn merge_transcript(me: Vec<Segment>, them: Vec<Segment>) -> String {
    let mut all: Vec<(&'static str, Segment)> = Vec::with_capacity(me.len() + them.len());
    all.extend(me.into_iter().map(|s| ("Me", s)));
    all.extend(them.into_iter().map(|s| ("Them", s)));
    all.sort_by_key(|(_, s)| s.start_ms);

    let mut out = String::from("# Transcript\n");
    if all.is_empty() {
        out.push_str("\n_(no speech detected)_\n");
        return out;
    }
    let mut current_speaker = "";
    for (speaker, segment) in &all {
        if *speaker != current_speaker {
            current_speaker = speaker;
            out.push_str(&format!("\n**{}** [{}]\n", speaker, mmss(segment.start_ms)));
        }
        out.push_str(&segment.text);
        out.push('\n');
    }
    out
}
