//! Voice-activity detection: find the stretches of a recording with speech in
//! them, so transcription can skip the silence.
//!
//! One Silero model (sherpa-onnx's copy, about 0.6 MB) serves both
//! transcribers. whisper.cpp has its own VAD, but whisper-rs only runs it from
//! `whisper_full()` and not from the `whisper_full_with_state()` that it calls,
//! so whisper gets the same treatment as Parakeet: the spans found here are
//! packed into one buffer ([`Packed`]) and the timestamps mapped back.

use std::ops::Range;
use std::path::Path;

use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};

pub const SAMPLE_RATE: usize = 16_000;

/// (file name, download URL) of the Silero VAD model, MIT licensed.
pub const MODEL: (&str, &str) = (
    "silero_vad.onnx",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
);

/// Samples per call into the detector: Silero v5 at 16 kHz takes exactly this.
const WINDOW: usize = 512;

/// The detector trims each span tight to the speech, which clips word endings.
const END_PAD: usize = SAMPLE_RATE / 5;

/// Spans that touch after padding are merged, up to this length, for engines
/// that take one span at a time. Parakeet slows down on long inputs, so a
/// conversation with only short pauses must not become one multi-minute span.
const MAX_SPAN: usize = SAMPLE_RATE * 30;

/// Zeros between spans in the packed buffer. whisper.cpp's own packer uses
/// 100 ms, but on a real two-person recording that was too short a pause for
/// whisper tiny.en to hear a sentence end: it dropped the last sentence of
/// the packed audio. 300 ms and up transcribed it.
const GAP: usize = SAMPLE_RATE * 3 / 10;

/// A boundary this close to a span edge is treated as sitting on the edge.
/// (Spans start with about this much pre-roll and end with `END_PAD` of
/// padding, so it is silence either way.)
const SNAP: usize = SAMPLE_RATE * 3 / 10;

/// A missing model file makes sherpa-onnx hand back a null detector that
/// segfaults on first use, so check before it gets the chance.
pub fn check_model(model: &Path) -> Result<(), String> {
    if model.is_file() {
        Ok(())
    } else {
        Err(format!("VAD model not found: {}", model.display()))
    }
}

/// Where the speech is in 16 kHz mono `samples`, as sample ranges on the
/// original timeline: sorted, end-padded and non-overlapping, and no longer
/// than 30 s unless one stretch of speech already is. For engines that
/// transcribe a span at a time.
pub fn speech_spans(model: &Path, samples: &[f32]) -> Result<Vec<Range<usize>>, String> {
    detect(model, samples, MAX_SPAN)
}

/// Like [`speech_spans`], but spans that touch are always joined, however long
/// the result. whisper chops its input into windows itself, so it gains nothing
/// from a split, and a split in unbroken speech would put a gap mid-word.
pub fn speech_spans_joined(model: &Path, samples: &[f32]) -> Result<Vec<Range<usize>>, String> {
    detect(model, samples, usize::MAX)
}

fn detect(model: &Path, samples: &[f32], max_span: usize) -> Result<Vec<Range<usize>>, String> {
    check_model(model)?;
    let config = SileroVadConfig {
        model: model
            .to_str()
            .ok_or("VAD model path is not valid utf-8")?
            .to_string(),
        threshold: 0.5,
        min_silence_duration: 0.5,
        min_speech_duration: 0.25,
        // The wrapper's default of 0.5 s cuts most speech short.
        max_speech_duration: 20.0,
        window_size: WINDOW as i32,
        num_threads: Some(1),
        sample_rate: SAMPLE_RATE as u32,
        ..Default::default()
    };
    let mut vad = SileroVad::new(config, 30.0).map_err(|e| format!("failed to start VAD: {e}"))?;

    let mut spans = Vec::new();
    let mut drain = |vad: &mut SileroVad| {
        while !vad.is_empty() {
            let segment = vad.front();
            vad.pop();
            // `start` is an absolute sample index.
            let start = segment.start.max(0) as usize;
            spans.push(start..start + segment.samples.len());
        }
    };
    // The detector only advances its state once per call, so feed it one
    // window at a time and drain after every call.
    for window in samples.chunks_exact(WINDOW) {
        vad.accept_waveform(window.to_vec());
        drain(&mut vad);
    }
    vad.flush();
    drain(&mut vad);

    Ok(pad_and_merge(&spans, END_PAD, samples.len(), max_span))
}

/// Extend each span's end by `pad` (up to `len`), then merge spans that touch
/// as long as the result stays within `max_len`. Spans too long to merge are
/// trimmed so they never share audio. Input order doesn't matter; empty and
/// out-of-range spans are dropped.
pub fn pad_and_merge(
    spans: &[Range<usize>],
    pad: usize,
    len: usize,
    max_len: usize,
) -> Vec<Range<usize>> {
    let mut padded: Vec<Range<usize>> = spans
        .iter()
        .filter(|s| s.start < s.end && s.start < len)
        .map(|s| s.start..(s.end + pad).min(len))
        .collect();
    padded.sort_by_key(|s| (s.start, s.end));

    let mut out: Vec<Range<usize>> = Vec::with_capacity(padded.len());
    for span in padded {
        if let Some(last) = out.last_mut() {
            if span.start <= last.end {
                let merged_end = last.end.max(span.end);
                if merged_end - last.start <= max_len || span.start == last.start {
                    last.end = merged_end;
                    continue;
                }
                last.end = span.start;
            }
        }
        out.push(span);
    }
    out
}

/// Where a span of the original recording sits in the packed buffer.
struct PackedSpan {
    packed_start: usize,
    orig_start: usize,
    len: usize,
}

/// The speech spans of a recording joined into one buffer, with the mapping
/// back to the original timeline.
pub struct Packed {
    pub samples: Vec<f32>,
    spans: Vec<PackedSpan>,
}

impl Packed {
    /// Join `spans` of `samples`, 300 ms of zeros apart. The spans must be
    /// sorted and non-overlapping, as [`speech_spans`] returns them.
    pub fn pack(samples: &[f32], spans: &[Range<usize>]) -> Packed {
        let mut packed = Packed {
            samples: Vec::new(),
            spans: Vec::new(),
        };
        for span in spans {
            let end = span.end.min(samples.len());
            if span.start >= end {
                continue;
            }
            if !packed.samples.is_empty() {
                packed.samples.resize(packed.samples.len() + GAP, 0.0);
            }
            packed.spans.push(PackedSpan {
                packed_start: packed.samples.len(),
                orig_start: span.start,
                len: end - span.start,
            });
            packed.samples.extend_from_slice(&samples[span.start..end]);
        }
        packed
    }

    /// A time in the packed buffer (ms) as a time in the original recording.
    ///
    /// Whisper's timestamps are 20 ms steps and often sit exactly on the
    /// previous segment's end, which is the gap between two spans. A boundary
    /// within `SNAP` of a span edge therefore jumps across the gap: a start
    /// moves forward to the next span, an end falls back to the previous one.
    /// A short span snaps over at most a quarter of its length, so a word in
    /// the middle of a half-second "Yeah." isn't mistaken for padding.
    pub fn to_original_ms(&self, packed_ms: i64, is_start: bool) -> i64 {
        if self.spans.is_empty() {
            return packed_ms;
        }
        let p = (packed_ms.max(0) as usize) * SAMPLE_RATE / 1000;
        let i = self
            .spans
            .partition_point(|s| s.packed_start <= p)
            .saturating_sub(1);
        let span = &self.spans[i];
        let offset = p.saturating_sub(span.packed_start);
        let snap = SNAP.min(span.len / 4);
        let orig = if is_start {
            match self.spans.get(i + 1) {
                Some(next) if offset + snap >= span.len => next.orig_start,
                _ => span.orig_start + offset.min(span.len),
            }
        } else if offset <= snap && i > 0 {
            let before = &self.spans[i - 1];
            before.orig_start + before.len
        } else {
            span.orig_start + offset.min(span.len)
        };
        (orig * 1000 / SAMPLE_RATE) as i64
    }

    /// A segment's (start, end) in the packed buffer as times in the original
    /// recording, with the end never before the start.
    pub fn segment_to_original_ms(&self, start_ms: i64, end_ms: i64) -> (i64, i64) {
        let start = self.to_original_ms(start_ms, true);
        (start, self.to_original_ms(end_ms, false).max(start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: usize = SAMPLE_RATE / 1000;
    const GAP_MS: i64 = (GAP / MS) as i64;

    fn ms(start: usize, end: usize) -> Range<usize> {
        start * MS..end * MS
    }

    #[test]
    fn padding_extends_ends_and_clamps_to_the_audio() {
        let spans = [ms(1_000, 2_000), ms(9_000, 10_000)];
        let out = pad_and_merge(&spans, 200 * MS, 10_100 * MS, usize::MAX);
        assert_eq!(out, vec![ms(1_000, 2_200), ms(9_000, 10_100)]);
    }

    #[test]
    fn spans_that_touch_after_padding_merge() {
        // 150 ms of silence between them: the 200 ms pad closes the gap.
        let spans = [ms(1_000, 2_000), ms(2_150, 3_000), ms(3_100, 4_000)];
        let out = pad_and_merge(&spans, 200 * MS, 60_000 * MS, usize::MAX);
        assert_eq!(out, vec![ms(1_000, 4_200)]);
    }

    #[test]
    fn a_real_gap_keeps_spans_apart() {
        let spans = [ms(1_000, 2_000), ms(2_500, 3_000)];
        let out = pad_and_merge(&spans, 200 * MS, 60_000 * MS, usize::MAX);
        assert_eq!(out, vec![ms(1_000, 2_200), ms(2_500, 3_200)]);
    }

    #[test]
    fn merging_stops_at_the_length_cap_without_sharing_audio() {
        // Back-to-back 20 s spans, as a forced split in unbroken speech gives.
        let spans = [ms(0, 20_000), ms(20_000, 40_000), ms(40_000, 45_000)];
        let out = pad_and_merge(&spans, 200 * MS, 60_000 * MS, 30_000 * MS);
        // 0..40.2 s is over the cap, but 20..45.2 s is not.
        assert_eq!(out, vec![ms(0, 20_000), ms(20_000, 45_200)]);
        // Every span is non-empty and none overlaps the next.
        assert!(out.iter().all(|s| s.start < s.end));
        assert!(out.windows(2).all(|w| w[0].end <= w[1].start));

        // Unbroken speech never becomes one long span.
        let spans: Vec<_> = (0..10).map(|i| ms(i * 20_000, (i + 1) * 20_000)).collect();
        let out = pad_and_merge(&spans, 200 * MS, 400_000 * MS, 30_000 * MS);
        assert_eq!(out.len(), 10);
        assert!(out.iter().all(|s| s.end - s.start <= 30_000 * MS));
        assert!(out.windows(2).all(|w| w[0].end <= w[1].start));
    }

    #[test]
    fn a_span_longer_than_the_cap_is_kept_whole() {
        let out = pad_and_merge(&[ms(0, 40_000)], 0, 60_000 * MS, 30_000 * MS);
        assert_eq!(out, vec![ms(0, 40_000)]);
    }

    #[test]
    fn unsorted_empty_and_out_of_range_spans_are_tidied() {
        let inverted = Range { start: 9, end: 3 };
        let spans = [
            ms(5_000, 6_000),
            7..7,
            ms(1_000, 2_000),
            ms(20_000, 21_000),
            inverted,
        ];
        let out = pad_and_merge(&spans, 0, 10_000 * MS, usize::MAX);
        assert_eq!(out, vec![ms(1_000, 2_000), ms(5_000, 6_000)]);
        assert!(pad_and_merge(&[], 200 * MS, 10_000 * MS, usize::MAX).is_empty());
    }

    #[test]
    fn spans_sharing_a_start_merge_even_over_the_cap() {
        let spans = [ms(0, 10_000), ms(0, 40_000)];
        let out = pad_and_merge(&spans, 0, 60_000 * MS, 30_000 * MS);
        assert_eq!(out, vec![ms(0, 40_000)]);
    }

    /// 2 s of speech at 1..3 s and 2 s at 6..8 s, in a 10 s recording where
    /// every sample's value is its index, so packing can be checked exactly.
    /// Packed, span 0 is at 0..2000 ms, the gap follows, and span 1 sits at
    /// 2300..4300 ms.
    fn two_span_fixture() -> (Vec<f32>, Packed) {
        assert_eq!(GAP_MS, 300, "the expected values below assume a 300 ms gap");
        let samples: Vec<f32> = (0..10 * SAMPLE_RATE).map(|i| i as f32).collect();
        let packed = Packed::pack(&samples, &[ms(1_000, 3_000), ms(6_000, 8_000)]);
        (samples, packed)
    }

    #[test]
    fn packing_joins_spans_with_a_zero_gap() {
        let (samples, packed) = two_span_fixture();
        assert_eq!(packed.samples.len(), 4 * SAMPLE_RATE + GAP);
        assert_eq!(packed.samples[0], samples[SAMPLE_RATE]);
        assert_eq!(
            packed.samples[2 * SAMPLE_RATE - 1],
            samples[3 * SAMPLE_RATE - 1]
        );
        assert!(packed.samples[2 * SAMPLE_RATE..2 * SAMPLE_RATE + GAP]
            .iter()
            .all(|s| *s == 0.0));
        assert_eq!(
            packed.samples[2 * SAMPLE_RATE + GAP],
            samples[6 * SAMPLE_RATE]
        );
        assert_eq!(
            *packed.samples.last().unwrap(),
            samples[8 * SAMPLE_RATE - 1]
        );
    }

    #[test]
    fn packing_skips_empty_spans_and_clamps_to_the_audio() {
        let samples = vec![1.0f32; SAMPLE_RATE];
        let packed = Packed::pack(&samples, &[10..10, ms(500, 5_000), ms(7_000, 8_000)]);
        assert_eq!(packed.samples.len(), SAMPLE_RATE / 2);
        let none = Packed::pack(&samples, &[]);
        assert!(none.samples.is_empty());
    }

    #[test]
    fn starts_inside_a_span_map_by_offset() {
        let (_, packed) = two_span_fixture();
        assert_eq!(packed.to_original_ms(0, true), 1_000);
        assert_eq!(packed.to_original_ms(500, true), 1_500);
        assert_eq!(packed.to_original_ms(1_000, true), 2_000);
        assert_eq!(packed.to_original_ms(2_300, true), 6_000);
        assert_eq!(packed.to_original_ms(2_800, true), 6_500);
        assert_eq!(packed.to_original_ms(3_500, true), 7_200);
    }

    #[test]
    fn starts_at_or_after_a_span_end_jump_to_the_next_span() {
        let (_, packed) = two_span_fixture();
        // Right on the end of span 0, in the gap, and 300 ms before the end.
        assert_eq!(packed.to_original_ms(2_000, true), 6_000);
        assert_eq!(packed.to_original_ms(2_150, true), 6_000);
        assert_eq!(packed.to_original_ms(2_299, true), 6_000);
        assert_eq!(packed.to_original_ms(1_700, true), 6_000);
        // 301 ms before the end is still inside span 0.
        assert_eq!(packed.to_original_ms(1_699, true), 2_699);
    }

    #[test]
    fn starts_in_the_last_span_stay_put() {
        let (_, packed) = two_span_fixture();
        // No next span to jump to: clamp to the span's end.
        assert_eq!(packed.to_original_ms(4_000, true), 7_700);
        assert_eq!(packed.to_original_ms(4_300, true), 8_000);
        assert_eq!(packed.to_original_ms(9_000, true), 8_000);
    }

    #[test]
    fn ends_inside_a_span_map_by_offset() {
        let (_, packed) = two_span_fixture();
        assert_eq!(packed.to_original_ms(0, false), 1_000);
        assert_eq!(packed.to_original_ms(500, false), 1_500);
        assert_eq!(packed.to_original_ms(2_000, false), 3_000);
        assert_eq!(packed.to_original_ms(3_000, false), 6_700);
        assert_eq!(packed.to_original_ms(4_300, false), 8_000);
        assert_eq!(packed.to_original_ms(9_000, false), 8_000);
    }

    #[test]
    fn ends_just_after_the_gap_fall_back_to_the_previous_span() {
        let (_, packed) = two_span_fixture();
        // In the gap: clamped to the end of span 0.
        assert_eq!(packed.to_original_ms(2_150, false), 3_000);
        // Up to 300 ms into span 1 still belongs to span 0's last segment.
        assert_eq!(packed.to_original_ms(2_300, false), 3_000);
        assert_eq!(packed.to_original_ms(2_600, false), 3_000);
        assert_eq!(packed.to_original_ms(2_601, false), 6_301);
    }

    #[test]
    fn the_first_span_never_snaps_back() {
        let (_, packed) = two_span_fixture();
        assert_eq!(packed.to_original_ms(100, false), 1_100);
        assert_eq!(packed.to_original_ms(-50, true), 1_000);
    }

    #[test]
    fn whisper_segments_across_the_gap_map_to_both_spans() {
        let (_, packed) = two_span_fixture();
        // A segment ending where the first span does, and the next one that
        // starts on that same boundary and runs to the end of the speech.
        assert_eq!(packed.segment_to_original_ms(0, 2_000), (1_000, 3_000));
        assert_eq!(packed.segment_to_original_ms(2_000, 4_300), (6_000, 8_000));
        // One segment spanning the gap keeps both ends.
        assert_eq!(packed.segment_to_original_ms(400, 3_500), (1_400, 7_200));
    }

    #[test]
    fn a_segment_inside_the_gap_never_ends_before_it_starts() {
        let (_, packed) = two_span_fixture();
        let (start, end) = packed.segment_to_original_ms(2_010, 2_290);
        assert!(end >= start, "{start}..{end}");
    }

    #[test]
    fn short_spans_snap_over_a_quarter_of_their_length() {
        // A 600 ms "Yeah." then a long span: packed, span 0 is 0..600 ms and
        // span 1 starts at 900 ms.
        let samples = vec![0.5f32; 10 * SAMPLE_RATE];
        let packed = Packed::pack(&samples, &[ms(1_000, 1_600), ms(6_000, 8_000)]);
        // Mid-span starts stay in the span, only starts in the last 150 ms jump.
        assert_eq!(packed.to_original_ms(300, true), 1_300);
        assert_eq!(packed.to_original_ms(449, true), 1_449);
        assert_eq!(packed.to_original_ms(450, true), 6_000);

        // The same holds for ends just inside a short span that follows a gap.
        let packed = Packed::pack(&samples, &[ms(1_000, 3_000), ms(6_000, 6_600)]);
        assert_eq!(packed.to_original_ms(2_450, false), 3_000);
        assert_eq!(packed.to_original_ms(2_500, false), 6_200);
    }

    #[test]
    fn a_single_span_maps_by_offset() {
        let samples = vec![0.5f32; 4 * SAMPLE_RATE];
        let packed = Packed::pack(&samples, &[ms(2_000, 3_500)]);
        assert_eq!(packed.to_original_ms(0, true), 2_000);
        assert_eq!(packed.to_original_ms(1_000, true), 3_000);
        assert_eq!(packed.to_original_ms(1_500, false), 3_500);
        assert_eq!(packed.to_original_ms(9_999, false), 3_500);
    }

    #[test]
    fn an_empty_buffer_maps_times_through_unchanged() {
        let packed = Packed::pack(&[], &[]);
        assert_eq!(packed.to_original_ms(1_234, true), 1_234);
        assert_eq!(packed.to_original_ms(1_234, false), 1_234);
        assert_eq!(packed.segment_to_original_ms(10, 20), (10, 20));
    }

    #[test]
    fn a_missing_model_is_an_error_rather_than_a_crash() {
        let missing = Path::new("/nonexistent/silero_vad.onnx");
        for result in [
            speech_spans(missing, &[0.0; 1024]),
            speech_spans_joined(missing, &[0.0; 1024]),
        ] {
            let err = result.expect_err("missing model");
            assert!(err.contains("VAD model not found"), "{err}");
        }
    }
}
