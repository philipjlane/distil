use std::path::Path;

use crate::transcribe::Segment;

/// A contiguous stretch of one speaker's voice in the system stream,
/// as clustered by sherpa-onnx (pyannote segmentation + speaker embeddings).
pub struct SpeakerSpan {
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker: usize,
}

pub const SEGMENTATION_MODEL: (&str, &str) = (
    "pyannote-segmentation-3-0.onnx",
    "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx",
);
pub const EMBEDDING_MODEL: (&str, &str) = (
    "nemo_en_titanet_small.onnx",
    "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/nemo_en_titanet_small.onnx",
);

/// Cluster the (16 kHz mono) system-stream WAV into speaker spans.
/// Blocking and heavy; call while holding the pipeline gate.
pub fn diarize_wav(
    segmentation_model: &Path,
    embedding_model: &Path,
    wav: &Path,
) -> Result<Vec<SpeakerSpan>, String> {
    let samples = crate::transcribe::read_wav_mono_f32(wav)?;
    if samples.len() < 32_000 {
        return Ok(Vec::new()); // under two seconds — nothing to cluster
    }
    let config = sherpa_rs::diarize::DiarizeConfig::default();
    let mut diarizer = sherpa_rs::diarize::Diarize::new(
        segmentation_model.to_str().ok_or("segmentation model path is not utf-8")?,
        embedding_model.to_str().ok_or("embedding model path is not utf-8")?,
        config,
    )
    .map_err(|e| format!("failed to initialise diarization: {e}"))?;

    let spans = diarizer
        .compute(samples, None)
        .map_err(|e| format!("diarization failed: {e}"))?;
    Ok(spans
        .into_iter()
        .map(|s| SpeakerSpan {
            start_ms: (f64::from(s.start) * 1000.0) as i64,
            end_ms: (f64::from(s.end) * 1000.0) as i64,
            speaker: s.speaker as usize,
        })
        .collect())
}

/// Label transcribed system-stream segments with their dominant speaker,
/// renumbered by first appearance: "Them 1", "Them 2", … Collapses to plain
/// "Them" when only one voice was found (or diarization produced nothing).
pub fn label_them_segments(
    them: Vec<Segment>,
    spans: &[SpeakerSpan],
) -> Vec<(String, Segment)> {
    let distinct: std::collections::BTreeSet<usize> =
        spans.iter().map(|s| s.speaker).collect();
    if distinct.len() <= 1 {
        return them.into_iter().map(|s| ("Them".to_string(), s)).collect();
    }

    let mut order: Vec<usize> = Vec::new();
    for span in spans {
        if !order.contains(&span.speaker) {
            order.push(span.speaker);
        }
    }
    let display_number = |speaker: usize| -> usize {
        order.iter().position(|s| *s == speaker).map_or(0, |i| i + 1)
    };

    them.into_iter()
        .map(|segment| {
            let dominant = spans
                .iter()
                .map(|span| {
                    let overlap = segment.end_ms.min(span.end_ms)
                        - segment.start_ms.max(span.start_ms);
                    (overlap, span)
                })
                .max_by_key(|(overlap, _)| *overlap)
                .filter(|(overlap, _)| *overlap > 0)
                .map(|(_, span)| span.speaker)
                // No overlapping span (silence-adjacent segment): nearest span wins.
                .or_else(|| {
                    spans
                        .iter()
                        .min_by_key(|span| {
                            (span.start_ms - segment.start_ms).abs()
                        })
                        .map(|span| span.speaker)
                });
            let label = match dominant {
                Some(speaker) => format!("Them {}", display_number(speaker)),
                None => "Them".to_string(),
            };
            (label, segment)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(text: &str, start_ms: i64, end_ms: i64) -> Segment {
        Segment { start_ms, end_ms, text: text.into() }
    }

    #[test]
    fn two_speakers_get_numbered_by_first_appearance() {
        let spans = vec![
            SpeakerSpan { start_ms: 0, end_ms: 5_000, speaker: 7 },
            SpeakerSpan { start_ms: 5_000, end_ms: 10_000, speaker: 2 },
        ];
        let them = vec![seg("hello", 1_000, 2_000), seg("hi back", 6_000, 7_000)];
        let labeled = label_them_segments(them, &spans);
        assert_eq!(labeled[0].0, "Them 1"); // speaker 7 appeared first
        assert_eq!(labeled[1].0, "Them 2");
    }

    #[test]
    fn single_speaker_collapses_to_plain_them() {
        let spans = vec![SpeakerSpan { start_ms: 0, end_ms: 10_000, speaker: 0 }];
        let them = vec![seg("solo voice", 1_000, 2_000)];
        let labeled = label_them_segments(them, &spans);
        assert_eq!(labeled[0].0, "Them");
    }

    #[test]
    fn no_spans_collapses_to_plain_them() {
        let them = vec![seg("anything", 0, 1_000)];
        let labeled = label_them_segments(them, &[]);
        assert_eq!(labeled[0].0, "Them");
    }

    #[test]
    fn segment_outside_all_spans_takes_nearest_speaker() {
        let spans = vec![
            SpeakerSpan { start_ms: 0, end_ms: 2_000, speaker: 0 },
            SpeakerSpan { start_ms: 10_000, end_ms: 12_000, speaker: 1 },
        ];
        let them = vec![seg("late remark", 11_500, 13_500), seg("gap talk", 3_000, 4_000)];
        let labeled = label_them_segments(them, &spans);
        assert_eq!(labeled[0].0, "Them 2");
        assert_eq!(labeled[1].0, "Them 1"); // nearest span is the first
    }
}
