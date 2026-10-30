use std::path::Path;

use serde::Deserialize;

use crate::llm::{max_output_tokens, Generation, Model, Preset, Stats};

pub fn list_models(base_url: &str, include_cloud: bool) -> Result<Vec<Model>, String> {
    #[derive(Deserialize)]
    struct Tags {
        #[serde(default)]
        models: Vec<TagModel>,
    }
    #[derive(Deserialize)]
    struct TagModel {
        name: String,
        #[serde(default)]
        details: Details,
    }
    #[derive(Deserialize, Default)]
    struct Details {
        #[serde(default)]
        parameter_size: Option<String>,
    }

    let mut response = ureq::get(format!("{base_url}/api/tags"))
        .call()
        .map_err(|e| format!("Ollama isn't reachable at {base_url} — is it running? ({e})"))?;
    let tags: Tags = response.body_mut().read_json().map_err(|e| e.to_string())?;
    Ok(tags
        .models
        .into_iter()
        .map(|m| Model {
            parameter_size: m.details.parameter_size.as_deref().and_then(parse_parameter_size),
            is_cloud: m.name.ends_with("-cloud"),
            name: m.name,
        })
        .filter(|m| include_cloud || !m.is_cloud)
        .collect())
}

/// Largest LOCAL model wins — auto-pick must never reach for a cloud model,
/// regardless of the opt-in; cloud use is always a deliberate user choice.
pub fn pick_largest_local(models: &[Model]) -> Option<String> {
    models
        .iter()
        .filter(|m| !m.is_cloud)
        .max_by(|a, b| {
            a.parameter_size
                .unwrap_or(0.0)
                .total_cmp(&b.parameter_size.unwrap_or(0.0))
        })
        .map(|m| m.name.clone())
}

pub fn auto_pick(base_url: &str) -> Result<String, String> {
    let models = list_models(base_url, true)?;
    pick_largest_local(&models).ok_or_else(|| {
        "no local Ollama models found — run e.g. `ollama pull qwen3.5:4b` first".to_string()
    })
}

pub fn generate(
    base_url: &str,
    model: &str,
    prompt: &str,
    preset: &Preset,
    think: bool,
) -> Result<Generation, String> {
    /// Durations are nanoseconds.
    #[derive(Deserialize)]
    struct Generated {
        #[serde(default)]
        response: String,
        done_reason: Option<String>,
        load_duration: Option<u64>,
        prompt_eval_count: Option<u64>,
        prompt_eval_duration: Option<u64>,
        eval_count: Option<u64>,
        eval_duration: Option<u64>,
    }

    let mut options = serde_json::json!({
        "num_ctx": 16_384,
        "num_predict": max_output_tokens(think),
        "temperature": preset.temperature,
    });
    if let Some(top_k) = preset.top_k {
        options["top_k"] = top_k.into();
    }
    if let Some(penalty) = preset.repeat_penalty {
        options["repeat_penalty"] = penalty.into();
    }
    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "think": think,
        "options": options,
    });
    if let Some(system) = preset.system {
        body["system"] = system.into();
    }

    let mut response = ureq::post(format!("{base_url}/api/generate"))
        .send_json(body)
        .map_err(|e| format!("Ollama generation with {model} failed: {e}"))?;
    let generated: Generated = response.body_mut().read_json().map_err(|e| e.to_string())?;
    let ms = |ns: Option<u64>| ns.map(|ns| ns as f64 / 1e6);
    Ok(Generation {
        text: generated.response,
        stats: Stats {
            prompt_tokens: generated.prompt_eval_count,
            output_tokens: generated.eval_count,
            load_ms: ms(generated.load_duration),
            prompt_ms: ms(generated.prompt_eval_duration),
            output_ms: ms(generated.eval_duration),
            truncated: generated.done_reason.as_deref() == Some("length"),
        },
    })
}

/// Thinking models may interleave `<think>…</think>` reasoning; notes must not
/// contain it. Thinking is switched off by request; this is the belt-and-braces
/// pass for servers or templates that ignore the switch.
pub fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    // Templates that open `<think>` inside the prompt (LFM3 does) leave only
    // the closing tag in the reply: everything before it is reasoning.
    if let Some(end) = rest.find("</think>") {
        if !rest[..end].contains("<think>") {
            rest = &rest[end + "</think>".len()..];
        }
    }
    loop {
        let Some(start) = rest.find("<think>") else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            Some(end_offset) => rest = &rest[start + end_offset + "</think>".len()..],
            None => break, // unterminated block: drop the tail
        }
    }
    out
}

/// Transcripts longer than this have their middle elided to keep the prompt
/// inside the 16k context.
pub const MAX_TRANSCRIPT_CHARS: usize = 45_000;

/// Keep prompts inside the 16k context: elide the middle of very long
/// transcripts, preserving the opening and the (usually decision-heavy) end.
pub fn elide_middle(transcript: &str, max_chars: usize) -> String {
    if transcript.len() <= max_chars {
        return transcript.to_string();
    }
    let keep = max_chars / 2;
    let head_end = transcript
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= keep)
        .last()
        .unwrap_or(0);
    let tail_start = transcript.len() - keep;
    let tail_start = transcript
        .char_indices()
        .map(|(i, _)| i)
        .find(|&i| i >= tail_start)
        .unwrap_or(transcript.len());
    format!(
        "{}\n\n[… middle of transcript elided for length …]\n\n{}",
        &transcript[..head_end],
        &transcript[tail_start..]
    )
}

/// The built-in notes prompt. Users can override it (Settings → Notes prompt /
/// notes-prompt.md in the app data dir); `{transcript}` marks where the
/// transcript is substituted.
pub const DEFAULT_NOTES_PROMPT: &str = "\
You are Distil. Turn this raw meeting transcript into crisp meeting notes.
In the transcript, \"Me\" is the user's own voice and \"Them\" is everyone else.

Write GitHub-flavoured Markdown with exactly these sections, omitting any
section that has nothing grounded in the transcript:

## TL;DR
Two or three sentences.

## Key points
Bullet list.

## Decisions
Bullets for decisions actually made in the meeting.

## Action items
Task-list lines like \"- [ ] Owner: task (due date if stated)\". Use \"Me\" as
the owner for the user's own items.

## Open questions
Bullets.

If the transcript distinguishes numbered speakers (Them 1, Them 2, …), map them
to real names only on strong in-transcript evidence: a self-introduction, or a
name used in direct address immediately followed by that speaker responding —
ideally corroborated more than once. A \"Known attendees\" line, when present, is
the candidate set. People who are merely talked about are not speakers. Use
inferred names in the notes and end with a line like
\"Speakers: Them 1 = Sarah (inferred) · Them 2 = unidentified\". When unsure,
keep the number — an accurate \"Them 2\" beats a wrong name.

Rules: use only information from the transcript — no invention. Be specific.
Keep the whole thing under 300 words. Output only the Markdown notes.

Transcript:

{transcript}";

/// Fill a prompt template with the (length-capped) transcript. Templates
/// without a `{transcript}` placeholder get it appended at the end, so a
/// user's stripped-down custom prompt still works.
pub fn build_notes_prompt(template: &str, transcript: &str) -> String {
    let transcript = elide_middle(transcript, MAX_TRANSCRIPT_CHARS);
    if template.contains("{transcript}") {
        template.replace("{transcript}", &transcript)
    } else {
        format!("{template}\n\nTranscript:\n\n{transcript}")
    }
}

/// Attendee names from a meeting's meta.json, present when the recording came
/// from a calendar prompt.
pub fn meeting_attendees(dir: &Path) -> Vec<String> {
    std::fs::read(dir.join("meta.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|meta| meta.get("attendees").cloned())
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

/// The transcript as the notes model sees it. Attendee names give the
/// name-inference instruction its candidate set.
pub fn notes_input(transcript: &str, attendees: &[String]) -> String {
    if attendees.is_empty() {
        transcript.to_string()
    } else {
        format!(
            "Known attendees (from the calendar invite): {}.\n\n{transcript}",
            attendees.join(", ")
        )
    }
}

fn parse_parameter_size(size: &str) -> Option<f64> {
    let size = size.trim();
    let (number, unit) = size.split_at(size.len().saturating_sub(1));
    let value: f64 = number.trim().parse().ok()?;
    match unit {
        "B" | "b" => Some(value),
        "M" | "m" => Some(value / 1000.0),
        _ => size.parse().ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_pick_never_chooses_cloud() {
        let models = vec![
            Model { name: "qwen3.5:2b".into(), parameter_size: Some(2.7), is_cloud: false },
            Model {
                name: "deepseek-v3.1:671b-cloud".into(),
                parameter_size: Some(671.0),
                is_cloud: true,
            },
        ];
        assert_eq!(pick_largest_local(&models), Some("qwen3.5:2b".into()));
        let cloud_only = vec![Model {
            name: "gpt-oss:20b-cloud".into(),
            parameter_size: Some(20.0),
            is_cloud: true,
        }];
        assert_eq!(pick_largest_local(&cloud_only), None);
    }

    #[test]
    fn parameter_sizes_parse() {
        assert_eq!(parse_parameter_size("2.7B"), Some(2.7));
        assert_eq!(parse_parameter_size("4B"), Some(4.0));
        assert_eq!(parse_parameter_size("780M"), Some(0.78));
        assert_eq!(parse_parameter_size("nonsense"), None);
    }

    #[test]
    fn think_blocks_are_stripped() {
        assert_eq!(
            strip_think_blocks("<think>hmm</think>## TL;DR\nGood."),
            "## TL;DR\nGood."
        );
        assert_eq!(strip_think_blocks("plain"), "plain");
        assert_eq!(strip_think_blocks("a<think>unterminated"), "a");
        assert_eq!(
            strip_think_blocks("<think>x</think>a<think>y</think>b"),
            "ab"
        );
        assert_eq!(
            strip_think_blocks("reasoning opened by the template</think>## TL;DR"),
            "## TL;DR"
        );
    }

    #[test]
    fn short_transcripts_are_untouched() {
        assert_eq!(elide_middle("hello", 100), "hello");
    }

    #[test]
    fn long_transcripts_keep_head_and_tail() {
        let long = format!("START{}END", "x".repeat(60_000));
        let elided = elide_middle(&long, 10_000);
        assert!(elided.starts_with("START"));
        assert!(elided.ends_with("END"));
        assert!(elided.contains("elided for length"));
        assert!(elided.len() < 11_000);
    }

    #[test]
    fn prompt_embeds_transcript() {
        let prompt = build_notes_prompt(DEFAULT_NOTES_PROMPT, "**Me** [00:00]\nHello there.");
        assert!(prompt.contains("Hello there."));
        assert!(prompt.contains("## Action items"));
        assert!(!prompt.contains("{transcript}"));
    }

    #[test]
    fn custom_prompt_without_placeholder_gets_transcript_appended() {
        let prompt = build_notes_prompt("Summarise in one paragraph.", "Hello world.");
        assert!(prompt.starts_with("Summarise in one paragraph."));
        assert!(prompt.contains("Transcript:\n\nHello world."));
    }

    #[test]
    fn attendees_head_the_transcript() {
        assert_eq!(notes_input("Hello.", &[]), "Hello.");
        assert_eq!(
            notes_input("Hello.", &["Sarah".into(), "James".into()]),
            "Known attendees (from the calendar invite): Sarah, James.\n\nHello."
        );
    }

    #[test]
    fn custom_prompt_with_placeholder_controls_position() {
        let prompt = build_notes_prompt("Before\n{transcript}\nAfter", "MIDDLE");
        assert_eq!(prompt, "Before\nMIDDLE\nAfter");
    }
}
