//! Compare notes models on your own meetings.
//!
//! Runs each candidate over your most recent transcripts with the app's own
//! prompt, model presets and think-stripping, then writes a blind review file,
//! its answer key and a speed/format report. Meetings are only read: results
//! go to a separate folder that the library never scans.
//!
//!   cargo run --release --example eval_notes
//!   cargo run --release --example eval_notes -- --limit 3 --models lfm3:1.7b,qwen3.5:4b
//!
//! Options:
//!   --models a,b,c   candidates; append `+think` to let a model reason first
//!   --engine NAME    ollama (default) or openai, for any OpenAI-compatible server
//!   --url URL        server address (defaults: Ollama's, or http://localhost:8080/v1)
//!   --meetings DIR   meetings folder (default ~/Documents/Distil)
//!   --limit N        most recent meetings with a transcript (default 5)
//!   --prompt FILE    notes prompt (default: your custom prompt from Settings, if any)
//!   --out DIR        results folder (default ~/Documents/Distil Eval/<timestamp>)
//!
//! See docs/model-eval.md for which models to pull and how to read the results.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use distil_lib::llm::{self, Engine, Stats};
use distil_lib::ollama;

const DEFAULT_MODELS: &str = "qwen3.5:4b,lfm3:1.7b,hf.co/LiquidAI/LFM2-2.6B-Transcript-GGUF:Q4_K_M";
const SECTIONS: [&str; 5] = [
    "## TL;DR",
    "## Key points",
    "## Decisions",
    "## Action items",
    "## Open questions",
];

struct Args {
    models: Vec<String>,
    engine: Engine,
    url: String,
    meetings: PathBuf,
    limit: usize,
    prompt: Option<PathBuf>,
    out: PathBuf,
}

struct Meeting {
    name: String,
    dir: PathBuf,
    transcript: String,
}

struct Run {
    model: String,
    meeting: usize,
    wall_s: f64,
    result: Result<(String, Stats), String>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (template, prompt_source) = load_template(args.prompt.as_deref())?;
    let meetings = load_meetings(&args.meetings, args.limit)?;
    preflight(&args)?;

    std::fs::create_dir_all(args.out.join("raw")).map_err(|e| e.to_string())?;
    eprintln!(
        "{} model(s) × {} meeting(s) via {:?} at {}. Prompt: {prompt_source}.",
        args.models.len(),
        meetings.len(),
        args.engine,
        args.url
    );

    // Models outer, meetings inner: each model loads once, which matters on
    // an 8 GB Mac where only one fits at a time.
    let total = args.models.len() * meetings.len();
    let mut runs = Vec::with_capacity(total);
    for spec in &args.models {
        let (model, think) = parse_spec(spec);
        for (index, meeting) in meetings.iter().enumerate() {
            eprint!("[{}/{total}] {spec} · {} … ", runs.len() + 1, meeting.name);
            let input = ollama::notes_input(
                &meeting.transcript,
                &ollama::meeting_attendees(&meeting.dir),
            );
            let prompt = ollama::build_notes_prompt(&template, &input);
            let started = Instant::now();
            let result = llm::generate(args.engine, &args.url, model, &prompt, think)
                .map(|generation| (generation.text, generation.stats));
            let wall_s = started.elapsed().as_secs_f64();
            match &result {
                Ok((text, _)) => {
                    eprintln!("{wall_s:.1}s");
                    let raw = args.out.join("raw").join(&meeting.name);
                    std::fs::create_dir_all(&raw).map_err(|e| e.to_string())?;
                    std::fs::write(raw.join(format!("{}.md", slug(spec))), text)
                        .map_err(|e| e.to_string())?;
                }
                Err(e) => eprintln!("failed: {e}"),
            }
            runs.push(Run {
                model: spec.clone(),
                meeting: index,
                wall_s,
                result,
            });
        }
    }

    write_review(&args, &meetings, &runs)?;
    write_report(&args, &meetings, &runs, &prompt_source)?;
    println!("Results in {}", args.out.display());
    println!("Read review.md first (blind), then key.md and report.md.");
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let home = PathBuf::from(std::env::var("HOME").map_err(|_| "HOME is not set")?);
    let mut models = DEFAULT_MODELS.to_string();
    let mut engine = Engine::Ollama;
    let mut url = None;
    let mut meetings = home.join("Documents/Distil");
    let mut limit = 5;
    let mut prompt = None;
    let mut out = None;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!("See the options at the top of src-tauri/examples/eval_notes.rs");
            std::process::exit(0);
        }
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--models" => models = value,
            "--engine" => {
                engine = serde_json::from_value(serde_json::Value::from(value.as_str()))
                    .map_err(|_| format!("unknown engine {value:?}; use ollama or openai"))?
            }
            "--url" => url = Some(value),
            "--meetings" => meetings = PathBuf::from(value),
            "--limit" => {
                limit = value
                    .parse()
                    .map_err(|_| format!("bad --limit {value:?}"))?
            }
            "--prompt" => prompt = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {flag}; try --help")),
        }
    }

    let models: Vec<String> = models
        .split(',')
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(String::from)
        .collect();
    if models.is_empty() {
        return Err("--models is empty".into());
    }
    let url = url.unwrap_or_else(|| match engine {
        Engine::Ollama => "http://localhost:11434".into(),
        Engine::OpenAi => "http://localhost:8080/v1".into(),
    });
    let out = out.unwrap_or_else(|| {
        home.join("Documents/Distil Eval")
            .join(chrono::Local::now().format("%Y-%m-%d %H.%M.%S").to_string())
    });
    if out.starts_with(&meetings) {
        return Err(
            "--out must not be inside the meetings folder, or results show up as meetings".into(),
        );
    }
    Ok(Args {
        models,
        engine,
        url,
        meetings,
        limit,
        prompt,
        out,
    })
}

/// `name+think` lets the model reason before answering; the app never does.
fn parse_spec(spec: &str) -> (&str, bool) {
    match spec.strip_suffix("+think") {
        Some(model) => (model, true),
        None => (spec, false),
    }
}

fn load_template(path: Option<&Path>) -> Result<(String, String), String> {
    if let Some(path) = path {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        return Ok((text, path.display().to_string()));
    }
    // Where the app keeps a prompt customised in Settings (macOS app data dir).
    let custom = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join("Library/Application Support/com.philipjlane.distil/notes-prompt.md");
    match std::fs::read_to_string(&custom) {
        Ok(text) if !text.trim().is_empty() => Ok((text, format!("custom ({})", custom.display()))),
        _ => Ok((
            ollama::DEFAULT_NOTES_PROMPT.to_string(),
            "built-in default".into(),
        )),
    }
}

/// Newest first. Folder names start with the recording time, so name order
/// is date order.
fn load_meetings(root: &Path, limit: usize) -> Result<Vec<Meeting>, String> {
    let entries = std::fs::read_dir(root).map_err(|e| format!("{}: {e}", root.display()))?;
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("transcript.md").is_file())
        .collect();
    dirs.sort();
    dirs.reverse();

    let mut meetings = Vec::new();
    for dir in dirs {
        if meetings.len() == limit {
            break;
        }
        let transcript =
            std::fs::read_to_string(dir.join("transcript.md")).map_err(|e| e.to_string())?;
        if transcript.contains("_(no speech detected)_") {
            continue;
        }
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        meetings.push(Meeting {
            name,
            dir,
            transcript,
        });
    }
    if meetings.is_empty() {
        return Err(format!("no transcripts found in {}", root.display()));
    }
    Ok(meetings)
}

/// Fail fast with the pull commands, rather than an hour in.
fn preflight(args: &Args) -> Result<(), String> {
    let available: Vec<String> = llm::list_models(args.engine, &args.url, true)?
        .into_iter()
        .map(|m| m.name)
        .collect();
    let missing: Vec<&str> = args
        .models
        .iter()
        .map(|spec| parse_spec(spec).0)
        .filter(|model| !available.iter().any(|name| same_model(name, model)))
        .collect();
    match (args.engine, missing.is_empty()) {
        (_, true) => Ok(()),
        (Engine::Ollama, false) => Err(format!(
            "not pulled yet:\n{}",
            missing
                .iter()
                .map(|m| format!("  ollama pull {m}"))
                .collect::<Vec<_>>()
                .join("\n")
        )),
        // OpenAI-compatible servers often load on demand by name, so a model
        // missing from the listing may still work.
        (Engine::OpenAi, false) => {
            eprintln!(
                "warning: not listed by the server, trying anyway: {}",
                missing.join(", ")
            );
            Ok(())
        }
    }
}

/// Ollama lists untagged models as `name:latest`.
fn same_model(listed: &str, wanted: &str) -> bool {
    let tagged = |name: &str| {
        let last = name.rsplit('/').next().unwrap_or(name);
        if last.contains(':') {
            name.to_string()
        } else {
            format!("{name}:latest")
        }
    };
    tagged(listed).eq_ignore_ascii_case(&tagged(wanted))
}

fn slug(spec: &str) -> String {
    spec.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn write_review(args: &Args, meetings: &[Meeting], runs: &[Run]) -> Result<(), String> {
    let mut review = String::from(
        "# Blind review\n\n\
         For each meeting, skim the transcript, then score every candidate from 1 to 5 on:\n\n\
         - **Accuracy**: nothing invented, owners and dates right\n\
         - **Coverage**: the decisions and action items that matter are there\n\
         - **Usefulness**: you would keep these notes as they are\n\n\
         Then open key.md to see which model wrote which.\n",
    );
    let mut key = String::from("# Answer key\n");
    for (index, meeting) in meetings.iter().enumerate() {
        // A fixed per-meeting shuffle, so letters don't give models away.
        let mut candidates: Vec<&Run> = runs.iter().filter(|r| r.meeting == index).collect();
        candidates.sort_by_key(|run| {
            let mut hasher = DefaultHasher::new();
            (&meeting.name, &run.model).hash(&mut hasher);
            hasher.finish()
        });

        review.push_str(&format!(
            "\n---\n\n## {}\n\nTranscript: `{}`\n",
            meeting.name,
            meeting.dir.join("transcript.md").display()
        ));
        key.push_str(&format!("\n## {}\n\n", meeting.name));
        for (letter, run) in ('A'..='Z').zip(candidates) {
            let body = match &run.result {
                Ok((text, _)) if !text.is_empty() => text.clone(),
                Ok(_) => "_(empty output)_".into(),
                Err(e) => format!("_(failed: {e})_"),
            };
            review.push_str(&format!("\n### Candidate {letter}\n\n{body}\n"));
            key.push_str(&format!("- {letter}: {}\n", run.model));
        }
    }
    std::fs::write(args.out.join("review.md"), review).map_err(|e| e.to_string())?;
    std::fs::write(args.out.join("key.md"), key).map_err(|e| e.to_string())
}

struct Checks {
    words: usize,
    sections: usize,
    has_tldr: bool,
    action_items: usize,
}

fn check(text: &str) -> Checks {
    Checks {
        words: text.split_whitespace().count(),
        sections: SECTIONS.iter().filter(|s| text.contains(*s)).count(),
        has_tldr: text.contains(SECTIONS[0]),
        // The library's action-item parser accepts `- [ ]` and `* [x]` lines.
        action_items: text
            .lines()
            .filter(|line| {
                let line = line.trim_start();
                ["- [ ]", "- [x]", "- [X]", "* [ ]", "* [x]", "* [X]"]
                    .iter()
                    .any(|prefix| line.starts_with(prefix))
            })
            .count(),
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

fn rate(tokens: Option<u64>, ms: Option<f64>) -> Option<f64> {
    match (tokens, ms) {
        (Some(tokens), Some(ms)) if ms > 0.0 => Some(tokens as f64 / (ms / 1000.0)),
        _ => None,
    }
}

fn cell(value: Option<f64>, decimals: usize) -> String {
    value.map_or_else(|| "–".into(), |v| format!("{v:.decimals$}"))
}

fn write_report(
    args: &Args,
    meetings: &[Meeting],
    runs: &[Run],
    prompt_source: &str,
) -> Result<(), String> {
    let elided = meetings
        .iter()
        .filter(|m| {
            ollama::notes_input(&m.transcript, &ollama::meeting_attendees(&m.dir)).len()
                > ollama::MAX_TRANSCRIPT_CHARS
        })
        .count();
    let mut report = format!(
        "# Notes model evaluation\n\n\
         - Engine: {:?} at {}\n\
         - Prompt: {prompt_source}\n\
         - Meetings: {} (newest first from {})\n\
         - Transcripts over {} characters, middle elided as in the app: {elided}\n\
         - Prefill = reading the transcript; decode = writing the notes. Wall time includes model load on each model's first meeting.\n",
        args.engine,
        args.url,
        meetings.len(),
        args.meetings.display(),
        ollama::MAX_TRANSCRIPT_CHARS,
    );

    let mut by_model: HashMap<&str, Vec<&Run>> = HashMap::new();
    for run in runs {
        by_model.entry(run.model.as_str()).or_default().push(run);
    }

    report.push_str(
        "\n## Summary\n\n\
         | Model | OK | Median wall s | Prefill tok/s | Decode tok/s | First load s | Median words | Missing TL;DR | Over 300 words | Hit token cap |\n\
         |---|---|---|---|---|---|---|---|---|---|\n",
    );
    for spec in &args.models {
        let model_runs = by_model.get(spec.as_str()).cloned().unwrap_or_default();
        let ok: Vec<(&String, &Stats, f64)> = model_runs
            .iter()
            .filter_map(|r| {
                r.result
                    .as_ref()
                    .ok()
                    .map(|(text, stats)| (text, stats, r.wall_s))
            })
            .collect();
        let checks: Vec<Checks> = ok.iter().map(|(text, ..)| check(text)).collect();
        let first_load = model_runs
            .first()
            .and_then(|r| r.result.as_ref().ok())
            .and_then(|(_, stats)| stats.load_ms)
            .map(|ms| ms / 1000.0);
        report.push_str(&format!(
            "| {spec} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            ok.len(),
            model_runs.len(),
            cell(median(ok.iter().map(|(.., wall)| *wall).collect()), 1),
            cell(
                median(
                    ok.iter()
                        .filter_map(|(_, s, _)| rate(s.prompt_tokens, s.prompt_ms))
                        .collect()
                ),
                0
            ),
            cell(
                median(
                    ok.iter()
                        .filter_map(|(_, s, _)| rate(s.output_tokens, s.output_ms))
                        .collect()
                ),
                1
            ),
            cell(first_load, 1),
            cell(median(checks.iter().map(|c| c.words as f64).collect()), 0),
            checks.iter().filter(|c| !c.has_tldr).count(),
            checks.iter().filter(|c| c.words > 300).count(),
            ok.iter().filter(|(_, s, _)| s.truncated).count(),
        ));
    }

    report.push_str(
        "\n## Per meeting\n\n\
         | Meeting | Model | Wall s | Prompt tokens | Output tokens | Words | Sections | Action items |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    for (index, meeting) in meetings.iter().enumerate() {
        for run in runs.iter().filter(|r| r.meeting == index) {
            let row = match &run.result {
                Ok((text, stats)) => {
                    let c = check(text);
                    format!(
                        "{:.1} | {} | {} | {} | {}/{} | {}",
                        run.wall_s,
                        stats.prompt_tokens.map_or("–".into(), |t| t.to_string()),
                        stats.output_tokens.map_or("–".into(), |t| t.to_string()),
                        c.words,
                        c.sections,
                        SECTIONS.len(),
                        c.action_items,
                    )
                }
                Err(e) => format!("failed: {} | | | | |", e.replace('|', "/")),
            };
            report.push_str(&format!("| {} | {} | {row} |\n", meeting.name, run.model));
        }
    }
    std::fs::write(args.out.join("report.md"), report).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_and_tags() {
        assert_eq!(parse_spec("lfm3:1.7b+think"), ("lfm3:1.7b", true));
        assert_eq!(parse_spec("qwen3.5:4b"), ("qwen3.5:4b", false));
        assert!(same_model("lfm3:latest", "lfm3"));
        assert!(same_model(
            "hf.co/LiquidAI/LFM2-2.6B-Transcript-GGUF:Q4_K_M",
            "hf.co/liquidai/LFM2-2.6B-Transcript-GGUF:Q4_K_M"
        ));
        assert!(!same_model("lfm3:1.7b", "lfm3"));
        assert_eq!(
            slug("hf.co/LiquidAI/X:Q4_K_M+think"),
            "hf.co_LiquidAI_X_Q4_K_M_think"
        );
    }

    #[test]
    fn format_checks() {
        let c = check(
            "## TL;DR\nShipped.\n\n## Action items\n- [ ] Me: send deck\n- [x] Sam: book room\n",
        );
        assert!(c.has_tldr);
        assert_eq!(c.sections, 2);
        assert_eq!(c.action_items, 2);
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(vec![1.0, 2.0, 3.0, 4.0]), Some(2.5));
        assert_eq!(rate(Some(100), Some(500.0)), Some(200.0));
    }
}
