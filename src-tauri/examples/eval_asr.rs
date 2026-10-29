//! Compare transcribers on your own recordings.
//!
//! Runs each engine over the audio of your most recent meetings (or WAVs you
//! name) with the app's own transcriber, then writes each engine's transcript
//! and a speed and accuracy report. Audio is only read: results go to a
//! separate folder that the library never scans.
//!
//! Distil deletes recordings after transcribing them, so turn on "Keep audio"
//! in Settings before the meetings you want to test.
//!
//!   cargo run --release --example eval_asr
//!   cargo run --release --example eval_asr -- --limit 2 --engines whisper:small,parakeet
//!
//! Options:
//!   --engines a,b,c   whisper:<model>, whisper:<model>+vad (skip silence first)
//!                     or parakeet (default: whisper:large-v3-turbo-q5_0,
//!                     whisper:large-v3-turbo-q5_0+vad,parakeet)
//!   --wav a.wav,b.wav 16 kHz mono WAVs to use instead of meetings
//!   --meetings DIR    meetings folder (default ~/Documents/Distil)
//!   --limit N         most recent meetings that still have audio (default 3)
//!   --models-dir DIR  where models are kept, and downloaded to if missing
//!                     (default: the app's own folder)
//!   --out DIR         results folder (default ~/Documents/Distil Eval/<timestamp> asr)
//!
//! A `<name>.txt` next to `<name>.wav` is taken as the correct transcript, and
//! the report then includes a word error rate. For a meeting stream that is
//! `audio-mic.txt` or `audio-system.txt` in the meeting's folder.
//!
//! See docs/model-eval.md for how to read the results.

use std::path::{Path, PathBuf};
use std::time::Instant;

use distil_lib::models::{download_model, download_parakeet, download_vad_model};
use distil_lib::transcribe::{Segment, Transcriber};

const DEFAULT_ENGINES: &str =
    "whisper:large-v3-turbo-q5_0,whisper:large-v3-turbo-q5_0+vad,parakeet";

#[derive(Debug, PartialEq)]
enum Engine {
    Whisper { model: String, vad: bool },
    Parakeet,
}

/// Where the models for one engine are, once fetched.
struct Models {
    whisper: Option<PathBuf>,
    parakeet: Option<PathBuf>,
    vad: Option<PathBuf>,
}

struct Args {
    engines: Vec<String>,
    wavs: Vec<PathBuf>,
    meetings: PathBuf,
    limit: usize,
    models_dir: PathBuf,
    out: PathBuf,
}

/// One recording to transcribe.
struct File {
    label: String,
    wav: PathBuf,
    audio_s: f64,
    reference: Option<String>,
}

struct Run {
    engine: usize,
    file: usize,
    transcribe_s: f64,
    result: Result<Vec<Segment>, String>,
    /// (word errors, reference words), when the file has a reference.
    errors: Option<(usize, usize)>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let engines = args
        .engines
        .iter()
        .map(|spec| parse_spec(spec))
        .collect::<Result<Vec<_>, _>>()?;
    let files = if args.wavs.is_empty() {
        load_meetings(&args.meetings, args.limit)?
    } else {
        load_wavs(&args.wavs)?
    };
    let models = engines
        .iter()
        .map(|engine| fetch_models(engine, &args.models_dir))
        .collect::<Result<Vec<_>, _>>()?;

    std::fs::create_dir_all(args.out.join("raw")).map_err(|e| e.to_string())?;
    eprintln!(
        "{} engine(s) × {} file(s), {:.0} s of audio. Models in {}.",
        engines.len(),
        files.len(),
        files.iter().map(|f| f.audio_s).sum::<f64>(),
        args.models_dir.display()
    );

    // Engines outer, files inner: each model loads once, which matters on an
    // 8 GB Mac where only one fits at a time.
    let total = engines.len() * files.len();
    let mut runs: Vec<Run> = Vec::with_capacity(total);
    let mut loads: Vec<Result<f64, String>> = Vec::new();
    for (index, spec) in args.engines.iter().enumerate() {
        eprint!("{spec}: loading … ");
        let started = Instant::now();
        let loaded = load(&engines[index], &models[index]);
        let load_s = started.elapsed().as_secs_f64();
        let mut transcriber = match loaded {
            Ok(transcriber) => {
                eprintln!("{load_s:.1}s");
                loads.push(Ok(load_s));
                transcriber
            }
            Err(e) => {
                eprintln!("failed: {e}");
                loads.push(Err(e.clone()));
                for file in 0..files.len() {
                    runs.push(Run {
                        engine: index,
                        file,
                        transcribe_s: 0.0,
                        result: Err(format!("model did not load: {e}")),
                        errors: None,
                    });
                }
                continue;
            }
        };

        for (file_index, file) in files.iter().enumerate() {
            eprint!("[{}/{total}] {spec} · {} … ", runs.len() + 1, file.label);
            let started = Instant::now();
            let result = transcriber.transcribe(&file.wav);
            let transcribe_s = started.elapsed().as_secs_f64();
            let mut errors = None;
            match &result {
                Ok(segments) => {
                    eprintln!(
                        "{transcribe_s:.1}s (RTF {:.2})",
                        rtf(transcribe_s, file.audio_s)
                    );
                    let raw = args.out.join("raw").join(&file.label);
                    std::fs::create_dir_all(&raw).map_err(|e| e.to_string())?;
                    std::fs::write(
                        raw.join(format!("{}.md", slug(spec))),
                        format_segments(segments),
                    )
                    .map_err(|e| e.to_string())?;
                    errors = file
                        .reference
                        .as_deref()
                        .map(|reference| word_errors(reference, &joined_text(segments)));
                }
                Err(e) => eprintln!("failed: {e}"),
            }
            runs.push(Run {
                engine: index,
                file: file_index,
                transcribe_s,
                result,
                errors,
            });
        }
    }

    write_report(&args, &files, &loads, &runs)?;
    println!("Results in {}", args.out.display());
    println!("Read report.md; raw/ holds each engine's transcript of each file.");
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let home = PathBuf::from(std::env::var("HOME").map_err(|_| "HOME is not set")?);
    let mut engines = DEFAULT_ENGINES.to_string();
    let mut wavs = String::new();
    let mut meetings = home.join("Documents/Distil");
    let mut limit = 3;
    let mut models_dir = home.join("Library/Application Support/com.philipjlane.distil/models");
    let mut out = None;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!("See the options at the top of src-tauri/examples/eval_asr.rs");
            std::process::exit(0);
        }
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--engines" => engines = value,
            "--wav" => wavs = value,
            "--meetings" => meetings = PathBuf::from(value),
            "--limit" => {
                limit = value
                    .parse()
                    .map_err(|_| format!("bad --limit {value:?}"))?
            }
            "--models-dir" => models_dir = PathBuf::from(value),
            "--out" => out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {flag}; try --help")),
        }
    }

    let list = |text: &str| -> Vec<String> {
        text.split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(String::from)
            .collect()
    };
    let engines = list(&engines);
    if engines.is_empty() {
        return Err("--engines is empty".into());
    }
    let out = out.unwrap_or_else(|| {
        home.join("Documents/Distil Eval").join(format!(
            "{} asr",
            chrono::Local::now().format("%Y-%m-%d %H.%M.%S")
        ))
    });
    if out.starts_with(&meetings) {
        return Err(
            "--out must not be inside the meetings folder, or results show up as meetings".into(),
        );
    }
    Ok(Args {
        engines,
        wavs: list(&wavs).into_iter().map(PathBuf::from).collect(),
        meetings,
        limit,
        models_dir,
        out,
    })
}

/// `whisper:<model>`, `whisper:<model>+vad` or `parakeet`.
fn parse_spec(spec: &str) -> Result<Engine, String> {
    if spec == "parakeet" {
        return Ok(Engine::Parakeet);
    }
    let unknown =
        || format!("unknown engine {spec:?}; use whisper:<model>, whisper:<model>+vad or parakeet");
    let rest = spec.strip_prefix("whisper:").ok_or_else(unknown)?;
    let (model, vad) = match rest.strip_suffix("+vad") {
        Some(model) => (model, true),
        None => (rest, false),
    };
    if model.is_empty() {
        return Err(unknown());
    }
    Ok(Engine::Whisper {
        model: model.to_string(),
        vad,
    })
}

/// Download the models an engine needs. Done for every engine before the
/// first is timed, so a bad name or a failed download stops the run early.
fn fetch_models(engine: &Engine, models_dir: &Path) -> Result<Models, String> {
    let mut models = Models {
        whisper: None,
        parakeet: None,
        vad: None,
    };
    match engine {
        Engine::Whisper { model, vad } => {
            models.whisper = Some(download_model(
                models_dir,
                model,
                progress(&format!("whisper {model}")),
            )?);
            if *vad {
                models.vad = Some(download_vad_model(models_dir, progress("VAD model"))?);
            }
        }
        Engine::Parakeet => {
            models.parakeet = Some(download_parakeet(models_dir, progress("Parakeet"))?);
            models.vad = Some(download_vad_model(models_dir, progress("VAD model"))?);
        }
    }
    Ok(models)
}

/// Report a download in tenths, the first time it reports anything.
fn progress(label: &str) -> impl FnMut(u64, Option<u64>) + '_ {
    let mut last = -1i64;
    move |received, total| {
        let Some(total) = total else { return };
        let tenth = (received * 10 / total.max(1)) as i64;
        if tenth != last {
            last = tenth;
            eprintln!("downloading {label}: {}%", tenth * 10);
        }
    }
}

fn load(engine: &Engine, models: &Models) -> Result<Transcriber, String> {
    match (engine, &models.whisper, &models.parakeet, &models.vad) {
        (Engine::Whisper { .. }, Some(model), _, vad) => {
            Transcriber::whisper(model, vad.as_deref())
        }
        (Engine::Parakeet, _, Some(dir), Some(vad)) => Transcriber::parakeet(dir, vad),
        _ => Err("models were not fetched".into()),
    }
}

/// The newest `limit` meetings that still have audio, each stream a file.
/// Folder names start with the recording time, so name order is date order.
fn load_meetings(root: &Path, limit: usize) -> Result<Vec<File>, String> {
    let entries = std::fs::read_dir(root).map_err(|e| format!("{}: {e}", root.display()))?;
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.join("audio-mic.wav").is_file() || path.join("audio-system.wav").is_file()
        })
        .collect();
    dirs.sort();
    dirs.reverse();
    dirs.truncate(limit);

    let mut files = Vec::new();
    for dir in dirs {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        for stream in ["mic", "system"] {
            let wav = dir.join(format!("audio-{stream}.wav"));
            if wav.is_file() {
                files.push(file_for(format!("{name} ({stream})"), wav)?);
            }
        }
    }
    if files.is_empty() {
        return Err(format!(
            "no meetings with audio in {}; turn on Keep audio in Settings and record one, or pass --wav",
            root.display()
        ));
    }
    Ok(files)
}

fn load_wavs(wavs: &[PathBuf]) -> Result<Vec<File>, String> {
    let mut files: Vec<File> = Vec::new();
    for wav in wavs {
        let stem = wav
            .file_stem()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut label = stem.clone();
        let mut copy = 2;
        while files.iter().any(|f| f.label == label) {
            label = format!("{stem} ({copy})");
            copy += 1;
        }
        files.push(file_for(label, wav.clone())?);
    }
    Ok(files)
}

fn file_for(label: String, wav: PathBuf) -> Result<File, String> {
    let reader = hound::WavReader::open(&wav).map_err(|e| format!("{}: {e}", wav.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        return Err(format!(
            "{}: expected 16 kHz mono, got {} Hz {} ch (ffmpeg -i in.wav -ar 16000 -ac 1 out.wav converts)",
            wav.display(),
            spec.sample_rate,
            spec.channels
        ));
    }
    let audio_s = f64::from(reader.duration()) / f64::from(spec.sample_rate);
    let reference = std::fs::read_to_string(reference_path(&wav))
        .ok()
        .filter(|text| !text.trim().is_empty());
    Ok(File {
        label,
        wav,
        audio_s,
        reference,
    })
}

/// `<wav stem>.txt` beside the WAV.
fn reference_path(wav: &Path) -> PathBuf {
    wav.with_extension("txt")
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

fn joined_text(segments: &[Segment]) -> String {
    segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_segments(segments: &[Segment]) -> String {
    if segments.is_empty() {
        return "_(no speech detected)_\n".into();
    }
    let mut out = String::new();
    for segment in segments {
        let seconds = segment.start_ms / 1000;
        out.push_str(&format!(
            "[{:02}:{:02}] {}\n",
            seconds / 60,
            seconds % 60,
            segment.text
        ));
    }
    out
}

/// Lowercase words with punctuation dropped; apostrophes vanish rather than
/// split, so "don't" stays one word.
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .filter(|c| !matches!(c, '\'' | '’'))
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(String::from)
        .collect()
}

/// (word errors, reference words): the word-level edit distance (substitutions,
/// insertions and deletions) between the two texts, and the length of the
/// reference. Their ratio is the word error rate.
fn word_errors(reference: &str, hypothesis: &str) -> (usize, usize) {
    let reference = words(reference);
    let hypothesis = words(hypothesis);
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut current = vec![0usize; hypothesis.len() + 1];
    for (i, expected) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, heard) in hypothesis.iter().enumerate() {
            let substitution = previous[j] + usize::from(expected != heard);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[hypothesis.len()], reference.len())
}

fn wer(errors: (usize, usize)) -> Option<f64> {
    (errors.1 > 0).then(|| errors.0 as f64 / errors.1 as f64)
}

fn rtf(transcribe_s: f64, audio_s: f64) -> f64 {
    if audio_s > 0.0 {
        transcribe_s / audio_s
    } else {
        0.0
    }
}

fn cell(value: Option<f64>, decimals: usize) -> String {
    value.map_or_else(|| "–".into(), |v| format!("{v:.decimals$}"))
}

fn percent(value: Option<f64>) -> String {
    value.map_or_else(|| "–".into(), |v| format!("{:.1}%", v * 100.0))
}

fn write_report(
    args: &Args,
    files: &[File],
    loads: &[Result<f64, String>],
    runs: &[Run],
) -> Result<(), String> {
    let with_reference = files.iter().filter(|f| f.reference.is_some()).count();
    let source = if args.wavs.is_empty() {
        format!("newest first from {}", args.meetings.display())
    } else {
        "named with --wav".to_string()
    };
    let mut report = format!(
        "# Transcription evaluation\n\n\
         - Files: {} ({source}), {:.0} s of audio\n\
         - Models: {}\n\
         - Real-time factor (RTF) = transcribe time / audio time. Below 1 is faster than real time; lower is faster.\n\
         - Transcribe time includes skipping silence where the engine does it, and not model loading, which is reported once per engine. The first file may include one-off warm-up.\n\
         - Word error rate (WER) compares each transcript with the `.txt` beside its WAV, ignoring case and punctuation; lower is better. {with_reference} of {} files have a reference.\n",
        files.len(),
        files.iter().map(|f| f.audio_s).sum::<f64>(),
        args.models_dir.display(),
        files.len(),
    );
    if with_reference == 0 {
        report.push_str(
            "- No reference transcripts: put the correct text in `<name>.txt` beside a WAV to get a WER.\n",
        );
    }

    report.push_str(
        "\n## Summary\n\n\
         | Engine | Load s | OK | Audio s | Transcribe s | RTF | Words | WER |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    for (index, spec) in args.engines.iter().enumerate() {
        let engine_runs: Vec<&Run> = runs.iter().filter(|r| r.engine == index).collect();
        let ok: Vec<&Run> = engine_runs
            .iter()
            .copied()
            .filter(|r| r.result.is_ok())
            .collect();
        let audio_s: f64 = ok.iter().map(|r| files[r.file].audio_s).sum();
        let transcribe_s: f64 = ok.iter().map(|r| r.transcribe_s).sum();
        let word_count: usize = ok
            .iter()
            .filter_map(|r| r.result.as_ref().ok())
            .map(|segments| words(&joined_text(segments)).len())
            .sum();
        // Pooled over files: total errors against total reference words.
        let pooled = ok
            .iter()
            .filter_map(|r| r.errors)
            .fold((0, 0), |sum, e| (sum.0 + e.0, sum.1 + e.1));
        let load_s = loads.get(index).and_then(|l| l.as_ref().ok()).copied();
        report.push_str(&format!(
            "| {spec} | {} | {}/{} | {audio_s:.0} | {transcribe_s:.1} | {} | {word_count} | {} |\n",
            cell(load_s, 1),
            ok.len(),
            engine_runs.len(),
            cell((audio_s > 0.0).then(|| rtf(transcribe_s, audio_s)), 2),
            percent(wer(pooled)),
        ));
    }

    report.push_str(
        "\n## Per file\n\n\
         | File | Engine | Audio s | Transcribe s | RTF | Segments | Words | WER |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    for (index, file) in files.iter().enumerate() {
        for run in runs.iter().filter(|r| r.file == index) {
            let spec = &args.engines[run.engine];
            let row = match &run.result {
                Ok(segments) => format!(
                    "{:.0} | {:.1} | {:.2} | {} | {} | {}",
                    file.audio_s,
                    run.transcribe_s,
                    rtf(run.transcribe_s, file.audio_s),
                    segments.len(),
                    words(&joined_text(segments)).len(),
                    percent(run.errors.and_then(wer)),
                ),
                Err(e) => format!("failed: {} | | | | |", e.replace('|', "/")),
            };
            report.push_str(&format!("| {} | {spec} | {row} |\n", file.label));
        }
    }
    std::fs::write(args.out.join("report.md"), report).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs() {
        assert_eq!(
            parse_spec("whisper:large-v3-turbo-q5_0"),
            Ok(Engine::Whisper {
                model: "large-v3-turbo-q5_0".into(),
                vad: false
            })
        );
        assert_eq!(
            parse_spec("whisper:small+vad"),
            Ok(Engine::Whisper {
                model: "small".into(),
                vad: true
            })
        );
        assert_eq!(parse_spec("parakeet"), Ok(Engine::Parakeet));
        assert!(parse_spec("whisper:").is_err());
        assert!(parse_spec("whisper:+vad").is_err());
        assert!(parse_spec("parakeet+vad").is_err());
        assert!(parse_spec("small").is_err());
        assert_eq!(
            DEFAULT_ENGINES
                .split(',')
                .map(|spec| parse_spec(spec).is_ok())
                .collect::<Vec<_>>(),
            [true, true, true]
        );
        assert_eq!(slug("whisper:small+vad"), "whisper_small_vad");
        assert_eq!(
            reference_path(Path::new("/m/2026 Sync/audio-mic.wav")),
            Path::new("/m/2026 Sync/audio-mic.txt")
        );
    }

    #[test]
    fn identical_text_has_no_errors_whatever_the_case_or_punctuation() {
        assert_eq!(
            word_errors("Hello, World! It's fine.", "hello world its FINE"),
            (0, 4)
        );
        assert_eq!(word_errors("Don't stop", "dont stop"), (0, 2));
    }

    #[test]
    fn substitutions_insertions_and_deletions_each_cost_one() {
        assert_eq!(word_errors("the cat sat down", "the cat sat up"), (1, 4));
        assert_eq!(word_errors("the cat sat", "the big cat sat"), (1, 3));
        assert_eq!(word_errors("the big cat sat", "the cat sat"), (1, 4));
        assert_eq!(word_errors("a b c d", "d c b a"), (4, 4));
        assert_eq!(
            wer(word_errors("the cat sat down", "the cat sat up")),
            Some(0.25)
        );
    }

    #[test]
    fn empty_sides() {
        // Nothing heard: every reference word was missed.
        assert_eq!(word_errors("one two three", ""), (3, 3));
        assert_eq!(wer((3, 3)), Some(1.0));
        // No reference words: the rate is undefined.
        assert_eq!(word_errors("", "something"), (1, 0));
        assert_eq!(wer((1, 0)), None);
        assert_eq!(word_errors("", ""), (0, 0));
    }

    #[test]
    fn a_large_error_rate_can_pass_one() {
        // Hallucinated text over silence: more words wrong than were spoken.
        let (errors, total) = word_errors("hi", "thank you so much for watching");
        assert_eq!((errors, total), (6, 1));
        assert!(wer((errors, total)).unwrap() > 1.0);
    }

    #[test]
    fn rates_and_cells() {
        assert_eq!(rtf(5.0, 10.0), 0.5);
        assert_eq!(rtf(5.0, 0.0), 0.0);
        assert_eq!(cell(Some(1.234), 2), "1.23");
        assert_eq!(cell(None, 2), "–");
        assert_eq!(percent(Some(0.0734)), "7.3%");
    }

    #[test]
    fn segments_are_written_as_timestamped_lines() {
        let segments = vec![
            Segment {
                start_ms: 0,
                end_ms: 2_000,
                text: "Hello.".into(),
            },
            Segment {
                start_ms: 65_400,
                end_ms: 70_000,
                text: "Again.".into(),
            },
        ];
        assert_eq!(
            format_segments(&segments),
            "[00:00] Hello.\n[01:05] Again.\n"
        );
        assert_eq!(format_segments(&[]), "_(no speech detected)_\n");
        assert_eq!(joined_text(&segments), "Hello. Again.");
    }
}
