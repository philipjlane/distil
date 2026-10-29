use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime};

use crate::llm::Engine;
use crate::transcribe::{TranscriberKind, PARAKEET_DIR, PARAKEET_FILES, PARAKEET_URL};

/// User-tunable settings, stored as settings.json in the app data dir.
/// The Settings UI (M4) is sugar over this file.
#[derive(Serialize, Deserialize, Clone)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    /// whisper.cpp model name as published in ggerganov/whisper.cpp on
    /// Hugging Face, without the "ggml-" prefix or ".bin" suffix.
    pub whisper_model: String,
    /// Which model transcribes: whisper (using `whisper_model`) or Parakeet.
    pub transcriber: TranscriberKind,
    /// Transcribe only the stretches of each stream that have speech in them.
    /// Faster, and stops whisper inventing text over silence. Applies to
    /// whisper; Parakeet always works this way.
    pub skip_silence: bool,
    /// Keep raw WAVs after transcription. Off by default — raw recordings are
    /// the most sensitive artifact and the least useful once transcribed.
    pub keep_audio: bool,
    /// Ollama endpoint; localhost keeps everything on-device.
    pub ollama_url: String,
    /// Distillation model. None = auto-pick the largest local model.
    pub ollama_model: Option<String>,
    /// Which server writes the notes: Ollama, or any OpenAI-compatible server
    /// (oMLX, mlx_lm.server, llama-server, LM Studio).
    pub engine: Engine,
    /// OpenAI-compatible base URL, including the `/v1` suffix.
    pub openai_url: String,
    /// Notes model on the OpenAI-compatible server. None = auto, which only
    /// works when the server serves exactly one local model.
    pub openai_model: Option<String>,
    /// Explicit opt-in: surface `*-cloud` Ollama models in the picker, and
    /// allow an OpenAI-compatible server that isn't on this Mac. Transcripts
    /// sent to them leave the Mac (Ollama states zero retention).
    /// Cloud models are NEVER auto-picked, only chosen deliberately.
    pub allow_cloud_models: bool,
    /// Cluster distinct voices in the system stream into Them 1..N.
    pub diarize_speakers: bool,
    /// Show a rough, English-only transcript in a small floating window while
    /// recording. A disposable preview: the saved transcript is still made
    /// from the full recording afterwards.
    pub live_transcript: bool,
    /// Notice when a call app (Teams, Zoom, ...) starts using the microphone
    /// and offer to record it.
    pub detect_meetings: bool,
    /// Stop recording once every call app seen during the recording has
    /// released the microphone, instead of only asking. Needs
    /// `detect_meetings`; off by default so a recording never ends unasked.
    pub auto_stop_recording: bool,
    /// Last known result of the system-audio permission probe: "unknown",
    /// "granted" or "denied".
    ///
    /// Cached rather than queried because Core Audio exposes no way to read
    /// kTCCServiceAudioCapture — a denied tap runs normally and returns
    /// silence. Measuring it means playing a tone, which is far too rude to do
    /// every time the Settings pane opens, so the answer is remembered.
    pub system_audio_access: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            whisper_model: "large-v3-turbo-q5_0".into(),
            transcriber: TranscriberKind::Whisper,
            skip_silence: true,
            keep_audio: false,
            ollama_url: "http://localhost:11434".into(),
            ollama_model: None,
            engine: Engine::Ollama,
            openai_url: "http://localhost:8080/v1".into(),
            openai_model: None,
            allow_cloud_models: false,
            diarize_speakers: true,
            live_transcript: true,
            detect_meetings: true,
            auto_stop_recording: false,
            system_audio_access: "unknown".into(),
        }
    }
}

impl Settings {
    /// (engine, base URL, selected model) for whichever engine is active.
    pub fn notes_engine(&self) -> (Engine, &str, Option<&str>) {
        match self.engine {
            Engine::Ollama => (
                Engine::Ollama,
                self.ollama_url.as_str(),
                self.ollama_model.as_deref(),
            ),
            Engine::OpenAi => (
                Engine::OpenAi,
                self.openai_url.as_str(),
                self.openai_model.as_deref(),
            ),
        }
    }
}

fn notes_prompt_path<R: Runtime>(app: &AppHandle<R>) -> Result<std::path::PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|e| format!("cannot resolve app data dir: {e}"))?
        .join("notes-prompt.md"))
}

/// Returns (template, is_custom). Missing or blank file → built-in default.
pub fn load_notes_prompt<R: Runtime>(app: &AppHandle<R>) -> (String, bool) {
    match notes_prompt_path(app).map(std::fs::read_to_string) {
        Ok(Ok(text)) if !text.trim().is_empty() => (text, true),
        _ => (crate::ollama::DEFAULT_NOTES_PROMPT.to_string(), false),
    }
}

/// Blank text resets to the built-in default (removes the override file).
pub fn save_notes_prompt<R: Runtime>(app: &AppHandle<R>, text: &str) -> Result<(), String> {
    let path = notes_prompt_path(app)?;
    if text.trim().is_empty() {
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, text).map_err(|e| e.to_string())
}

pub fn save_settings<R: Runtime>(app: &AppHandle<R>, settings: &Settings) -> Result<(), String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("cannot resolve app data dir: {e}"))?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("settings.json"),
        serde_json::to_vec_pretty(settings).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

pub fn load_settings<R: Runtime>(app: &AppHandle<R>) -> Settings {
    let Ok(dir) = app.path().app_data_dir() else {
        return Settings::default();
    };
    let path = dir.join("settings.json");
    match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => {
            // First run: write the defaults so the file is discoverable.
            let defaults = Settings::default();
            if fs::create_dir_all(&dir).is_ok() {
                let _ = fs::write(
                    &path,
                    serde_json::to_vec_pretty(&defaults).unwrap_or_default(),
                );
            }
            defaults
        }
    }
}

/// Download `ggml-<model_name>.bin` from the whisper.cpp Hugging Face repo
/// into `models_dir`, reporting (received, total) as it goes. Blocking.
pub fn download_model(
    models_dir: &Path,
    model_name: &str,
    on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    let file_name = format!("ggml-{model_name}.bin");
    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file_name}");
    let target = models_dir.join(&file_name);
    download_file(&url, &target, on_progress)?;
    Ok(target)
}

/// Download an arbitrary file to `target` (atomically, via a .part file),
/// reporting (received, total). No-op if `target` already exists. Blocking.
pub fn download_file(
    url: &str,
    target: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<(), String> {
    if target.exists() {
        return Ok(());
    }
    let models_dir = target
        .parent()
        .ok_or("download target has no parent directory")?;
    fs::create_dir_all(models_dir).map_err(|e| e.to_string())?;
    let file_name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut response = ureq::get(url)
        .call()
        .map_err(|e| format!("model download failed ({file_name}): {e}"))?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let part = models_dir.join(format!("{file_name}.part"));
    let mut out = fs::File::create(&part).map_err(|e| e.to_string())?;
    let mut reader = response.body_mut().as_reader();
    let mut buf = [0u8; 1 << 16];
    let mut received: u64 = 0;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("download interrupted: {e}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        received += n as u64;
        on_progress(received, total);
    }
    drop(out);

    if let Some(total) = total {
        if received != total {
            let _ = fs::remove_file(&part);
            return Err(format!(
                "model download incomplete ({received} of {total} bytes)"
            ));
        }
    }
    fs::rename(&part, target).map_err(|e| e.to_string())?;
    Ok(())
}

/// Download the Silero VAD model into `models_dir`, reporting (received,
/// total). No-op if it is already there. Blocking.
pub fn download_vad_model(
    models_dir: &Path,
    on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    let (file_name, url) = crate::vad::MODEL;
    let target = models_dir.join(file_name);
    download_file(url, &target, on_progress)?;
    Ok(target)
}

/// Where Parakeet's files live under `models_dir`.
pub fn parakeet_dir(models_dir: &Path) -> PathBuf {
    models_dir.join(PARAKEET_DIR)
}

/// Whether every Parakeet file is on disk.
pub fn parakeet_present(models_dir: &Path) -> bool {
    let dir = parakeet_dir(models_dir);
    PARAKEET_FILES.iter().all(|(file_name, _)| dir.join(file_name).is_file())
}

/// Download Parakeet's four files into `models_dir`/`PARAKEET_DIR`, reporting
/// (received, total) summed across them. Files already present are skipped
/// and count as received. Blocking.
pub fn download_parakeet(
    models_dir: &Path,
    on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    let dir = parakeet_dir(models_dir);
    download_set(PARAKEET_URL, &dir, &PARAKEET_FILES, on_progress)?;
    Ok(dir)
}

/// Download each (name, size) of `files` from `base_url` into `dir`, with one
/// progress count across the set.
fn download_set(
    base_url: &str,
    dir: &Path,
    files: &[(&str, u64)],
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<(), String> {
    let total: u64 = files.iter().map(|(_, size)| size).sum();
    let mut finished = 0u64;
    for (file_name, size) in files {
        let url = format!("{base_url}/{file_name}");
        download_file(&url, &dir.join(file_name), |received, _| {
            on_progress((finished + received).min(total), Some(total));
        })?;
        finished += size;
    }
    Ok(())
}

/// Paths to the diarization models, downloading on first use.
/// Blocking; emits `model-download` events like the whisper path.
pub fn ensure_diarization_models<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<(PathBuf, PathBuf), String> {
    let models_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("cannot resolve app data dir: {e}"))?
        .join("models");

    let fetch = |(file_name, url): (&str, &str)| -> Result<PathBuf, String> {
        let target = models_dir.join(file_name);
        let missing = !target.exists();
        if missing {
            let _ = app.emit(
                "model-download",
                serde_json::json!({ "status": "started", "model": file_name }),
            );
        }
        let mut last_percent = -1i64;
        let result = download_file(url, &target, |received, total| {
            if let Some(total) = total {
                let percent = (received * 100 / total.max(1)) as i64;
                if percent != last_percent {
                    last_percent = percent;
                    let _ = app.emit(
                        "model-download",
                        serde_json::json!({
                            "status": "progress",
                            "receivedBytes": received,
                            "totalBytes": total,
                            "percent": percent,
                        }),
                    );
                }
            }
        });
        if missing {
            let status = if result.is_ok() { "done" } else { "failed" };
            let _ = app.emit("model-download", serde_json::json!({ "status": status }));
        }
        result.map(|()| target)
    };

    let segmentation = fetch(crate::diarize::SEGMENTATION_MODEL)?;
    let embedding = fetch(crate::diarize::EMBEDDING_MODEL)?;
    Ok((segmentation, embedding))
}

/// Path to the configured whisper model, downloading it on first use.
/// Blocking; call from a blocking task. Emits `model-download` events.
pub fn ensure_whisper_model<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    ensure_model(app, &load_settings(app).whisper_model)
}

/// Path to the named whisper model (`ggml-<name>.bin`), downloading it on
/// first use. Independent of the selected model, so the live preview can fetch
/// its own. Blocking; call from a blocking task. Emits `model-download` events.
pub fn ensure_model<R: Runtime>(app: &AppHandle<R>, name: &str) -> Result<PathBuf, String> {
    let dir = models_dir(app)?;
    let missing = !dir.join(format!("ggml-{name}.bin")).exists();
    download_with_events(app, name, missing, |progress| {
        download_model(&dir, name, progress)
    })
}

pub fn models_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|e| format!("cannot resolve app data dir: {e}"))?
        .join("models"))
}

/// Run `download`, emitting `model-download` events the way
/// `ensure_whisper_model` does. Nothing is emitted when `missing` is false.
fn download_with_events<R: Runtime>(
    app: &AppHandle<R>,
    model: &str,
    missing: bool,
    download: impl FnOnce(&mut dyn FnMut(u64, Option<u64>)) -> Result<PathBuf, String>,
) -> Result<PathBuf, String> {
    if missing {
        let _ = app.emit(
            "model-download",
            serde_json::json!({ "status": "started", "model": model }),
        );
    }
    let mut last_percent = -1i64;
    let result = download(&mut |received, total| {
        if let Some(total) = total {
            let percent = (received * 100 / total.max(1)) as i64;
            if percent != last_percent {
                last_percent = percent;
                let _ = app.emit(
                    "model-download",
                    serde_json::json!({
                        "status": "progress",
                        "receivedBytes": received,
                        "totalBytes": total,
                        "percent": percent,
                    }),
                );
            }
        }
    });
    if missing {
        let status = if result.is_ok() { "done" } else { "failed" };
        let _ = app.emit("model-download", serde_json::json!({ "status": status }));
    }
    result
}

/// Path to the VAD model, downloading it on first use (about 0.6 MB).
/// Blocking; call from a blocking task. Emits `model-download` events.
pub fn ensure_vad_model<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    let dir = models_dir(app)?;
    let (file_name, _) = crate::vad::MODEL;
    let missing = !dir.join(file_name).exists();
    download_with_events(app, file_name, missing, |progress| {
        download_vad_model(&dir, progress)
    })
}

/// Path to the folder holding Parakeet's files, downloading them on first
/// use (about 670 MB). Blocking; call from a blocking task. Emits
/// `model-download` events, with progress across all four files.
pub fn ensure_parakeet_model<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    let dir = models_dir(app)?;
    let missing = !parakeet_present(&dir);
    download_with_events(app, "parakeet-tdt-0.6b-v3", missing, |progress| {
        download_parakeet(&dir, progress)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    /// Serve `files` (path, bytes) over HTTP on a local port until the test ends.
    fn serve(files: &'static [(&'static str, &'static [u8])]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut request = String::new();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                reader.read_line(&mut request).unwrap();
                // Skip the headers.
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                }
                let path = request.split_whitespace().nth(1).unwrap_or("/");
                let reply = match files.iter().find(|(name, _)| path == format!("/{name}")) {
                    Some((_, body)) => {
                        let mut reply = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .into_bytes();
                        reply.extend_from_slice(body);
                        reply
                    }
                    None => {
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_vec()
                    }
                };
                let _ = stream.write_all(&reply);
            }
        });
        base
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("distil-models-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_set_downloads_with_progress_summed_across_files() {
        static FILES: [(&str, &[u8]); 3] =
            [("a.bin", &[1; 6000]), ("b.bin", &[2; 3000]), ("c.txt", &[3; 100])];
        let base = serve(&FILES);
        let dir = scratch_dir("set");

        let mut seen: Vec<(u64, Option<u64>)> = Vec::new();
        download_set(
            &base,
            &dir,
            &[("a.bin", 6000), ("b.bin", 3000), ("c.txt", 100)],
            |received, total| seen.push((received, total)),
        )
        .unwrap();

        assert_eq!(fs::read(dir.join("a.bin")).unwrap(), vec![1; 6000]);
        assert_eq!(fs::read(dir.join("b.bin")).unwrap(), vec![2; 3000]);
        assert_eq!(fs::read(dir.join("c.txt")).unwrap(), vec![3; 100]);
        assert!(seen.iter().all(|(_, total)| *total == Some(9100)));
        assert!(seen.windows(2).all(|w| w[0].0 <= w[1].0), "went backwards: {seen:?}");
        assert_eq!(seen.last().unwrap().0, 9100);
        // Progress reached the second and third files' ranges, not just the first's.
        assert!(seen.iter().any(|(received, _)| *received > 6000));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn files_already_on_disk_are_skipped_but_counted() {
        static FILES: [(&str, &[u8]); 2] = [("a.bin", &[1; 4000]), ("b.bin", &[2; 1000])];
        let base = serve(&FILES);
        let dir = scratch_dir("skip");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.bin"), vec![9; 4000]).unwrap();

        let mut last = (0, None);
        download_set(&base, &dir, &[("a.bin", 4000), ("b.bin", 1000)], |received, total| {
            last = (received, total)
        })
        .unwrap();

        assert_eq!(fs::read(dir.join("a.bin")).unwrap(), vec![9; 4000], "must not be re-fetched");
        assert_eq!(fs::read(dir.join("b.bin")).unwrap(), vec![2; 1000]);
        assert_eq!(last, (5000, Some(5000)));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_file_fails_the_set_and_leaves_no_partial() {
        static FILES: [(&str, &[u8]); 1] = [("a.bin", &[1; 100])];
        let base = serve(&FILES);
        let dir = scratch_dir("fail");

        let err = download_set(&base, &dir, &[("a.bin", 100), ("missing.bin", 50)], |_, _| {})
            .expect_err("404 must fail the set");
        assert!(err.contains("missing.bin"), "{err}");
        assert!(dir.join("a.bin").exists());
        assert!(!dir.join("missing.bin").exists());
        assert!(!dir.join("missing.bin.part").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parakeet_counts_as_present_only_with_all_four_files() {
        let models = scratch_dir("present");
        let dir = parakeet_dir(&models);
        assert!(!parakeet_present(&models));
        fs::create_dir_all(&dir).unwrap();
        for (file_name, _) in &PARAKEET_FILES[..3] {
            fs::write(dir.join(file_name), b"x").unwrap();
        }
        assert!(!parakeet_present(&models));
        fs::write(dir.join(PARAKEET_FILES[3].0), b"x").unwrap();
        assert!(parakeet_present(&models));
        let _ = fs::remove_dir_all(&models);
    }

    #[test]
    fn old_settings_files_get_the_new_defaults() {
        let old = r#"{ "whisperModel": "small", "keepAudio": true }"#;
        let settings: Settings = serde_json::from_str(old).unwrap();
        assert_eq!(settings.whisper_model, "small");
        assert!(settings.keep_audio);
        assert_eq!(settings.transcriber, TranscriberKind::Whisper);
        assert!(settings.skip_silence);
        assert!(settings.live_transcript);
    }

    #[test]
    fn transcriber_settings_round_trip() {
        let settings = Settings {
            transcriber: TranscriberKind::Parakeet,
            skip_silence: false,
            ..Settings::default()
        };
        let json = serde_json::to_value(&settings).unwrap();
        assert_eq!(json["transcriber"], "parakeet");
        assert_eq!(json["skipSilence"], false);
        let back: Settings = serde_json::from_value(json).unwrap();
        assert_eq!(back.transcriber, TranscriberKind::Parakeet);
        assert!(!back.skip_silence);
        assert_eq!(serde_json::to_value(Settings::default()).unwrap()["transcriber"], "whisper");
    }
}
