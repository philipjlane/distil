use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime};

use crate::llm::Engine;

/// User-tunable settings, stored as settings.json in the app data dir.
/// The Settings UI (M4) is sugar over this file.
#[derive(Serialize, Deserialize, Clone)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    /// whisper.cpp model name as published in ggerganov/whisper.cpp on
    /// Hugging Face, without the "ggml-" prefix or ".bin" suffix.
    pub whisper_model: String,
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
            keep_audio: false,
            ollama_url: "http://localhost:11434".into(),
            ollama_model: None,
            engine: Engine::Ollama,
            openai_url: "http://localhost:8080/v1".into(),
            openai_model: None,
            allow_cloud_models: false,
            diarize_speakers: true,
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
    let settings = load_settings(app);
    let models_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("cannot resolve app data dir: {e}"))?
        .join("models");

    let already_present = models_dir
        .join(format!("ggml-{}.bin", settings.whisper_model))
        .exists();
    if !already_present {
        let _ = app.emit(
            "model-download",
            serde_json::json!({ "status": "started", "model": settings.whisper_model }),
        );
    }

    let mut last_percent = -1i64;
    let result = download_model(&models_dir, &settings.whisper_model, |received, total: Option<u64>| {
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

    if !already_present {
        let status = if result.is_ok() { "done" } else { "failed" };
        let _ = app.emit("model-download", serde_json::json!({ "status": status }));
    }
    result
}
