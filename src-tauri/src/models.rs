use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime};

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
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            whisper_model: "large-v3-turbo-q5_0".into(),
            keep_audio: false,
            ollama_url: "http://localhost:11434".into(),
            ollama_model: None,
        }
    }
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
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf, String> {
    fs::create_dir_all(models_dir).map_err(|e| e.to_string())?;
    let file_name = format!("ggml-{model_name}.bin");
    let target = models_dir.join(&file_name);
    if target.exists() {
        return Ok(target);
    }

    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file_name}");
    let mut response = ureq::get(&url)
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
    fs::rename(&part, &target).map_err(|e| e.to_string())?;
    Ok(target)
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
    let result = download_model(&models_dir, &settings.whisper_model, |received, total| {
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
