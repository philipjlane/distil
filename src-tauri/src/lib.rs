pub mod models;
pub mod transcribe;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::Value;
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
    AppHandle, Emitter, Manager, Runtime,
};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

const TRAY_ID: &str = "distil-tray";

/// UI-facing pipeline state, pushed via the `pipeline-state` event.
/// M2/M3 will add `transcribing` / `distilling` phases.
#[derive(Clone, serde::Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
enum UiState {
    Idle { last_meeting: Option<String> },
    Recording { dir: String },
    Stopping,
    Transcribing { dir: String },
    Error { message: String },
}

struct AppState {
    session: Option<RecordingSession>,
    last_state: UiState,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            session: None,
            last_state: UiState::Idle { last_meeting: None },
        }
    }
}

struct RecordingSession {
    child: CommandChild,
}

fn build_tray_menu<R: Runtime>(app: &AppHandle<R>, state: &UiState) -> tauri::Result<Menu<R>> {
    let (toggle_label, toggle_enabled) = match state {
        UiState::Recording { .. } => ("Stop Recording", true),
        UiState::Stopping => ("Stopping…", false),
        _ => ("Start Recording", true),
    };
    let toggle = MenuItem::with_id(app, "toggle-recording", toggle_label, toggle_enabled, None::<&str>)?;
    let open = MenuItem::with_id(app, "open-library", "Open Distil", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Distil", true, None::<&str>)?;
    let sep1 = PredefinedMenuItem::separator(app)?;
    let sep2 = PredefinedMenuItem::separator(app)?;
    Menu::with_items(app, &[&toggle, &sep1, &open, &sep2, &quit])
}

/// Single place that mutates UI-visible state: stores it, refreshes the tray,
/// and notifies the frontend. Callers must not hold the state lock.
fn emit_state<R: Runtime>(app: &AppHandle<R>, state: UiState) {
    app.state::<Mutex<AppState>>().lock().unwrap().last_state = state.clone();

    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        if let Ok(menu) = build_tray_menu(app, &state) {
            let _ = tray.set_menu(Some(menu));
        }
        let tooltip = match state {
            UiState::Recording { .. } => "Distil — recording",
            UiState::Transcribing { .. } => "Distil — transcribing",
            _ => "Distil",
        };
        let _ = tray.set_tooltip(Some(tooltip));
    }

    let _ = app.emit("pipeline-state", &state);
}

fn meetings_root<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    let docs = app
        .path()
        .document_dir()
        .map_err(|e| format!("cannot resolve Documents dir: {e}"))?;
    Ok(docs.join("Distil"))
}

fn start_recording<R: Runtime>(app: &AppHandle<R>) {
    if app.state::<Mutex<AppState>>().lock().unwrap().session.is_some() {
        return;
    }

    let spawned = (|| -> Result<_, String> {
        let dir = meetings_root(app)?.join(format!(
            "{} Meeting",
            chrono::Local::now().format("%Y-%m-%d %H.%M")
        ));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let command = app
            .shell()
            .sidecar("distil-capture")
            .map_err(|e| e.to_string())?
            .args(["--out-dir", &dir.to_string_lossy()]);
        let (rx, child) = command.spawn().map_err(|e| e.to_string())?;
        Ok((dir, rx, child))
    })();

    let (dir, mut rx, child) = match spawned {
        Ok(parts) => parts,
        Err(message) => {
            emit_state(app, UiState::Error { message });
            return;
        }
    };

    app.state::<Mutex<AppState>>().lock().unwrap().session =
        Some(RecordingSession { child });
    emit_state(app, UiState::Recording { dir: dir.display().to_string() });

    let app = app.clone();
    let started_at = chrono::Local::now();
    tauri::async_runtime::spawn(async move {
        let mut summary: Option<Value> = None;
        let mut error: Option<String> = None;

        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    for parsed in text
                        .lines()
                        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    {
                        match parsed.get("event").and_then(Value::as_str) {
                            Some("stopped") => summary = Some(parsed.clone()),
                            Some("error") => {
                                error = parsed
                                    .get("message")
                                    .and_then(Value::as_str)
                                    .map(String::from)
                            }
                            _ => {}
                        }
                        let _ = app.emit("capture-event", &parsed);
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    eprintln!("distil-capture: {}", String::from_utf8_lossy(&bytes));
                }
                CommandEvent::Terminated(status) => {
                    app.state::<Mutex<AppState>>().lock().unwrap().session = None;
                    let clean_exit = status.code == Some(0) && error.is_none();
                    if clean_exit {
                        let meta = serde_json::json!({
                            "startedAt": started_at.to_rfc3339(),
                            "stoppedAt": chrono::Local::now().to_rfc3339(),
                            "capture": summary,
                            "appVersion": env!("CARGO_PKG_VERSION"),
                        });
                        let _ = std::fs::write(
                            dir.join("meta.json"),
                            serde_json::to_vec_pretty(&meta).unwrap_or_default(),
                        );
                        emit_state(
                            &app,
                            UiState::Transcribing { dir: dir.display().to_string() },
                        );
                        let app = app.clone();
                        let dir = dir.clone();
                        tauri::async_runtime::spawn_blocking(move || {
                            run_transcription(&app, &dir);
                        });
                    } else {
                        // Nothing usable was captured; don't leave an empty husk.
                        let _ = std::fs::remove_dir_all(&dir);
                        emit_state(
                            &app,
                            UiState::Error {
                                message: error.unwrap_or_else(|| {
                                    format!("capture exited unexpectedly (status {:?})", status.code)
                                }),
                            },
                        );
                    }
                    break;
                }
                _ => {}
            }
        }
    });
}

/// Serialises transcription jobs so only one whisper model is ever resident —
/// 8 GB machines are a design target.
static TRANSCRIBE_GATE: Mutex<()> = Mutex::new(());

fn run_transcription<R: Runtime>(app: &AppHandle<R>, dir: &Path) {
    let _gate = TRANSCRIBE_GATE.lock().unwrap();

    let result = (|| -> Result<(), String> {
        let settings = models::load_settings(app);
        let model = models::ensure_whisper_model(app)?;
        let me = transcribe::transcribe(&model, &dir.join("audio-mic.wav"))?;
        let them = transcribe::transcribe(&model, &dir.join("audio-system.wav"))?;
        let transcript = transcribe::merge_transcript(me, them);
        std::fs::write(dir.join("transcript.md"), &transcript).map_err(|e| e.to_string())?;

        let meta_path = dir.join("meta.json");
        if let Ok(bytes) = std::fs::read(&meta_path) {
            if let Ok(mut meta) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                meta["whisperModel"] = serde_json::Value::from(settings.whisper_model.clone());
                meta["transcribedAt"] =
                    serde_json::Value::from(chrono::Local::now().to_rfc3339());
                let _ = std::fs::write(
                    &meta_path,
                    serde_json::to_vec_pretty(&meta).unwrap_or_default(),
                );
            }
        }

        if !settings.keep_audio {
            let _ = std::fs::remove_file(dir.join("audio-mic.wav"));
            let _ = std::fs::remove_file(dir.join("audio-system.wav"));
        }
        Ok(())
    })();

    // Don't clobber the UI if a new recording started while we transcribed.
    let owns_ui = {
        let state = app.state::<Mutex<AppState>>();
        let guard = state.lock().unwrap();
        matches!(&guard.last_state, UiState::Transcribing { dir: d } if *d == dir.display().to_string())
    };

    match result {
        Ok(()) => {
            let _ = app.emit("meeting-ready", dir.display().to_string());
            if owns_ui {
                emit_state(
                    app,
                    UiState::Idle { last_meeting: Some(dir.display().to_string()) },
                );
            }
        }
        Err(message) => {
            let message = format!("{message} — audio kept in {}", dir.display());
            if owns_ui {
                emit_state(app, UiState::Error { message });
            } else {
                eprintln!("transcription failed: {message}");
            }
        }
    }
}

fn stop_recording<R: Runtime>(app: &AppHandle<R>) {
    let session = app.state::<Mutex<AppState>>().lock().unwrap().session.take();
    if let Some(mut session) = session {
        emit_state(app, UiState::Stopping);
        if session.child.write(b"stop\n").is_err() {
            let _ = session.child.kill();
        }
        // The reader task emits the final Idle/Error state when the sidecar exits.
    }
}

fn toggle_recording_impl<R: Runtime>(app: &AppHandle<R>) {
    let (recording, stopping) = {
        let state = app.state::<Mutex<AppState>>();
        let guard = state.lock().unwrap();
        (
            guard.session.is_some(),
            matches!(guard.last_state, UiState::Stopping),
        )
    };
    if stopping {
        return;
    }
    if recording {
        stop_recording(app);
    } else {
        start_recording(app);
    }
}

fn show_library<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

#[tauri::command]
fn current_state(state: tauri::State<'_, Mutex<AppState>>) -> UiState {
    state.lock().unwrap().last_state.clone()
}

#[tauri::command]
fn toggle_recording(app: AppHandle) {
    toggle_recording_impl(&app);
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .manage(Mutex::new(AppState::default()))
        .invoke_handler(tauri::generate_handler![current_state, toggle_recording])
        .setup(|app| {
            // Menu-bar app: no Dock icon, lives in the tray.
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let initial = UiState::Idle { last_meeting: None };
            let menu = build_tray_menu(app.handle(), &initial)?;
            TrayIconBuilder::with_id(TRAY_ID)
                .icon(app.default_window_icon().unwrap().clone())
                .icon_as_template(true)
                .menu(&menu)
                .show_menu_on_left_click(true)
                .tooltip("Distil")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "toggle-recording" => toggle_recording_impl(app),
                    "open-library" => show_library(app),
                    "quit" => {
                        // Never orphan a running capture.
                        stop_recording(app);
                        app.exit(0);
                    }
                    _ => {}
                })
                .build(app)?;

            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the library window hides it; the app lives in the tray.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
