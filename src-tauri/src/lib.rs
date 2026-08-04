pub mod calendar;
pub mod models;
pub mod ollama;
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
    Distilling { dir: String },
    Error { message: String },
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct UpcomingMeeting {
    title: String,
    starts_at_display: String,
}

struct AppState {
    session: Option<RecordingSession>,
    last_state: UiState,
    upcoming: Option<UpcomingMeeting>,
    prompted_meetings: std::collections::HashSet<String>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            session: None,
            last_state: UiState::Idle { last_meeting: None },
            upcoming: None,
            prompted_meetings: std::collections::HashSet::new(),
        }
    }
}

struct RecordingSession {
    child: CommandChild,
}

fn build_tray_menu<R: Runtime>(
    app: &AppHandle<R>,
    state: &UiState,
    upcoming: Option<&UpcomingMeeting>,
) -> tauri::Result<Menu<R>> {
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

    let recording = matches!(state, UiState::Recording { .. } | UiState::Stopping);
    if let (Some(meeting), false) = (upcoming, recording) {
        let record_upcoming = MenuItem::with_id(
            app,
            "record-upcoming",
            format!("Record: {} ({})", meeting.title, meeting.starts_at_display),
            true,
            None::<&str>,
        )?;
        let sep0 = PredefinedMenuItem::separator(app)?;
        return Menu::with_items(app, &[&record_upcoming, &sep0, &toggle, &sep1, &open, &sep2, &quit]);
    }
    Menu::with_items(app, &[&toggle, &sep1, &open, &sep2, &quit])
}

/// Single place that mutates UI-visible state: stores it, refreshes the tray,
/// and notifies the frontend. Callers must not hold the state lock.
fn emit_state<R: Runtime>(app: &AppHandle<R>, state: UiState) {
    let upcoming = {
        let handle = app.state::<Mutex<AppState>>();
        let mut guard = handle.lock().unwrap();
        guard.last_state = state.clone();
        guard.upcoming.clone()
    };

    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        if let Ok(menu) = build_tray_menu(app, &state, upcoming.as_ref()) {
            let _ = tray.set_menu(Some(menu));
        }
        let tooltip = match state {
            UiState::Recording { .. } => "Distil — recording",
            UiState::Transcribing { .. } => "Distil — transcribing",
            UiState::Distilling { .. } => "Distil — distilling",
            _ => "Distil",
        };
        let _ = tray.set_tooltip(Some(tooltip));
        // Menu-bar text beside the icon: recording / working indicator.
        let title = match state {
            UiState::Recording { .. } => Some("●"),
            UiState::Transcribing { .. } | UiState::Distilling { .. } => Some("⋯"),
            _ => None,
        };
        let _ = tray.set_title(title);
    }

    let _ = app.emit("pipeline-state", &state);
}

fn notify<R: Runtime>(app: &AppHandle<R>, title: &str, body: &str) {
    use tauri_plugin_notification::NotificationExt;
    let _ = app.notification().builder().title(title).body(body).show();
}

fn meeting_label(dir: &Path) -> String {
    dir.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.display().to_string())
}

fn meetings_root<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    let docs = app
        .path()
        .document_dir()
        .map_err(|e| format!("cannot resolve Documents dir: {e}"))?;
    Ok(docs.join("Distil"))
}

fn start_recording<R: Runtime>(app: &AppHandle<R>, title: Option<&str>) {
    if app.state::<Mutex<AppState>>().lock().unwrap().session.is_some() {
        return;
    }

    let title = title.map(calendar::sanitize_title).unwrap_or_else(|| "Meeting".to_string());
    let spawned = (|| -> Result<_, String> {
        let dir = meetings_root(app)?.join(format!(
            "{} {}",
            chrono::Local::now().format("%Y-%m-%d %H.%M"),
            title
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

/// Serialises heavy pipeline jobs (whisper, Ollama) so only one model is ever
/// resident — 8 GB machines are a design target.
static PIPELINE_GATE: Mutex<()> = Mutex::new(());

fn pipeline_owns_ui<R: Runtime>(app: &AppHandle<R>, dir: &Path) -> bool {
    let state = app.state::<Mutex<AppState>>();
    let guard = state.lock().unwrap();
    let dir = dir.display().to_string();
    matches!(
        &guard.last_state,
        UiState::Transcribing { dir: d } | UiState::Distilling { dir: d } if *d == dir
    )
}

/// Transcript → notes.md via the configured (or best local) Ollama model.
/// Blocking; call while holding PIPELINE_GATE.
fn distil_core<R: Runtime>(app: &AppHandle<R>, dir: &Path) -> Result<(), String> {
    let transcript = std::fs::read_to_string(dir.join("transcript.md"))
        .map_err(|e| format!("cannot read transcript: {e}"))?;
    let settings = models::load_settings(app);
    let model = match settings.ollama_model.clone() {
        Some(model) => {
            if model.ends_with("-cloud") && !settings.allow_cloud_models {
                return Err(format!(
                    "{model} is a cloud model but cloud models are disabled — enable them in Settings or pick a local model"
                ));
            }
            model
        }
        None => ollama::auto_pick(&settings.ollama_url)?,
    };
    let notes = ollama::generate(
        &settings.ollama_url,
        &model,
        &ollama::build_notes_prompt(&transcript),
    )?;
    if notes.is_empty() {
        return Err(format!("{model} returned empty notes"));
    }
    std::fs::write(dir.join("notes.md"), format!("{notes}\n")).map_err(|e| e.to_string())?;

    let meta_path = dir.join("meta.json");
    if let Ok(bytes) = std::fs::read(&meta_path) {
        if let Ok(mut meta) = serde_json::from_slice::<Value>(&bytes) {
            meta["ollamaModel"] = Value::from(model);
            meta["distilledAt"] = Value::from(chrono::Local::now().to_rfc3339());
            let _ = std::fs::write(
                &meta_path,
                serde_json::to_vec_pretty(&meta).unwrap_or_default(),
            );
        }
    }
    Ok(())
}

fn run_transcription<R: Runtime>(app: &AppHandle<R>, dir: &Path) {
    let _gate = PIPELINE_GATE.lock().unwrap();

    let result = (|| -> Result<(), String> {
        let settings = models::load_settings(app);
        let model = models::ensure_whisper_model(app)?;

        // A stream whose peak never left the noise floor is dead — skip it
        // rather than let whisper hallucinate over silence.
        const SILENCE_PEAK: i64 = 100;
        let capture = std::fs::read(dir.join("meta.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .and_then(|meta| meta.get("capture").cloned());
        let peak_of = |key: &str| {
            capture
                .as_ref()
                .and_then(|c| c.get(key))
                .and_then(Value::as_i64)
        };
        let mic_silent = peak_of("micPeak").is_some_and(|p| p < SILENCE_PEAK);
        let system_silent = peak_of("systemPeak").is_some_and(|p| p < SILENCE_PEAK);

        let me = if mic_silent {
            Vec::new()
        } else {
            transcribe::transcribe(&model, &dir.join("audio-mic.wav"))?
        };
        let them = if system_silent {
            Vec::new()
        } else {
            transcribe::transcribe(&model, &dir.join("audio-system.wav"))?
        };
        let me = transcribe::drop_mic_bleed(me, &them);
        let mut transcript = transcribe::merge_transcript(me, them);
        if mic_silent {
            transcript.push_str("\n_Note: the microphone stream was silent for this recording._\n");
        }
        if system_silent {
            transcript.push_str("\n_Note: the system-audio stream was silent for this recording._\n");
        }
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

    match result {
        Ok(()) => {
            // Transcript is on disk — let the library refresh, then distil.
            let _ = app.emit("meeting-ready", dir.display().to_string());
            if pipeline_owns_ui(app, dir) {
                emit_state(app, UiState::Distilling { dir: dir.display().to_string() });
            }
            match distil_core(app, dir) {
                Ok(()) => {
                    let _ = app.emit("meeting-ready", dir.display().to_string());
                    notify(app, "Notes ready", &meeting_label(dir));
                    if pipeline_owns_ui(app, dir) {
                        emit_state(
                            app,
                            UiState::Idle { last_meeting: Some(dir.display().to_string()) },
                        );
                    }
                }
                Err(message) => {
                    let message = format!(
                        "notes failed: {message} — the transcript is saved; use Generate notes to retry"
                    );
                    notify(app, "Distil — notes failed", &meeting_label(dir));
                    if pipeline_owns_ui(app, dir) {
                        emit_state(app, UiState::Error { message });
                    } else {
                        eprintln!("{message}");
                    }
                }
            }
        }
        Err(message) => {
            let message = format!("{message} — audio kept in {}", dir.display());
            notify(app, "Distil — transcription failed", &meeting_label(dir));
            if pipeline_owns_ui(app, dir) {
                emit_state(app, UiState::Error { message });
            } else {
                eprintln!("transcription failed: {message}");
            }
        }
    }
}

/// Quit without ever orphaning a recording: ask the sidecar to stop, give it a
/// moment to finalise the WAVs and for the reader task to write meta.json,
/// then exit. (A capture interrupted this way keeps its audio; transcription
/// can be re-run via Generate notes once a transcript exists — see PLAN M4.)
fn quit_app<R: Runtime>(app: &AppHandle<R>) {
    let recording = app.state::<Mutex<AppState>>().lock().unwrap().session.is_some();
    if !recording {
        app.exit(0);
        return;
    }
    stop_recording(app);
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        for _ in 0..40 {
            if app.state::<Mutex<AppState>>().lock().unwrap().session.is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        // Grace period: session clears just before meta.json is written.
        std::thread::sleep(std::time::Duration::from_millis(300));
        app.exit(0);
    });
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
        start_recording(app, None);
    }
}

/// Start recording the meeting the calendar watcher surfaced.
fn record_upcoming_impl<R: Runtime>(app: &AppHandle<R>) {
    let title = app
        .state::<Mutex<AppState>>()
        .lock()
        .unwrap()
        .upcoming
        .as_ref()
        .map(|m| m.title.clone());
    start_recording(app, title.as_deref());
}

const CALENDAR_POLL_SECS: u64 = 60;
/// Prompt window: from 2 minutes before a meeting starts until 10 minutes in.
const LEAD_SECS: i64 = 120;
const LATE_SECS: i64 = 600;

fn start_calendar_watcher<R: Runtime>(app: AppHandle<R>) {
    std::thread::spawn(move || loop {
        poll_calendar(&app);
        std::thread::sleep(std::time::Duration::from_secs(CALENDAR_POLL_SECS));
    });
}

fn poll_calendar<R: Runtime>(app: &AppHandle<R>) {
    let Ok(command) = app.shell().sidecar("distil-capture") else {
        return;
    };
    let Ok(output) =
        tauri::async_runtime::block_on(command.args(["calendar", "--hours", "12"]).output())
    else {
        return;
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let Some(report) = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|v| v.get("event").and_then(Value::as_str) == Some("calendar"))
    else {
        return;
    };
    if report.get("authorized").and_then(Value::as_bool) != Some(true) {
        return;
    }
    let events: Vec<calendar::CalendarEvent> =
        serde_json::from_value(report.get("events").cloned().unwrap_or_default())
            .unwrap_or_default();

    let now = chrono::Local::now();
    let candidate = events
        .iter()
        .filter(|e| calendar::is_real_meeting(e))
        .filter_map(|e| {
            let starts = chrono::DateTime::parse_from_rfc3339(&e.starts_at).ok()?;
            let delta = starts.signed_duration_since(now).num_seconds();
            (-LATE_SECS..=LEAD_SECS).contains(&delta).then_some((e, starts, delta))
        })
        .min_by_key(|(_, _, delta)| *delta);

    let upcoming = candidate.map(|(event, starts, _)| UpcomingMeeting {
        title: event.title.clone(),
        starts_at_display: starts.with_timezone(&chrono::Local).format("%H:%M").to_string(),
    });

    let (changed, should_notify, current_state) = {
        let handle = app.state::<Mutex<AppState>>();
        let mut guard = handle.lock().unwrap();
        let changed = match (&guard.upcoming, &upcoming) {
            (None, None) => false,
            (Some(a), Some(b)) => a.title != b.title || a.starts_at_display != b.starts_at_display,
            _ => true,
        };
        guard.upcoming = upcoming.clone();
        let should_notify = match (&upcoming, guard.session.is_some()) {
            (Some(meeting), false) => {
                let key = format!("{}|{}", meeting.title, meeting.starts_at_display);
                guard.prompted_meetings.insert(key)
            }
            _ => false,
        };
        (changed, should_notify, guard.last_state.clone())
    };

    if changed {
        // Rebuild the tray so the "Record: …" item appears/disappears.
        emit_state(app, current_state);
        let _ = app.emit("upcoming-meeting", &upcoming);
    }
    if should_notify {
        if let Some(meeting) = &upcoming {
            notify(
                app,
                &format!("{} at {}", meeting.title, meeting.starts_at_display),
                "Click the Distil menu-bar icon to record this meeting.",
            );
        }
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

#[tauri::command]
fn record_upcoming(app: AppHandle) {
    record_upcoming_impl(&app);
}

#[tauri::command]
fn upcoming_meeting(state: tauri::State<'_, Mutex<AppState>>) -> Option<UpcomingMeeting> {
    state.lock().unwrap().upcoming.clone()
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct MeetingSummary {
    dir: String,
    name: String,
    started_at: Option<String>,
    has_notes: bool,
    has_transcript: bool,
}

#[tauri::command]
fn list_meetings(app: AppHandle) -> Result<Vec<MeetingSummary>, String> {
    let root = meetings_root(&app)?;
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Ok(Vec::new()); // nothing recorded yet
    };
    let mut meetings: Vec<MeetingSummary> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .map(|path| MeetingSummary {
            name: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            started_at: std::fs::read(path.join("meta.json"))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .and_then(|meta| {
                    meta.get("startedAt").and_then(Value::as_str).map(String::from)
                }),
            has_notes: path.join("notes.md").exists(),
            has_transcript: path.join("transcript.md").exists(),
            dir: path.display().to_string(),
        })
        .collect();
    meetings.sort_by(|a, b| b.started_at.cmp(&a.started_at).then(b.name.cmp(&a.name)));
    Ok(meetings)
}

/// Webview-supplied paths are only trusted once proven to live inside the
/// meetings root.
fn validated_meeting_dir<R: Runtime>(app: &AppHandle<R>, dir: &str) -> Result<PathBuf, String> {
    let root = meetings_root(app)?
        .canonicalize()
        .map_err(|e| format!("meetings folder missing: {e}"))?;
    let path = PathBuf::from(dir)
        .canonicalize()
        .map_err(|e| format!("no such meeting: {e}"))?;
    if !path.starts_with(&root) || !path.is_dir() {
        return Err("not a distil meeting folder".into());
    }
    Ok(path)
}

#[tauri::command]
fn read_meeting(app: AppHandle, dir: String) -> Result<Value, String> {
    let path = validated_meeting_dir(&app, &dir)?;
    let read = |name: &str| std::fs::read_to_string(path.join(name)).ok();
    Ok(serde_json::json!({
        "notes": read("notes.md"),
        "transcript": read("transcript.md"),
        "meta": std::fs::read(path.join("meta.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok()),
    }))
}

#[tauri::command]
fn distil_meeting(app: AppHandle, dir: String) -> Result<(), String> {
    let path = validated_meeting_dir(&app, &dir)?;
    tauri::async_runtime::spawn_blocking(move || {
        let _gate = PIPELINE_GATE.lock().unwrap();
        let can_own_ui = {
            let state = app.state::<Mutex<AppState>>();
            let guard = state.lock().unwrap();
            matches!(guard.last_state, UiState::Idle { .. } | UiState::Error { .. })
        };
        if can_own_ui {
            emit_state(&app, UiState::Distilling { dir: path.display().to_string() });
        }
        match distil_core(&app, &path) {
            Ok(()) => {
                let _ = app.emit("meeting-ready", path.display().to_string());
                notify(&app, "Notes ready", &meeting_label(&path));
                if pipeline_owns_ui(&app, &path) {
                    emit_state(
                        &app,
                        UiState::Idle { last_meeting: Some(path.display().to_string()) },
                    );
                }
            }
            Err(message) => {
                let message = format!("notes failed: {message}");
                if pipeline_owns_ui(&app, &path) {
                    emit_state(&app, UiState::Error { message });
                } else {
                    eprintln!("{message}");
                }
            }
        }
    });
    Ok(())
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct OllamaModelInfo {
    name: String,
    parameter_size: Option<f64>,
    is_cloud: bool,
}

#[tauri::command]
fn list_ollama_models(app: AppHandle) -> Result<Vec<OllamaModelInfo>, String> {
    let settings = models::load_settings(&app);
    Ok(
        ollama::list_models(&settings.ollama_url, settings.allow_cloud_models)?
            .into_iter()
            .map(|m| OllamaModelInfo {
                name: m.name,
                parameter_size: m.parameter_size,
                is_cloud: m.is_cloud,
            })
            .collect(),
    )
}

#[tauri::command]
fn set_allow_cloud_models(app: AppHandle, allow: bool) -> Result<(), String> {
    let mut settings = models::load_settings(&app);
    settings.allow_cloud_models = allow;
    // Disabling the opt-in must also drop any cloud model still selected.
    if !allow && settings.ollama_model.as_deref().is_some_and(|m| m.ends_with("-cloud")) {
        settings.ollama_model = None;
    }
    models::save_settings(&app, &settings)
}

#[tauri::command]
fn get_settings(app: AppHandle) -> models::Settings {
    models::load_settings(&app)
}

#[tauri::command]
fn set_ollama_model(app: AppHandle, model: Option<String>) -> Result<(), String> {
    let mut settings = models::load_settings(&app);
    settings.ollama_model = model.filter(|m| !m.is_empty());
    models::save_settings(&app, &settings)
}

/// Curated whisper choices — name in ggerganov/whisper.cpp, blurb, size on disk.
const WHISPER_MODELS: &[(&str, &str, u64)] = &[
    ("large-v3-turbo-q5_0", "Best balance of accuracy and speed (default)", 574),
    ("large-v3-q5_0", "Maximum accuracy, slower", 1_081),
    ("small", "Faster, lighter accuracy", 466),
    ("tiny.en-q8_0", "Fastest, English-only — for quick tests", 43),
];

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct WhisperModelInfo {
    name: String,
    description: String,
    size_mb: u64,
    downloaded: bool,
    selected: bool,
}

#[tauri::command]
fn list_whisper_models(app: AppHandle) -> Result<Vec<WhisperModelInfo>, String> {
    let settings = models::load_settings(&app);
    let models_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models");
    Ok(WHISPER_MODELS
        .iter()
        .map(|(name, description, size_mb)| WhisperModelInfo {
            name: (*name).to_string(),
            description: (*description).to_string(),
            size_mb: *size_mb,
            downloaded: models_dir.join(format!("ggml-{name}.bin")).exists(),
            selected: settings.whisper_model == *name,
        })
        .collect())
}

#[tauri::command]
fn set_whisper_model(app: AppHandle, model: String) -> Result<(), String> {
    if !WHISPER_MODELS.iter().any(|(name, ..)| *name == model) {
        return Err("unknown whisper model".into());
    }
    let mut settings = models::load_settings(&app);
    settings.whisper_model = model;
    models::save_settings(&app, &settings)
}

#[tauri::command]
fn set_keep_audio(app: AppHandle, keep: bool) -> Result<(), String> {
    let mut settings = models::load_settings(&app);
    settings.keep_audio = keep;
    models::save_settings(&app, &settings)
}

#[tauri::command]
async fn permissions_status(app: AppHandle) -> Result<Value, String> {
    let output = app
        .shell()
        .sidecar("distil-capture")
        .map_err(|e| e.to_string())?
        .args(["--check-permissions"])
        .output()
        .await
        .map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|v| v.get("event").and_then(Value::as_str) == Some("permissions"))
        .ok_or_else(|| "capture helper produced no permissions report".to_string())
}

const PRIVACY_PANES: &[(&str, &str)] = &[
    (
        "microphone",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    ),
    (
        "screen",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    ),
    (
        "calendar",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Calendars",
    ),
];

#[tauri::command]
fn open_privacy_settings(pane: String) -> Result<(), String> {
    let url = PRIVACY_PANES
        .iter()
        .find(|(key, _)| *key == pane)
        .map(|(_, url)| *url)
        .ok_or("unknown settings pane")?;
    tauri_plugin_opener::open_url(url, None::<String>).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .manage(Mutex::new(AppState::default()))
        .invoke_handler(tauri::generate_handler![
            current_state,
            toggle_recording,
            record_upcoming,
            upcoming_meeting,
            list_meetings,
            read_meeting,
            distil_meeting,
            list_ollama_models,
            get_settings,
            set_ollama_model,
            set_allow_cloud_models,
            list_whisper_models,
            set_whisper_model,
            set_keep_audio,
            permissions_status,
            open_privacy_settings
        ])
        .setup(|app| {
            // Menu-bar app: no Dock icon, lives in the tray.
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let initial = UiState::Idle { last_meeting: None };
            let menu = build_tray_menu(app.handle(), &initial, None)?;
            TrayIconBuilder::with_id(TRAY_ID)
                .icon(app.default_window_icon().unwrap().clone())
                .icon_as_template(true)
                .menu(&menu)
                .show_menu_on_left_click(true)
                .tooltip("Distil")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "toggle-recording" => toggle_recording_impl(app),
                    "record-upcoming" => record_upcoming_impl(app),
                    "open-library" => show_library(app),
                    "quit" => quit_app(app),
                    _ => {}
                })
                .build(app)?;

            start_calendar_watcher(app.handle().clone());

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
