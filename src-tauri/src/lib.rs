pub mod calendar;
pub mod diarize;
pub mod models;
pub mod ollama;
pub mod transcribe;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::Value;
use tauri::{
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
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
    #[serde(skip)]
    attendees: Vec<String>,
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
    started: std::time::Instant,
}

/// Show/hide the tray panel window, positioned under the tray icon.
fn toggle_tray_panel<R: Runtime>(app: &AppHandle<R>, rect: tauri::Rect) {
    let Some(panel) = app.get_webview_window("tray") else {
        return;
    };
    if panel.is_visible().unwrap_or(false) {
        let _ = panel.hide();
        return;
    }
    let scale = panel.scale_factor().unwrap_or(2.0);
    let (icon_x, icon_y) = match rect.position {
        tauri::Position::Physical(p) => (f64::from(p.x), f64::from(p.y)),
        tauri::Position::Logical(l) => (l.x * scale, l.y * scale),
    };
    let (icon_w, icon_h) = match rect.size {
        tauri::Size::Physical(s) => (f64::from(s.width), f64::from(s.height)),
        tauri::Size::Logical(l) => (l.width * scale, l.height * scale),
    };
    let panel_width = 300.0 * scale;
    let x = icon_x + icon_w / 2.0 - panel_width / 2.0;
    let y = icon_y + icon_h + 6.0 * scale;
    let _ = panel.set_position(tauri::PhysicalPosition::new(x as i32, y as i32));
    let _ = panel.show();
    let _ = panel.set_focus();
}

/// Single place that mutates UI-visible state: stores it, refreshes the tray,
/// and notifies the frontend. Callers must not hold the state lock.
fn emit_state<R: Runtime>(app: &AppHandle<R>, state: UiState) {
    app.state::<Mutex<AppState>>().lock().unwrap().last_state = state.clone();

    if let Some(tray) = app.tray_by_id(TRAY_ID) {
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

fn start_recording<R: Runtime>(app: &AppHandle<R>, title: Option<&str>, attendees: Vec<String>) {
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

    app.state::<Mutex<AppState>>().lock().unwrap().session = Some(RecordingSession {
        child,
        started: std::time::Instant::now(),
    });
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
                            "attendees": attendees,
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
    // Attendee names (when the recording came from a calendar prompt) give the
    // name-inference instruction its candidate set.
    let attendees: Vec<String> = std::fs::read(dir.join("meta.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|meta| meta.get("attendees").cloned())
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default();
    let transcript = if attendees.is_empty() {
        transcript
    } else {
        format!(
            "Known attendees (from the calendar invite): {}.\n\n{transcript}",
            attendees.join(", ")
        )
    };

    let (template, _) = models::load_notes_prompt(app);
    let notes = ollama::generate(
        &settings.ollama_url,
        &model,
        &ollama::build_notes_prompt(&template, &transcript),
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

        // Optional speaker fidelity: cluster the system stream into Them 1..N.
        // Diarization failure must never sink the pipeline — fall back to "Them".
        let them_labeled: Vec<(String, transcribe::Segment)> =
            if settings.diarize_speakers && !system_silent && !them.is_empty() {
                let diarized = models::ensure_diarization_models(app).and_then(
                    |(segmentation, embedding)| {
                        diarize::diarize_wav(
                            &segmentation,
                            &embedding,
                            &dir.join("audio-system.wav"),
                        )
                    },
                );
                match diarized {
                    Ok(spans) => diarize::label_them_segments(them, &spans),
                    Err(message) => {
                        eprintln!("diarization skipped: {message}");
                        them.into_iter().map(|s| ("Them".to_string(), s)).collect()
                    }
                }
            } else {
                them.into_iter().map(|s| ("Them".to_string(), s)).collect()
            };

        let mut labeled: Vec<(String, transcribe::Segment)> =
            me.into_iter().map(|s| ("Me".to_string(), s)).collect();
        labeled.extend(them_labeled);
        let mut transcript = transcribe::merge_labeled(labeled);
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
        start_recording(app, None, Vec::new());
    }
}

/// Start recording the meeting the calendar watcher surfaced.
fn record_upcoming_impl<R: Runtime>(app: &AppHandle<R>) {
    let upcoming = app
        .state::<Mutex<AppState>>()
        .lock()
        .unwrap()
        .upcoming
        .clone();
    match upcoming {
        Some(meeting) => start_recording(app, Some(&meeting.title), meeting.attendees),
        None => start_recording(app, None, Vec::new()),
    }
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
        attendees: event.attendees.clone(),
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

    let _ = &current_state;
    if changed {
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

#[tauri::command]
fn recording_elapsed(state: tauri::State<'_, Mutex<AppState>>) -> Option<u64> {
    state
        .lock()
        .unwrap()
        .session
        .as_ref()
        .map(|s| s.started.elapsed().as_secs())
}

fn hide_tray_panel<R: Runtime>(app: &AppHandle<R>) {
    if let Some(panel) = app.get_webview_window("tray") {
        let _ = panel.hide();
    }
}

#[tauri::command]
fn open_meeting(app: AppHandle, dir: String) -> Result<(), String> {
    let path = validated_meeting_dir(&app, &dir)?;
    hide_tray_panel(&app);
    show_library(&app);
    let _ = app.emit("open-meeting", path.display().to_string());
    Ok(())
}

#[tauri::command]
fn open_settings(app: AppHandle) {
    hide_tray_panel(&app);
    show_library(&app);
    let _ = app.emit("open-settings", ());
}

#[tauri::command]
fn quit(app: AppHandle) {
    hide_tray_panel(&app);
    quit_app(&app);
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct MeetingSummary {
    dir: String,
    name: String,
    /// Folder name minus the leading "YYYY-MM-DD HH.MM " timestamp.
    title: String,
    started_at: Option<String>,
    duration_secs: Option<f64>,
    has_notes: bool,
    has_transcript: bool,
}

fn title_from_folder_name(name: &str) -> String {
    let rest = name
        .split_once(' ')
        .and_then(|(date, rest)| {
            (date.len() == 10 && date.chars().filter(|c| *c == '-').count() == 2)
                .then_some(rest)
        })
        .and_then(|rest| rest.split_once(' ').map(|(_, title)| title))
        .unwrap_or(name)
        .trim();
    if rest.is_empty() { "Meeting".to_string() } else { rest.to_string() }
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
        .map(|path| {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let meta = std::fs::read(path.join("meta.json"))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
            let capture = meta.as_ref().and_then(|m| m.get("capture"));
            let duration_secs = capture.and_then(|c| {
                let mic = c.get("micSeconds").and_then(Value::as_f64).unwrap_or(0.0);
                let system = c.get("systemSeconds").and_then(Value::as_f64).unwrap_or(0.0);
                let longest = mic.max(system);
                (longest > 0.0).then_some(longest)
            });
            MeetingSummary {
                title: title_from_folder_name(&name),
                started_at: meta.as_ref().and_then(|m| {
                    m.get("startedAt").and_then(Value::as_str).map(String::from)
                }),
                duration_secs,
                has_notes: path.join("notes.md").exists(),
                has_transcript: path.join("transcript.md").exists(),
                dir: path.display().to_string(),
                name,
            }
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

/// Overwrite a meeting's notes.md — used by the interactive action-item
/// checklist to persist `- [ ]` ⇄ `- [x]` toggles.
#[tauri::command]
fn update_notes(app: AppHandle, dir: String, content: String) -> Result<(), String> {
    let path = validated_meeting_dir(&app, &dir)?;
    if content.len() > 1_000_000 {
        return Err("notes unexpectedly large".into());
    }
    std::fs::write(path.join("notes.md"), content).map_err(|e| e.to_string())
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

#[tauri::command]
fn reveal_meeting(app: AppHandle, dir: String) -> Result<(), String> {
    let path = validated_meeting_dir(&app, &dir)?;
    tauri_plugin_opener::reveal_item_in_dir(&path).map_err(|e| e.to_string())
}

/// Moves the meeting folder to the user's Trash (recoverable) rather than
/// deleting it outright.
#[tauri::command]
fn delete_meeting(app: AppHandle, dir: String) -> Result<(), String> {
    let path = validated_meeting_dir(&app, &dir)?;
    let trash = app
        .path()
        .home_dir()
        .map_err(|e| e.to_string())?
        .join(".Trash");
    let base_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Distil meeting".into());
    let mut target = trash.join(&base_name);
    if target.exists() {
        target = trash.join(format!(
            "{base_name} {}",
            chrono::Local::now().format("%H.%M.%S")
        ));
    }
    std::fs::rename(&path, &target).map_err(|e| format!("could not move to Trash: {e}"))
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
fn set_diarize_speakers(app: AppHandle, enable: bool) -> Result<(), String> {
    let mut settings = models::load_settings(&app);
    settings.diarize_speakers = enable;
    models::save_settings(&app, &settings)
}

/// Run the capture helper and return the permissions report it prints.
async fn capture_permissions(app: &AppHandle, args: &[&str]) -> Result<Value, String> {
    let output = app
        .shell()
        .sidecar("distil-capture")
        .map_err(|e| e.to_string())?
        .args(args)
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

#[tauri::command]
async fn permissions_status(app: AppHandle) -> Result<Value, String> {
    capture_permissions(&app, &["--check-permissions"]).await
}

/// Raise the real macOS prompt for one permission and report the new state.
///
/// Distinct from `open_privacy_settings`, which only opens the pane: until the
/// app has actually *requested* a permission, macOS does not list it there at
/// all, so opening Settings first is a dead end.
#[tauri::command]
async fn request_permission(app: AppHandle, pane: String) -> Result<Value, String> {
    let pane = PRIVACY_PANES
        .iter()
        .find(|(key, _)| *key == pane)
        .map(|(key, _)| *key)
        .ok_or("unknown permission")?;
    capture_permissions(&app, &["--request-permission", pane]).await
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct NotesPrompt {
    text: String,
    is_custom: bool,
    default_text: String,
}

fn notes_prompt_state<R: Runtime>(app: &AppHandle<R>) -> NotesPrompt {
    let (text, is_custom) = models::load_notes_prompt(app);
    NotesPrompt {
        text,
        is_custom,
        default_text: ollama::DEFAULT_NOTES_PROMPT.to_string(),
    }
}

#[tauri::command]
fn get_notes_prompt(app: AppHandle) -> NotesPrompt {
    notes_prompt_state(&app)
}

#[tauri::command]
fn set_notes_prompt(app: AppHandle, text: String) -> Result<NotesPrompt, String> {
    models::save_notes_prompt(&app, &text)?;
    Ok(notes_prompt_state(&app))
}

#[tauri::command]
fn get_autostart(app: AppHandle) -> bool {
    use tauri_plugin_autostart::ManagerExt;
    app.autolaunch().is_enabled().unwrap_or(false)
}

#[tauri::command]
fn set_autostart(app: AppHandle, enable: bool) -> Result<(), String> {
    use tauri_plugin_autostart::ManagerExt;
    let launcher = app.autolaunch();
    if enable {
        launcher.enable().map_err(|e| e.to_string())
    } else {
        launcher.disable().map_err(|e| e.to_string())
    }
}

const PRIVACY_PANES: &[(&str, &str)] = &[
    (
        "microphone",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
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
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .manage(Mutex::new(AppState::default()))
        .invoke_handler(tauri::generate_handler![
            current_state,
            toggle_recording,
            record_upcoming,
            upcoming_meeting,
            recording_elapsed,
            open_meeting,
            open_settings,
            quit,
            list_meetings,
            read_meeting,
            update_notes,
            distil_meeting,
            reveal_meeting,
            delete_meeting,
            list_ollama_models,
            get_settings,
            set_ollama_model,
            set_allow_cloud_models,
            list_whisper_models,
            set_whisper_model,
            set_keep_audio,
            set_diarize_speakers,
            permissions_status,
            request_permission,
            open_privacy_settings,
            get_autostart,
            set_autostart,
            get_notes_prompt,
            set_notes_prompt
        ])
        .setup(|app| {
            // Menu-bar app: no Dock icon, lives in the tray.
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let tray_icon = tauri::image::Image::from_bytes(include_bytes!("../icons/tray.png"))
                .unwrap_or_else(|_| app.default_window_icon().unwrap().clone());
            TrayIconBuilder::with_id(TRAY_ID)
                .icon(tray_icon)
                .icon_as_template(true)
                .tooltip("Distil")
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        rect,
                        ..
                    } = event
                    {
                        toggle_tray_panel(tray.app_handle(), rect);
                    }
                })
                .build(app)?;

            start_calendar_watcher(app.handle().clone());

            Ok(())
        })
        .on_window_event(|window, event| match event {
            // Closing a window hides it; the app lives in the tray.
            tauri::WindowEvent::CloseRequested { api, .. } => {
                let _ = window.hide();
                api.prevent_close();
            }
            // The tray panel behaves like a popover: losing focus dismisses it.
            tauri::WindowEvent::Focused(false) if window.label() == "tray" => {
                let _ = window.hide();
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
