pub mod calendar;
pub mod diarize;
pub mod llm;
pub mod meetings;
pub mod models;
pub mod ollama;
pub mod transcribe;
pub mod vad;

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
    detector: meetings::Detector,
    /// Display name of the call app on the microphone, for the tray row.
    detected: Option<String>,
    /// Mirrors the detect_meetings setting, so the 2 s mic snapshots needn't
    /// read settings.json to know whether to surface a call.
    detect_meetings: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            session: None,
            last_state: UiState::Idle { last_meeting: None },
            upcoming: None,
            prompted_meetings: std::collections::HashSet::new(),
            detector: meetings::Detector::default(),
            detected: None,
            detect_meetings: true,
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

/// Transcript → notes.md via the configured (or best local) notes model.
/// Blocking; call while holding PIPELINE_GATE.
fn distil_core<R: Runtime>(app: &AppHandle<R>, dir: &Path) -> Result<(), String> {
    let transcript = std::fs::read_to_string(dir.join("transcript.md"))
        .map_err(|e| format!("cannot read transcript: {e}"))?;
    let settings = models::load_settings(app);
    let (engine, url, selected) = settings.notes_engine();
    let model = llm::resolve_model(engine, url, selected)?;
    llm::check_privacy(engine, url, &model, settings.allow_cloud_models)?;
    let transcript = ollama::notes_input(&transcript, &ollama::meeting_attendees(dir));

    let (template, _) = models::load_notes_prompt(app);
    let notes = llm::generate(
        engine,
        url,
        &model,
        &ollama::build_notes_prompt(&template, &transcript),
        false,
    )?
    .text;
    if notes.is_empty() {
        return Err(format!("{model} returned empty notes"));
    }
    std::fs::write(dir.join("notes.md"), format!("{notes}\n")).map_err(|e| e.to_string())?;

    let meta_path = dir.join("meta.json");
    if let Ok(bytes) = std::fs::read(&meta_path) {
        if let Ok(mut meta) = serde_json::from_slice::<Value>(&bytes) {
            meta["ollamaModel"] = Value::from(model);
            meta["notesEngine"] = serde_json::to_value(engine).unwrap_or_default();
            meta["distilledAt"] = Value::from(chrono::Local::now().to_rfc3339());
            let _ = std::fs::write(
                &meta_path,
                serde_json::to_vec_pretty(&meta).unwrap_or_default(),
            );
        }
    }
    Ok(())
}

/// What transcribed a meeting, for meta.json. The fallbacks in
/// `build_transcriber` mean it can differ from what the settings ask for.
struct TranscriberUsed {
    kind: transcribe::TranscriberKind,
    skipped_silence: bool,
}

/// Fetch the models the settings ask for and load the transcriber.
/// Blocking; call while holding PIPELINE_GATE. A meeting is never lost to a
/// model that can't be fetched: without the VAD model whisper runs over the
/// whole stream, and Parakeet, which can't run without it, hands over to whisper.
fn build_transcriber<R: Runtime>(
    app: &AppHandle<R>,
    settings: &models::Settings,
) -> Result<(transcribe::Transcriber, TranscriberUsed), String> {
    let parakeet = settings.transcriber == transcribe::TranscriberKind::Parakeet;
    let vad_model = if parakeet || settings.skip_silence {
        match models::ensure_vad_model(app) {
            Ok(path) => Some(path),
            Err(message) => {
                eprintln!("silence skipping unavailable: {message}");
                None
            }
        }
    } else {
        None
    };

    if parakeet {
        match &vad_model {
            Some(vad_model) => match models::ensure_parakeet_model(app)
                .and_then(|dir| transcribe::Transcriber::parakeet(&dir, vad_model))
            {
                Ok(transcriber) => {
                    let used = TranscriberUsed {
                        kind: transcribe::TranscriberKind::Parakeet,
                        skipped_silence: true,
                    };
                    return Ok((transcriber, used));
                }
                Err(message) => eprintln!("Parakeet unavailable, using whisper: {message}"),
            },
            None => eprintln!("Parakeet needs the VAD model, using whisper"),
        }
    }

    // Parakeet users who fall back get whisper's own setting for silence.
    let whisper_vad = vad_model.as_deref().filter(|_| settings.skip_silence);
    let model = models::ensure_whisper_model(app)?;
    let transcriber = transcribe::Transcriber::whisper(&model, whisper_vad)?;
    let used = TranscriberUsed {
        kind: transcribe::TranscriberKind::Whisper,
        skipped_silence: whisper_vad.is_some(),
    };
    Ok((transcriber, used))
}

fn run_transcription<R: Runtime>(app: &AppHandle<R>, dir: &Path) {
    let _gate = PIPELINE_GATE.lock().unwrap();

    let result = (|| -> Result<(), String> {
        let settings = models::load_settings(app);

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

        // A recording that actually captured system audio proves the permission
        // is granted, so the Settings badge can be right without ever running
        // the tone probe. Silence proves nothing — the meeting may simply have
        // been quiet — so only the positive case updates the cache.
        if peak_of("systemPeak").is_some() && !system_silent {
            set_system_audio_access(app, "granted");
        }

        // One transcriber for both streams, loaded only if there is audio to
        // transcribe, and gone before diarization loads its models.
        let (me, them, used) = if mic_silent && system_silent {
            (Vec::new(), Vec::new(), None)
        } else {
            let (mut transcriber, used) = build_transcriber(app, &settings)?;
            let me = if mic_silent {
                Vec::new()
            } else {
                transcriber.transcribe(&dir.join("audio-mic.wav"))?
            };
            let them = if system_silent {
                Vec::new()
            } else {
                transcriber.transcribe(&dir.join("audio-system.wav"))?
            };
            drop(transcriber);
            (me, them, Some(used))
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
                if let Some(used) = &used {
                    meta["transcriber"] = serde_json::to_value(used.kind).unwrap_or_default();
                    if used.kind == transcribe::TranscriberKind::Whisper {
                        meta["whisperModel"] =
                            serde_json::Value::from(settings.whisper_model.clone());
                    }
                    meta["skippedSilence"] = serde_json::Value::from(used.skipped_silence);
                }
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

/// Start recording the call the mic watcher spotted: named after the calendar
/// meeting when one is on, otherwise after the app ("Teams call").
fn record_detected_impl<R: Runtime>(app: &AppHandle<R>) {
    let (has_upcoming, detected) = {
        let state = app.state::<Mutex<AppState>>();
        let guard = state.lock().unwrap();
        (guard.upcoming.is_some(), guard.detected.clone())
    };
    if has_upcoming {
        record_upcoming_impl(app);
    } else {
        let title = detected.map(|name| format!("{name} call"));
        start_recording(app, title.as_deref(), Vec::new());
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

/// How long to wait before restarting a mic watcher that exited.
const MIC_WATCHER_RESTART_SECS: u64 = 30;

fn start_mic_watcher<R: Runtime>(app: AppHandle<R>) {
    std::thread::spawn(move || loop {
        tauri::async_runtime::block_on(watch_mic_once(&app));
        clear_detected(&app);
        std::thread::sleep(std::time::Duration::from_secs(MIC_WATCHER_RESTART_SECS));
    });
}

/// Run the `watch-mic` sidecar until it exits, feeding its snapshots to the
/// call detector.
async fn watch_mic_once<R: Runtime>(app: &AppHandle<R>) {
    let spawned = app
        .shell()
        .sidecar("distil-capture")
        .map_err(|e| e.to_string())
        .and_then(|command| {
            command
                .args(["watch-mic"])
                .spawn()
                .map_err(|e| e.to_string())
        });
    // `_child` must outlive the loop: dropping it closes the sidecar's stdin,
    // which it reads as "the host is gone" and exits.
    let (mut rx, _child) = match spawned {
        Ok(parts) => parts,
        Err(message) => {
            eprintln!("mic watcher failed to start: {message}");
            return;
        }
    };

    // Stdout arrives in arbitrary chunks, and this stream never ends, so lines
    // are reassembled rather than assumed whole.
    let mut lines = meetings::LineBuffer::default();
    while let Some(event) = rx.recv().await {
        match event {
            CommandEvent::Stdout(bytes) => {
                for line in lines.push(&bytes) {
                    if let Some(users) = meetings::parse_mic_activity(&line) {
                        handle_mic_snapshot(app, &users);
                    }
                }
            }
            CommandEvent::Stderr(bytes) => {
                eprintln!(
                    "distil-capture watch-mic: {}",
                    String::from_utf8_lossy(&bytes)
                );
            }
            CommandEvent::Terminated(status) => {
                eprintln!("mic watcher exited (status {:?})", status.code);
                break;
            }
            _ => {}
        }
    }
}

fn handle_mic_snapshot<R: Runtime>(app: &AppHandle<R>, users: &[meetings::MicUser]) {
    let (signal, detected_change, calendar_title) = {
        let handle = app.state::<Mutex<AppState>>();
        let mut guard = handle.lock().unwrap();
        let recording = guard.session.is_some();
        let calendar_meeting = guard.upcoming.is_some();
        let signal = guard.detector.update(
            users,
            recording,
            calendar_meeting,
            std::time::Instant::now(),
        );
        let detected = guard
            .detector
            .current()
            .filter(|_| guard.detect_meetings)
            .map(str::to_string);
        let changed = detected != guard.detected;
        guard.detected = detected.clone();
        (
            signal,
            changed.then_some(detected),
            guard.upcoming.as_ref().map(|meeting| meeting.title.clone()),
        )
    };

    if let Some(detected) = detected_change {
        let _ = app.emit("meeting-detected", &detected);
    }
    if let Some(signal) = signal {
        act_on_call_signal(app, signal, calendar_title);
    }
}

/// The watcher is gone, so what it last reported is stale: forget it rather
/// than leave the tray offering a call that may have ended.
fn clear_detected<R: Runtime>(app: &AppHandle<R>) {
    let had_detected = {
        let handle = app.state::<Mutex<AppState>>();
        let mut guard = handle.lock().unwrap();
        guard.detector = meetings::Detector::default();
        guard.detected.take().is_some()
    };
    if had_detected {
        let _ = app.emit("meeting-detected", None::<String>);
    }
}

/// Notify about, or act on, a call starting or ending. Settings are read here,
/// when a signal actually fires, rather than on every 2 s snapshot.
fn act_on_call_signal<R: Runtime>(
    app: &AppHandle<R>,
    signal: meetings::Signal,
    calendar_title: Option<String>,
) {
    let settings = models::load_settings(app);
    if !settings.detect_meetings {
        return;
    }
    match signal {
        meetings::Signal::Started { app: call_app } => {
            let title = calendar_title.unwrap_or_else(|| format!("{call_app} call"));
            notify(
                app,
                &format!("{title} detected"),
                "Click the Distil menu-bar icon to record it.",
            );
        }
        meetings::Signal::Ended { app: call_app } => {
            if settings.auto_stop_recording {
                stop_recording(app);
                notify(
                    app,
                    "Recording stopped",
                    &format!("The {call_app} call ended, so Distil stopped recording."),
                );
            } else {
                notify(
                    app,
                    &format!("The {call_app} call ended"),
                    "Stop recording? Click the Distil menu-bar icon to stop it.",
                );
            }
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
fn record_detected(app: AppHandle) {
    record_detected_impl(&app);
}

#[tauri::command]
fn detected_call(state: tauri::State<'_, Mutex<AppState>>) -> Option<String> {
    state.lock().unwrap().detected.clone()
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
struct NotesModelInfo {
    name: String,
    parameter_size: Option<f64>,
    is_cloud: bool,
}

/// Models on the active engine's server.
#[tauri::command]
fn list_notes_models(app: AppHandle) -> Result<Vec<NotesModelInfo>, String> {
    let settings = models::load_settings(&app);
    let (engine, url, _) = settings.notes_engine();
    Ok(llm::list_models(engine, url, settings.allow_cloud_models)?
        .into_iter()
        .map(|m| NotesModelInfo {
            name: m.name,
            parameter_size: m.parameter_size,
            is_cloud: m.is_cloud,
        })
        .collect())
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

/// Sets the notes model for whichever engine is active.
#[tauri::command]
fn set_notes_model(app: AppHandle, model: Option<String>) -> Result<(), String> {
    let mut settings = models::load_settings(&app);
    let model = model.filter(|m| !m.is_empty());
    match settings.engine {
        llm::Engine::Ollama => settings.ollama_model = model,
        llm::Engine::OpenAi => settings.openai_model = model,
    }
    models::save_settings(&app, &settings)
}

/// Switches the notes engine; `openai_url` (when given) replaces the
/// OpenAI-compatible server address, and clears a model picked on the old one.
#[tauri::command]
fn set_notes_engine(
    app: AppHandle,
    engine: llm::Engine,
    openai_url: Option<String>,
) -> Result<(), String> {
    let mut settings = models::load_settings(&app);
    settings.engine = engine;
    if let Some(url) = openai_url {
        let url = url.trim().trim_end_matches('/').to_string();
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err("server address must start with http:// or https://".into());
        }
        if url != settings.openai_url {
            settings.openai_url = url;
            settings.openai_model = None;
        }
    }
    models::save_settings(&app, &settings)
}

/// Curated whisper choices — name in ggerganov/whisper.cpp, blurb, size on disk.
const WHISPER_MODELS: &[(&str, &str, u64)] = &[
    ("large-v3-turbo-q5_0", "Best balance of accuracy and speed (default)", 574),
    ("large-v3-q5_0", "Maximum accuracy, slower", 1_081),
    ("small", "Faster, lighter accuracy", 466),
    ("tiny.en-q8_0", "Fastest, English-only — for quick tests", 43),
];

/// The Parakeet choice beside the whisper models: name, blurb, size on disk.
const PARAKEET_MODEL: (&str, &str, u64) =
    ("parakeet-tdt-0.6b-v3", "Fastest, 25 European languages", 670);

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct TranscriptionModelInfo {
    engine: transcribe::TranscriberKind,
    name: String,
    description: String,
    size_mb: u64,
    downloaded: bool,
    selected: bool,
}

/// Every transcription choice: the whisper models, then Parakeet.
#[tauri::command]
fn list_transcription_models(app: AppHandle) -> Result<Vec<TranscriptionModelInfo>, String> {
    let settings = models::load_settings(&app);
    let models_dir = models::models_dir(&app)?;
    let whisper_selected = settings.transcriber == transcribe::TranscriberKind::Whisper;
    let mut rows: Vec<TranscriptionModelInfo> = WHISPER_MODELS
        .iter()
        .map(|(name, description, size_mb)| TranscriptionModelInfo {
            engine: transcribe::TranscriberKind::Whisper,
            name: (*name).to_string(),
            description: (*description).to_string(),
            size_mb: *size_mb,
            downloaded: models_dir.join(format!("ggml-{name}.bin")).exists(),
            selected: whisper_selected && settings.whisper_model == *name,
        })
        .collect();
    let (name, description, size_mb) = PARAKEET_MODEL;
    rows.push(TranscriptionModelInfo {
        engine: transcribe::TranscriberKind::Parakeet,
        name: name.to_string(),
        description: description.to_string(),
        size_mb,
        downloaded: models::parakeet_present(&models_dir),
        selected: !whisper_selected,
    });
    Ok(rows)
}

/// Choose what transcribes. Picking a whisper model also switches to whisper,
/// and the whisper model stays remembered while Parakeet is selected.
#[tauri::command]
fn set_transcription_model(
    app: AppHandle,
    engine: transcribe::TranscriberKind,
    model: String,
) -> Result<(), String> {
    let mut settings = models::load_settings(&app);
    match engine {
        transcribe::TranscriberKind::Whisper => {
            if !WHISPER_MODELS.iter().any(|(name, ..)| *name == model) {
                return Err("unknown whisper model".into());
            }
            settings.whisper_model = model;
        }
        transcribe::TranscriberKind::Parakeet => {
            if model != PARAKEET_MODEL.0 {
                return Err("unknown parakeet model".into());
            }
        }
    }
    settings.transcriber = engine;
    models::save_settings(&app, &settings)
}

#[tauri::command]
fn set_skip_silence(app: AppHandle, enable: bool) -> Result<(), String> {
    let mut settings = models::load_settings(&app);
    settings.skip_silence = enable;
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

#[tauri::command]
fn set_detect_meetings(app: AppHandle, enable: bool) -> Result<(), String> {
    let mut settings = models::load_settings(&app);
    settings.detect_meetings = enable;
    models::save_settings(&app, &settings)?;
    // Switching off also takes down a call the tray is already offering.
    let cleared = {
        let state = app.state::<Mutex<AppState>>();
        let mut guard = state.lock().unwrap();
        guard.detect_meetings = enable;
        !enable && guard.detected.take().is_some()
    };
    if cleared {
        let _ = app.emit("meeting-detected", None::<String>);
    }
    Ok(())
}

#[tauri::command]
fn set_auto_stop_recording(app: AppHandle, enable: bool) -> Result<(), String> {
    let mut settings = models::load_settings(&app);
    settings.auto_stop_recording = enable;
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
    let mut report = capture_permissions(&app, &["--check-permissions"]).await?;
    // System audio is merged in from the cache rather than checked live: the
    // probe plays a tone, which would be obnoxious on every Settings render.
    if let Some(map) = report.as_object_mut() {
        let settings = models::load_settings(&app);
        map.insert("systemAudio".into(), Value::from(settings.system_audio_access));
    }
    Ok(report)
}

/// Measure whether system-audio capture actually works, and remember the answer.
///
/// Creating the tap is also what raises the macOS prompt, so this doubles as
/// the request for the permission.
#[tauri::command]
async fn probe_system_audio(app: AppHandle) -> Result<Value, String> {
    let output = app
        .shell()
        .sidecar("distil-capture")
        .map_err(|e| e.to_string())?
        .args(["--check-system-audio"])
        .output()
        .await
        .map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let status = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|v| v.get("event").and_then(Value::as_str) == Some("systemAudio"))
        .and_then(|v| v.get("status").and_then(Value::as_str).map(str::to_string))
        .ok_or_else(|| "capture helper produced no system-audio report".to_string())?;

    // "unavailable" means the probe itself could not run (no output device,
    // for instance) — that says nothing about the permission, so don't cache it.
    if status != "unavailable" {
        set_system_audio_access(&app, &status);
    }
    permissions_status(app).await
}

fn set_system_audio_access<R: Runtime>(app: &AppHandle<R>, status: &str) {
    let mut settings = models::load_settings(app);
    if settings.system_audio_access != status {
        settings.system_audio_access = status.to_string();
        let _ = models::save_settings(app, &settings);
    }
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
            record_detected,
            detected_call,
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
            list_notes_models,
            get_settings,
            set_notes_model,
            set_notes_engine,
            set_allow_cloud_models,
            list_transcription_models,
            set_transcription_model,
            set_skip_silence,
            set_keep_audio,
            set_diarize_speakers,
            set_detect_meetings,
            set_auto_stop_recording,
            permissions_status,
            request_permission,
            probe_system_audio,
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
            app.state::<Mutex<AppState>>().lock().unwrap().detect_meetings =
                models::load_settings(app.handle()).detect_meetings;
            start_mic_watcher(app.handle().clone());

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
