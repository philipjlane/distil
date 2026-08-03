use std::sync::Mutex;

use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
    AppHandle, Emitter, Manager, Runtime,
};

const TRAY_ID: &str = "distil-tray";

/// Recording lifecycle. M0 only toggles between the first two states;
/// the pipeline states arrive with M1–M3.
#[derive(Default)]
struct AppState {
    recording: bool,
}

fn build_tray_menu<R: Runtime>(app: &AppHandle<R>, recording: bool) -> tauri::Result<Menu<R>> {
    let toggle_label = if recording { "Stop Recording" } else { "Start Recording" };
    let toggle = MenuItem::with_id(app, "toggle-recording", toggle_label, true, None::<&str>)?;
    let open = MenuItem::with_id(app, "open-library", "Open Distil", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Distil", true, None::<&str>)?;
    let sep1 = PredefinedMenuItem::separator(app)?;
    let sep2 = PredefinedMenuItem::separator(app)?;
    Menu::with_items(app, &[&toggle, &sep1, &open, &sep2, &quit])
}

fn set_recording<R: Runtime>(app: &AppHandle<R>, recording: bool) {
    app.state::<Mutex<AppState>>().lock().unwrap().recording = recording;

    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        if let Ok(menu) = build_tray_menu(app, recording) {
            let _ = tray.set_menu(Some(menu));
        }
        let tooltip = if recording { "Distil — recording" } else { "Distil" };
        let _ = tray.set_tooltip(Some(tooltip));
    }

    let _ = app.emit("recording-state", recording);
}

fn show_library<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

#[tauri::command]
fn recording_state(state: tauri::State<'_, Mutex<AppState>>) -> bool {
    state.lock().unwrap().recording
}

#[tauri::command]
fn toggle_recording(app: AppHandle) -> bool {
    let next = !app.state::<Mutex<AppState>>().lock().unwrap().recording;
    set_recording(&app, next);
    next
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(Mutex::new(AppState::default()))
        .invoke_handler(tauri::generate_handler![recording_state, toggle_recording])
        .setup(|app| {
            // Menu-bar app: no Dock icon, lives in the tray.
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let menu = build_tray_menu(app.handle(), false)?;
            TrayIconBuilder::with_id(TRAY_ID)
                .icon(app.default_window_icon().unwrap().clone())
                .icon_as_template(true)
                .menu(&menu)
                .show_menu_on_left_click(true)
                .tooltip("Distil")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "toggle-recording" => {
                        let current = app.state::<Mutex<AppState>>().lock().unwrap().recording;
                        set_recording(app, !current);
                    }
                    "open-library" => show_library(app),
                    "quit" => app.exit(0),
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
