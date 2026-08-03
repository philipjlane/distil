import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

type UiState =
  | { phase: "idle"; last_meeting: string | null }
  | { phase: "recording"; dir: string }
  | { phase: "stopping" }
  | { phase: "error"; message: string };

function basename(path: string): string {
  return path.split("/").filter(Boolean).pop() ?? path;
}

function App() {
  const [state, setState] = useState<UiState>({ phase: "idle", last_meeting: null });
  const [elapsed, setElapsed] = useState(0);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    invoke<UiState>("current_state").then(setState);
    const unlisten = listen<UiState>("pipeline-state", (event) => {
      setState(event.payload);
    });
    return () => {
      unlisten.then((stop) => stop());
    };
  }, []);

  useEffect(() => {
    if (state.phase === "recording") {
      setElapsed(0);
      timer.current = window.setInterval(() => setElapsed((s) => s + 1), 1000);
      return () => window.clearInterval(timer.current);
    }
  }, [state.phase]);

  const recording = state.phase === "recording";
  const stopping = state.phase === "stopping";
  const mmss = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(
    elapsed % 60,
  ).padStart(2, "0")}`;

  return (
    <main className="app">
      <header className="titlebar">
        <h1>Distil</h1>
        <button
          className={recording ? "record recording" : "record"}
          disabled={stopping}
          onClick={() => invoke("toggle_recording")}
        >
          <span className="dot" aria-hidden />
          {recording ? `Stop · ${mmss}` : stopping ? "Stopping…" : "Start recording"}
        </button>
      </header>

      {state.phase === "error" && (
        <div className="banner error">{state.message}</div>
      )}
      {state.phase === "idle" && state.last_meeting && (
        <div className="banner saved">Saved · {basename(state.last_meeting)}</div>
      )}

      <section className="library">
        <div className="empty">
          <p className="empty-title">
            {recording ? "Recording…" : "No meetings yet"}
          </p>
          <p className="empty-hint">
            {recording
              ? "Capturing your mic and system audio. Stop here or from the menu bar when the meeting ends."
              : "Start a recording from the menu bar when your next meeting begins. Distil will transcribe it and distil the notes here."}
          </p>
        </div>
      </section>
    </main>
  );
}

export default App;
