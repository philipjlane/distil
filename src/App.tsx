import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

function App() {
  const [recording, setRecording] = useState(false);

  useEffect(() => {
    invoke<boolean>("recording_state").then(setRecording);
    const unlisten = listen<boolean>("recording-state", (event) => {
      setRecording(event.payload);
    });
    return () => {
      unlisten.then((stop) => stop());
    };
  }, []);

  return (
    <main className="app">
      <header className="titlebar">
        <h1>Distil</h1>
        <button
          className={recording ? "record recording" : "record"}
          onClick={() => invoke<boolean>("toggle_recording").then(setRecording)}
        >
          <span className="dot" aria-hidden />
          {recording ? "Stop recording" : "Start recording"}
        </button>
      </header>

      <section className="library">
        <div className="empty">
          <p className="empty-title">No meetings yet</p>
          <p className="empty-hint">
            Start a recording from the menu bar when your next meeting begins.
            Distil will transcribe it and distil the notes here.
          </p>
        </div>
      </section>
    </main>
  );
}

export default App;
