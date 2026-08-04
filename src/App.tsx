import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { marked } from "marked";
import { SettingsPane } from "./Settings";
import "./App.css";

type UiState =
  | { phase: "idle"; last_meeting: string | null }
  | { phase: "recording"; dir: string }
  | { phase: "stopping" }
  | { phase: "transcribing"; dir: string }
  | { phase: "distilling"; dir: string }
  | { phase: "error"; message: string };

type ModelDownload = {
  status: "started" | "progress" | "done" | "failed";
  percent?: number;
};

type Meeting = {
  dir: string;
  name: string;
  startedAt: string | null;
  hasNotes: boolean;
  hasTranscript: boolean;
};

type MeetingDoc = { notes: string | null; transcript: string | null };

type OllamaModel = { name: string; parameterSize: number | null };

type Settings = {
  whisperModel: string;
  keepAudio: boolean;
  ollamaUrl: string;
  ollamaModel: string | null;
};

function render(md: string): string {
  return marked.parse(md) as string;
}

function App() {
  const [state, setState] = useState<UiState>({ phase: "idle", last_meeting: null });
  const [download, setDownload] = useState<ModelDownload | null>(null);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [doc, setDoc] = useState<MeetingDoc | null>(null);
  const [view, setView] = useState<"notes" | "transcript">("notes");
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [modelsError, setModelsError] = useState(false);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const timer = useRef<number | undefined>(undefined);
  const selectedRef = useRef<string | null>(null);
  selectedRef.current = selected;

  const refreshMeetings = useCallback(() => {
    invoke<Meeting[]>("list_meetings")
      .then((list) => {
        setMeetings(list);
        setSelected((current) =>
          current && list.some((m) => m.dir === current) ? current : (list[0]?.dir ?? null),
        );
      })
      .catch(() => {});
  }, []);

  const loadDoc = useCallback((dir: string) => {
    invoke<MeetingDoc>("read_meeting", { dir })
      .then((d) => {
        setDoc(d);
        setView((v) => (d.notes ? v : "transcript"));
      })
      .catch(() => setDoc(null));
  }, []);

  useEffect(() => {
    invoke<UiState>("current_state").then(setState);
    invoke<Settings>("get_settings").then(setSettings).catch(() => {});
    invoke<OllamaModel[]>("list_ollama_models")
      .then((list) => {
        setModels(list);
        setModelsError(false);
      })
      .catch(() => {
        setModels([]);
        setModelsError(true);
      });
    refreshMeetings();

    const subscriptions = [
      listen<UiState>("pipeline-state", (event) => {
        setState(event.payload);
        if (event.payload.phase === "idle" || event.payload.phase === "error") {
          refreshMeetings();
        }
      }),
      listen<ModelDownload>("model-download", (event) => {
        setDownload(event.payload.status === "done" ? null : event.payload);
      }),
      listen<string>("meeting-ready", (event) => {
        refreshMeetings();
        if (selectedRef.current === event.payload) {
          loadDoc(event.payload);
        }
      }),
    ];
    return () => {
      subscriptions.forEach((s) => s.then((stop) => stop()));
    };
  }, [refreshMeetings, loadDoc]);

  useEffect(() => {
    if (selected) loadDoc(selected);
    else setDoc(null);
  }, [selected, loadDoc]);

  useEffect(() => {
    if (state.phase === "recording") {
      setElapsed(0);
      timer.current = window.setInterval(() => setElapsed((s) => s + 1), 1000);
      return () => window.clearInterval(timer.current);
    }
  }, [state.phase]);

  const recording = state.phase === "recording";
  const stopping = state.phase === "stopping";
  const pipelineBusy = state.phase === "transcribing" || state.phase === "distilling";
  const mmss = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(
    elapsed % 60,
  ).padStart(2, "0")}`;

  const current = meetings.find((m) => m.dir === selected) ?? null;

  const banner = (() => {
    if (state.phase === "error") return <div className="banner error">{state.message}</div>;
    if (state.phase === "transcribing")
      return (
        <div className="banner progress">
          {download?.status === "started" || download?.status === "progress"
            ? `Downloading whisper model… ${download.percent ?? 0}%`
            : `Transcribing · ${basename(state.dir)}`}
        </div>
      );
    if (state.phase === "distilling")
      return <div className="banner progress">Distilling notes · {basename(state.dir)}</div>;
    return null;
  })();

  return (
    <main className="app">
      <header className="titlebar">
        <h1>Distil</h1>
        <div className="titlebar-actions">
          <button
            className={recording ? "record recording" : "record"}
            disabled={stopping}
            onClick={() => invoke("toggle_recording")}
          >
            <span className="dot" aria-hidden />
            {recording ? `Stop · ${mmss}` : stopping ? "Stopping…" : "Start recording"}
          </button>
          <button
            className={showSettings ? "gear active" : "gear"}
            title="Settings"
            onClick={() => setShowSettings((s) => !s)}
          >
            ⚙
          </button>
        </div>
      </header>

      {banner}

      {showSettings && settings && (
        <SettingsPane
          settings={settings}
          ollamaModels={models}
          ollamaError={modelsError}
          onSettingsChange={setSettings}
        />
      )}

      <div className="body" style={showSettings ? { display: "none" } : undefined}>
        <aside className="sidebar">
          {meetings.length === 0 && <p className="sidebar-empty">No meetings yet</p>}
          {meetings.map((m) => (
            <button
              key={m.dir}
              className={m.dir === selected ? "meeting selected" : "meeting"}
              onClick={() => setSelected(m.dir)}
            >
              <span className="meeting-name">{prettyName(m.name)}</span>
              <span className="meeting-tags">
                {m.hasNotes ? <em className="tag notes">notes</em> : null}
                {!m.hasNotes && m.hasTranscript ? (
                  <em className="tag transcript">transcript</em>
                ) : null}
              </span>
            </button>
          ))}
        </aside>

        <section className="detail">
          {!current && (
            <div className="empty">
              <p className="empty-title">{recording ? "Recording…" : "No meetings yet"}</p>
              <p className="empty-hint">
                {recording
                  ? "Capturing your mic and system audio. Stop from here or the menu bar when the meeting ends."
                  : "Start a recording from the menu bar when your next meeting begins. Distil transcribes it and distils notes, all on this Mac."}
              </p>
            </div>
          )}
          {current && (
            <>
              <div className="detail-header">
                <h2>{prettyName(current.name)}</h2>
                <div className="detail-actions">
                  <div className="tabs">
                    <button
                      className={view === "notes" ? "tab active" : "tab"}
                      disabled={!doc?.notes}
                      onClick={() => setView("notes")}
                    >
                      Notes
                    </button>
                    <button
                      className={view === "transcript" ? "tab active" : "tab"}
                      disabled={!doc?.transcript}
                      onClick={() => setView("transcript")}
                    >
                      Transcript
                    </button>
                  </div>
                  {current.hasTranscript && !current.hasNotes && (
                    <button
                      className="generate"
                      disabled={pipelineBusy}
                      onClick={() => invoke("distil_meeting", { dir: current.dir })}
                    >
                      Generate notes
                    </button>
                  )}
                </div>
              </div>
              <article
                className="markdown"
                dangerouslySetInnerHTML={{
                  __html: render(
                    (view === "notes" ? doc?.notes : doc?.transcript) ??
                      "_Nothing here yet._",
                  ),
                }}
              />
            </>
          )}
        </section>
      </div>

      <footer className="statusbar">
        <span className="whisper-label">
          whisper · {settings?.whisperModel ?? "…"} &nbsp;·&nbsp; notes ·{" "}
          {settings?.ollamaModel ?? "auto"}
        </span>
        <button className="statusbar-link" onClick={() => setShowSettings(true)}>
          Settings
        </button>
      </footer>
    </main>
  );
}

function basename(path: string): string {
  return path.split("/").filter(Boolean).pop() ?? path;
}

function prettyName(name: string): string {
  return name.replace(/ Meeting$/, "");
}

export default App;
