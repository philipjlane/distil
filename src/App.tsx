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
  title: string;
  startedAt: string | null;
  durationSecs: number | null;
  hasNotes: boolean;
  hasTranscript: boolean;
};

type MeetingMeta = {
  startedAt?: string;
  whisperModel?: string;
  ollamaModel?: string;
} | null;

type MeetingDoc = { notes: string | null; transcript: string | null; meta: MeetingMeta };

function relativeStamp(iso: string | null | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  const now = new Date();
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const sameDay = (a: Date, b: Date) => a.toDateString() === b.toDateString();
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (sameDay(date, now)) return `Today ${time}`;
  if (sameDay(date, yesterday)) return `Yesterday ${time}`;
  return `${date.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" })} ${time}`;
}

function durationLabel(secs: number | null | undefined): string | null {
  if (!secs || secs <= 0) return null;
  const minutes = Math.floor(secs / 60);
  const seconds = Math.round(secs % 60);
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

const Icon = {
  folder: (
    <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round">
      <path d="M1.75 4.25a1 1 0 0 1 1-1h3l1.5 1.75h6a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1h-10.5a1 1 0 0 1-1-1z" />
    </svg>
  ),
  copy: (
    <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round">
      <rect x="5.5" y="5.5" width="8" height="8" rx="1.5" />
      <path d="M10.5 3.5v-1a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h1" />
    </svg>
  ),
  trash: (
    <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2.5 4.5h11M6.5 2.5h3M4 4.5l.7 9a1 1 0 0 0 1 .9h4.6a1 1 0 0 0 1-.9l.7-9M6.5 7v4.5M9.5 7v4.5" />
    </svg>
  ),
  check: (
    <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 8.5l3.5 3.5L13 5" />
    </svg>
  ),
};

type OllamaModel = { name: string; parameterSize: number | null; isCloud: boolean };

type Settings = {
  whisperModel: string;
  keepAudio: boolean;
  ollamaUrl: string;
  ollamaModel: string | null;
  allowCloudModels: boolean;
  diarizeSpeakers: boolean;
};

type UpcomingMeeting = { title: string; startsAtDisplay: string } | null;

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
  const [upcoming, setUpcoming] = useState<UpcomingMeeting>(null);
  const [elapsed, setElapsed] = useState(0);
  const [copied, setCopied] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
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

  const loadModels = useCallback(() => {
    invoke<OllamaModel[]>("list_ollama_models")
      .then((list) => {
        setModels(list);
        setModelsError(false);
      })
      .catch(() => {
        setModels([]);
        setModelsError(true);
      });
  }, []);

  useEffect(() => {
    invoke<UiState>("current_state").then(setState);
    invoke<Settings>("get_settings").then(setSettings).catch(() => {});
    loadModels();
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
      listen<UpcomingMeeting>("upcoming-meeting", (event) => {
        setUpcoming(event.payload);
      }),
    ];
    invoke<UpcomingMeeting>("upcoming_meeting").then(setUpcoming).catch(() => {});
    return () => {
      subscriptions.forEach((s) => s.then((stop) => stop()));
    };
  }, [refreshMeetings, loadDoc]);

  useEffect(() => {
    if (selected) loadDoc(selected);
    else setDoc(null);
    setCopied(false);
    setConfirmingDelete(false);
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
  const processingSelected =
    (state.phase === "transcribing" || state.phase === "distilling") &&
    state.dir === selected;

  const copyCurrent = () => {
    const content = view === "notes" ? doc?.notes : doc?.transcript;
    if (!content) return;
    navigator.clipboard
      .writeText(content)
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {});
  };

  const deleteCurrent = () => {
    if (!current) return;
    if (!confirmingDelete) {
      setConfirmingDelete(true);
      window.setTimeout(() => setConfirmingDelete(false), 3000);
      return;
    }
    invoke("delete_meeting", { dir: current.dir })
      .then(() => {
        setConfirmingDelete(false);
        setSelected(null);
        refreshMeetings();
      })
      .catch(() => setConfirmingDelete(false));
  };

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
      <header className="titlebar" data-tauri-drag-region>
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

      {upcoming && !recording && !stopping && (
        <div className="banner upcoming">
          <span>
            {upcoming.title} at {upcoming.startsAtDisplay}
          </span>
          <button onClick={() => invoke("record_upcoming")}>Record</button>
        </div>
      )}

      {showSettings && settings && (
        <SettingsPane
          settings={settings}
          ollamaModels={models}
          ollamaError={modelsError}
          onSettingsChange={setSettings}
          onReloadModels={loadModels}
        />
      )}

      <div className="body" style={showSettings ? { display: "none" } : undefined}>
        <aside className="sidebar">
          {meetings.length === 0 && <p className="sidebar-empty">No meetings yet</p>}
          {meetings.map((m) => {
            const busyHere =
              (state.phase === "transcribing" || state.phase === "distilling") &&
              state.dir === m.dir;
            const sub = [relativeStamp(m.startedAt), durationLabel(m.durationSecs)]
              .filter(Boolean)
              .join(" · ");
            return (
              <button
                key={m.dir}
                className={m.dir === selected ? "meeting selected" : "meeting"}
                onClick={() => setSelected(m.dir)}
              >
                <span className="meeting-row">
                  <span className="meeting-name">{m.title}</span>
                  {busyHere ? (
                    <em className="tag working">working</em>
                  ) : m.hasNotes ? (
                    <em className="tag notes">notes</em>
                  ) : m.hasTranscript ? (
                    <em className="tag transcript">transcript</em>
                  ) : null}
                </span>
                {sub && <span className="meeting-sub">{sub}</span>}
              </button>
            );
          })}
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
                <div className="detail-title">
                  <h2>{current.title}</h2>
                  <span className="detail-meta">
                    {[
                      relativeStamp(current.startedAt),
                      durationLabel(current.durationSecs),
                      doc?.meta?.ollamaModel ?? undefined,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                </div>
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
                  {current.hasTranscript && !current.hasNotes && !processingSelected && (
                    <button
                      className="generate"
                      disabled={pipelineBusy}
                      onClick={() => invoke("distil_meeting", { dir: current.dir })}
                    >
                      Generate notes
                    </button>
                  )}
                  <button
                    className="icon-button"
                    title="Reveal in Finder"
                    onClick={() => invoke("reveal_meeting", { dir: current.dir })}
                  >
                    {Icon.folder}
                  </button>
                  <button
                    className="icon-button"
                    title="Copy to clipboard"
                    disabled={!(view === "notes" ? doc?.notes : doc?.transcript)}
                    onClick={copyCurrent}
                  >
                    {copied ? Icon.check : Icon.copy}
                  </button>
                  <button
                    className={confirmingDelete ? "icon-button danger confirming" : "icon-button danger"}
                    title={confirmingDelete ? "Click again to move to Trash" : "Delete meeting"}
                    onClick={deleteCurrent}
                  >
                    {Icon.trash}
                    {confirmingDelete && <span className="confirm-label">Sure?</span>}
                  </button>
                </div>
              </div>
              {processingSelected ? (
                <div className="empty processing">
                  <p className="empty-title">
                    {state.phase === "transcribing" ? "Transcribing…" : "Distilling notes…"}
                  </p>
                  <p className="empty-hint">
                    Running locally — this view updates the moment it's done.
                  </p>
                </div>
              ) : (
                <article
                  className="markdown"
                  dangerouslySetInnerHTML={{
                    __html: render(
                      (view === "notes" ? doc?.notes : doc?.transcript) ??
                        "_Nothing here yet._",
                    ),
                  }}
                />
              )}
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

export default App;
