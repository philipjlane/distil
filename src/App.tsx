import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { marked } from "marked";
import {
  Check,
  Copy,
  Folder,
  List,
  Mic,
  Search,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { SettingsPane } from "./Settings";
import { Onboarding } from "./Onboarding";
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
  model?: string;
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

type NotesModel = { name: string; parameterSize: number | null; isCloud: boolean };

type Settings = {
  whisperModel: string;
  transcriber: "whisper" | "parakeet";
  skipSilence: boolean;
  keepAudio: boolean;
  ollamaUrl: string;
  ollamaModel: string | null;
  engine: "ollama" | "openai";
  openaiUrl: string;
  openaiModel: string | null;
  allowCloudModels: boolean;
  diarizeSpeakers: boolean;
  liveTranscript: boolean;
  detectMeetings: boolean;
  autoStopRecording: boolean;
};

type UpcomingMeeting = { title: string; startsAtDisplay: string } | null;

/* ---------- formatting helpers ---------- */

function stamp(iso: string | null | undefined, withAt: boolean): string {
  if (!iso) return "";
  const date = new Date(iso);
  const now = new Date();
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const sameDay = (a: Date, b: Date) => a.toDateString() === b.toDateString();
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  const sep = withAt ? " at " : " ";
  if (sameDay(date, now)) return `Today${sep}${time}`;
  if (sameDay(date, yesterday)) return `Yesterday${sep}${time}`;
  const day = date.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" });
  return `${day}${sep}${time}`;
}

function durationLabel(secs: number | null | undefined): string | null {
  if (!secs || secs <= 0) return null;
  const minutes = Math.round(secs / 60);
  return minutes >= 1 ? `${minutes} min` : `${Math.round(secs)}s`;
}

function basename(path: string): string {
  return path.split("/").filter(Boolean).pop() ?? path;
}

function render(md: string): string {
  return marked.parse(md) as string;
}

/* ---------- notes / transcript parsing ---------- */

type Section = { label: string; body: string };

function splitSections(md: string): Section[] {
  const sections: Section[] = [];
  let label = "Notes";
  let buffer: string[] = [];
  const flush = () => {
    const body = buffer.join("\n").trim();
    if (body) sections.push({ label, body });
    buffer = [];
  };
  for (const line of md.split("\n")) {
    const heading = line.match(/^#{1,3}\s+(.*)$/);
    if (heading) {
      flush();
      label = heading[1].trim();
    } else {
      buffer.push(line);
    }
  }
  flush();
  return sections.filter((s) => !/^transcript$/i.test(s.label) || s.body);
}

type ActionItem = { lineIndex: number; checked: boolean; owner: string; task: string };

function parseActionItems(notes: string): ActionItem[] {
  const items: ActionItem[] = [];
  let inActions = false;
  notes.split("\n").forEach((line, lineIndex) => {
    const heading = line.match(/^#{1,3}\s+(.*)$/);
    if (heading) {
      inActions = /action items/i.test(heading[1]);
      return;
    }
    if (!inActions) return;
    const task = line.match(/^\s*[-*]\s*\[( |x|X)\]\s*(.*)$/);
    if (!task) return;
    const checked = task[1].toLowerCase() === "x";
    const rest = task[2].trim();
    const owned = rest.match(/^([^:—]{1,28}?)\s*[:—]\s+(.+)$/);
    items.push({
      lineIndex,
      checked,
      owner: owned ? owned[1].replace(/\*\*/g, "").trim() : "—",
      task: (owned ? owned[2] : rest).replace(/\*\*/g, "").trim(),
    });
  });
  return items;
}

function toggleTaskLine(notes: string, lineIndex: number): string {
  const lines = notes.split("\n");
  const line = lines[lineIndex];
  if (!line) return notes;
  lines[lineIndex] = /\[\s\]/.test(line)
    ? line.replace(/\[\s\]/, "[x]")
    : line.replace(/\[(x|X)\]/, "[ ]");
  return lines.join("\n");
}

type Utterance = { who: string; text: string };

function parseTranscript(md: string): Utterance[] {
  const utterances: Utterance[] = [];
  let current: Utterance | null = null;
  for (const line of md.split("\n")) {
    const head = line.match(/^\*\*(.+?)\*\*\s*\[\d{1,2}:\d{2}\]\s*$/);
    if (head) {
      if (current) utterances.push(current);
      current = { who: head[1], text: "" };
    } else if (current && line.trim() && !/^#/.test(line)) {
      current.text += (current.text ? "\n" : "") + line.trim();
    }
  }
  if (current) utterances.push(current);
  return utterances;
}

function initialsFor(who: string): string {
  if (who === "Me") return "ME";
  const numbered = who.match(/^Them\s+(\d+)$/);
  if (numbered) return `T${numbered[1]}`;
  return "TH";
}

/* ---------- app ---------- */

function App() {
  const [state, setState] = useState<UiState>({ phase: "idle", last_meeting: null });
  const [download, setDownload] = useState<ModelDownload | null>(null);
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [doc, setDoc] = useState<MeetingDoc | null>(null);
  const [view, setView] = useState<"notes" | "transcript">("notes");
  const [nav, setNav] = useState<"library" | "settings">("library");
  const [query, setQuery] = useState("");
  const [models, setModels] = useState<NotesModel[]>([]);
  const [modelsError, setModelsError] = useState(false);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [upcoming, setUpcoming] = useState<UpcomingMeeting>(null);
  const [elapsed, setElapsed] = useState(0);
  const [copied, setCopied] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [onboarding, setOnboarding] = useState(false);
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
    invoke<NotesModel[]>("list_notes_models")
      .then((list) => {
        setModels(list);
        setModelsError(false);
      })
      .catch(() => {
        setModels([]);
        setModelsError(true);
      });
  }, []);

  // Ollama is often started after Distil. While it's unreachable keep
  // re-checking, so the banner clears itself rather than stranding the user on
  // a stale error until the next launch. Stops as soon as a check succeeds.
  useEffect(() => {
    if (!modelsError) return;
    const poll = window.setInterval(loadModels, 5000);
    return () => window.clearInterval(poll);
  }, [modelsError, loadModels]);

  useEffect(() => {
    invoke<UiState>("current_state").then(setState);
    invoke<Settings>("get_settings").then(setSettings).catch(() => {});
    invoke<UpcomingMeeting>("upcoming_meeting").then(setUpcoming).catch(() => {});
    loadModels();
    refreshMeetings();

    if (!localStorage.getItem("distil-onboarded")) {
      invoke<{ microphone: string }>("permissions_status")
        .then((p) => {
          if (p.microphone !== "granted") {
            setOnboarding(true);
          } else {
            localStorage.setItem("distil-onboarded", "1");
          }
        })
        .catch(() => {});
    }

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
      listen<string>("open-meeting", (event) => {
        setNav("library");
        setSelected(event.payload);
      }),
      listen("open-settings", () => {
        setNav("settings");
      }),
    ];
    return () => {
      subscriptions.forEach((s) => s.then((stop) => stop()));
    };
  }, [refreshMeetings, loadDoc, loadModels]);

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
  const working =
    state.phase === "transcribing" || state.phase === "distilling" ? state : null;
  const mmss = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(
    elapsed % 60,
  ).padStart(2, "0")}`;

  const q = query.trim().toLowerCase();
  const filtered = q
    ? meetings.filter((m) => m.title.toLowerCase().includes(q))
    : meetings;

  useEffect(() => {
    if (filtered.length === 0) return;
    if (!selected || !filtered.some((m) => m.dir === selected)) {
      setSelected(filtered[0].dir);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, meetings]);

  const current = filtered.find((m) => m.dir === selected) ?? null;

  const busyDir = (dir: string) =>
    (state.phase === "transcribing" || state.phase === "distilling") &&
    state.dir === dir;

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

  const toggleAction = (item: ActionItem) => {
    if (!current || !doc?.notes) return;
    const next = toggleTaskLine(doc.notes, item.lineIndex);
    setDoc({ ...doc, notes: next });
    invoke("update_notes", { dir: current.dir, content: next }).catch(() =>
      loadDoc(current.dir),
    );
  };

  if (onboarding) {
    return (
      <Onboarding
        onDone={() => {
          localStorage.setItem("distil-onboarded", "1");
          setOnboarding(false);
        }}
      />
    );
  }

  const dateline = current
    ? [
        stamp(current.startedAt, true),
        durationLabel(current.durationSecs),
        doc?.meta?.ollamaModel
          ? `distilled by ${doc.meta.ollamaModel}`
          : current.hasTranscript
            ? "transcript only"
            : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : "";

  const workingLabel = (() => {
    if (download && (download.status === "started" || download.status === "progress")) {
      return `Downloading ${download.model ?? "model"}… ${download.percent ?? 0}%`;
    }
    if (!working) return null;
    const name = basename(working.dir);
    return working.phase === "transcribing"
      ? `Transcribing ${name} — runs locally, this view updates the moment it's done.`
      : `Distilling notes for ${name}…`;
  })();

  const sections = doc?.notes ? splitSections(doc.notes) : [];
  const actionItems = doc?.notes ? parseActionItems(doc.notes) : [];
  const utterances = doc?.transcript ? parseTranscript(doc.transcript) : [];

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="traffic-strip" data-tauri-drag-region />
        <div className="brand-row">
          <span className="brand-tile">
            <Mic size={15} strokeWidth={2} />
          </span>
          <span className="brand-name">Distil</span>
        </div>
        <div className="rec-wrap">
          <button
            className={`ui-btn ${recording ? "ui-btn-destructive" : "ui-btn-default"}`}
            style={{ width: "100%" }}
            disabled={stopping}
            onClick={() => invoke("toggle_recording")}
          >
            <span className="rec-dot" data-pulse={recording} />
            {recording ? `Stop · ${mmss}` : stopping ? "Stopping…" : "Start recording"}
          </button>
        </div>
        <nav className="side-nav">
          <button
            className="nav-item"
            data-on={nav === "library"}
            onClick={() => setNav("library")}
          >
            <List size={16} strokeWidth={2} /> Library
          </button>
          <button
            className="nav-item"
            data-on={nav === "settings"}
            onClick={() => setNav("settings")}
          >
            <SlidersHorizontal size={16} strokeWidth={2} /> Settings
          </button>
        </nav>
        <div className="nav-section">Meetings</div>
        <div className="meeting-list">
          {filtered.map((m) => {
            const dotColor = busyDir(m.dir)
              ? "var(--sky-500)"
              : m.hasNotes
                ? "var(--teal-500)"
                : "var(--gray-300)";
            const sub = [stamp(m.startedAt, false), durationLabel(m.durationSecs)]
              .filter(Boolean)
              .join(" · ");
            return (
              <button
                key={m.dir}
                className="meeting-item"
                data-on={m.dir === selected}
                onClick={() => {
                  setSelected(m.dir);
                  setNav("library");
                }}
              >
                <span className="meeting-head">
                  <span className="meeting-title">{m.title}</span>
                  <span
                    className="meeting-dot"
                    data-pulse={busyDir(m.dir)}
                    style={{ background: dotColor }}
                  />
                </span>
                {sub && <span className="meeting-sub">{sub}</span>}
              </button>
            );
          })}
          {filtered.length === 0 && q && (
            <p className="no-results">No meetings match “{query.trim()}”.</p>
          )}
        </div>
        {upcoming && !recording && (
          <div className="upgrade-card">
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                opacity: 0.8,
                marginBottom: 4,
              }}
            >
              Up next
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 2 }}>
              {upcoming.title}
            </div>
            <div style={{ fontSize: 12.5, opacity: 0.85, marginBottom: 12 }}>
              Today at {upcoming.startsAtDisplay}
            </div>
            <button
              className="ui-btn ui-btn-sm"
              style={{ background: "#fff", color: "var(--sky-700)", width: "100%" }}
              onClick={() => invoke("record_upcoming")}
            >
              Record this meeting
            </button>
          </div>
        )}
      </aside>

      <div className="main">
        {/* Draggable: the only other drag region is the 38px strip in the
            sidebar, most of which the traffic lights occupy — leaving nowhere
            obvious to grab. Interactive children keep their own events; a drag
            only starts when the mousedown target is the bar itself. */}
        <div className="topbar" data-tauri-drag-region>
          <div className="search-wrap">
            <span className="search-icon">
              <Search size={15} strokeWidth={2} />
            </span>
            <input
              className="ui-input search-input"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search meetings…"
            />
          </div>
          <div className="badges">
            {recording && (
              <span className="ui-badge badge-rec">
                <span className="meeting-dot" data-pulse="true" />
                REC {mmss}
              </span>
            )}
            <span className="ui-badge badge-mono">
              {settings?.transcriber === "parakeet"
                ? "parakeet · tdt-0.6b-v3"
                : `whisper · ${settings?.whisperModel ?? "…"}`}
            </span>
            <span className="ui-badge badge-mono">
              {settings
                ? ((settings.engine === "openai" ? settings.openaiModel : settings.ollamaModel) ??
                  `${settings.engine === "openai" ? "server" : "ollama"} · auto`)
                : "…"}
            </span>
          </div>
        </div>

        {(workingLabel || state.phase === "error") && (
          <div
            className={`ui-alert working-alert ${
              state.phase === "error" ? "ui-alert-error" : "ui-alert-info"
            }`}
            data-error={state.phase === "error"}
          >
            <span className="dot" />
            <span>{state.phase === "error" ? state.message : workingLabel}</span>
          </div>
        )}

        {nav === "settings" && settings ? (
          <div className="content">
            <SettingsPane
              settings={settings}
              notesModels={models}
              notesError={modelsError}
              onSettingsChange={setSettings}
              onReloadModels={loadModels}
            />
          </div>
        ) : (
          <div className="content">
            {!current && (
              <div className="empty-wrap">
                <span className="empty-tile" data-rec={recording}>
                  <Mic size={22} strokeWidth={2} />
                </span>
                <h2 className="app-h2">
                  {recording ? "Recording…" : "No meetings yet"}
                </h2>
                <p>
                  {recording
                    ? "Capturing your mic and system audio as separate streams. Stop from the sidebar or the menu bar when the meeting ends."
                    : "Start a recording when your next meeting begins. Distil transcribes it and writes the notes, all on this Mac."}
                </p>
                {!recording && (
                  <button
                    className="ui-btn ui-btn-default"
                    onClick={() => invoke("toggle_recording")}
                  >
                    Start recording
                  </button>
                )}
              </div>
            )}
            {current && (
              <div className="article">
                <div className="article-head">
                  <div style={{ minWidth: 0 }}>
                    <h1 className="app-h1 article-title">{current.title}</h1>
                    <div className="dateline">{dateline}</div>
                  </div>
                  <div className="article-actions">
                    <button
                      className="ui-btn ui-btn-ghost ui-btn-sm"
                      title="Reveal in Finder"
                      onClick={() => invoke("reveal_meeting", { dir: current.dir })}
                    >
                      <Folder size={15} strokeWidth={2} />
                    </button>
                    <button
                      className="ui-btn ui-btn-ghost ui-btn-sm"
                      title="Copy"
                      disabled={!(view === "notes" ? doc?.notes : doc?.transcript)}
                      onClick={copyCurrent}
                    >
                      {copied ? (
                        <Check size={15} strokeWidth={2} />
                      ) : (
                        <Copy size={15} strokeWidth={2} />
                      )}
                    </button>
                    <button
                      className="ui-btn ui-btn-ghost ui-btn-sm"
                      title="Delete"
                      style={confirmingDelete ? { color: "var(--coral-600)" } : undefined}
                      onClick={deleteCurrent}
                    >
                      <Trash2 size={15} strokeWidth={2} />
                      {confirmingDelete && (
                        <span style={{ fontSize: 12, fontWeight: 600 }}>Sure?</span>
                      )}
                    </button>
                  </div>
                </div>

                <div className="tab-row">
                  <div className="ui-tabs">
                    <button
                      className="ui-tab"
                      data-on={view === "notes"}
                      onClick={() => setView("notes")}
                      disabled={!doc?.notes}
                    >
                      Notes
                    </button>
                    <button
                      className="ui-tab"
                      data-on={view === "transcript"}
                      onClick={() => setView("transcript")}
                      disabled={!doc?.transcript}
                    >
                      Transcript
                    </button>
                  </div>
                  {current.hasTranscript && !current.hasNotes && !busyDir(current.dir) && (
                    <button
                      className="ui-btn ui-btn-default ui-btn-sm"
                      disabled={!!working}
                      onClick={() => invoke("distil_meeting", { dir: current.dir })}
                    >
                      Generate notes
                    </button>
                  )}
                </div>

                {view === "notes" && doc?.notes && (
                  <div className="cards">
                    {sections.map((section) =>
                      /action items/i.test(section.label) && actionItems.length > 0 ? (
                        <div key={section.label} className="ui-card section-card">
                          <div className="card-label">{section.label}</div>
                          <div className="action-list">
                            {actionItems.map((item) => (
                              <button
                                key={item.lineIndex}
                                className="action-row"
                                onClick={() => toggleAction(item)}
                              >
                                <span className="ui-check" data-on={item.checked}>
                                  {item.checked && <Check size={12} strokeWidth={2.5} />}
                                </span>
                                <span className="ui-badge owner-badge">{item.owner}</span>
                                <span className="action-task" data-done={item.checked}>
                                  {item.task}
                                </span>
                              </button>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <div key={section.label} className="ui-card section-card">
                          <div className="card-label">{section.label}</div>
                          <div
                            className="card-md"
                            dangerouslySetInnerHTML={{ __html: render(section.body) }}
                          />
                        </div>
                      ),
                    )}
                  </div>
                )}

                {view === "transcript" && doc?.transcript && (
                  <div className="ui-card transcript-card">
                    {utterances.length > 0 ? (
                      utterances.map((u, i) => (
                        <div key={i} className="utterance">
                          <span className="ui-avatar" data-them={u.who !== "Me"}>
                            {initialsFor(u.who)}
                          </span>
                          <div style={{ minWidth: 0 }}>
                            <div className="who">{u.who}</div>
                            <div className="utter-text">{u.text}</div>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div
                        className="card-md"
                        dangerouslySetInnerHTML={{ __html: render(doc.transcript) }}
                      />
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
