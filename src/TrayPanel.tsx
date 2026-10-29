import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow, LogicalSize } from "@tauri-apps/api/window";
import { Mic } from "lucide-react";
import "./App.css";

type UiState = { phase: string; dir?: string };
type Meeting = { dir: string; title: string; startedAt: string | null };
type UpcomingMeeting = { title: string; startsAtDisplay: string } | null;

function latestStamp(iso: string | null): string {
  if (!iso) return "";
  const date = new Date(iso);
  const now = new Date();
  if (date.toDateString() === now.toDateString()) {
    return date.getHours() < 12 ? "this morning" : "today";
  }
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (date.toDateString() === yesterday.toDateString()) return "yesterday";
  return date.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" });
}

export function TrayPanel() {
  const [state, setState] = useState<UiState>({ phase: "idle" });
  const [upcoming, setUpcoming] = useState<UpcomingMeeting>(null);
  const [detected, setDetected] = useState<string | null>(null);
  const [latest, setLatest] = useState<Meeting | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);

  const refresh = useCallback(() => {
    invoke<UiState>("current_state").then(setState).catch(() => {});
    invoke<UpcomingMeeting>("upcoming_meeting").then(setUpcoming).catch(() => {});
    invoke<Meeting[]>("list_meetings")
      .then((list) => setLatest(list[0] ?? null))
      .catch(() => {});
    invoke<number | null>("recording_elapsed").then(setElapsed).catch(() => {});
  }, []);

  useEffect(() => {
    document.documentElement.style.background = "transparent";
    document.body.style.background = "transparent";
    refresh();
    const poll = window.setInterval(refresh, 1000);
    const unlisten = listen<UiState>("pipeline-state", (event) => {
      setState(event.payload);
      refresh();
    });
    return () => {
      window.clearInterval(poll);
      unlisten.then((stop) => stop());
    };
  }, [refresh]);

  useEffect(() => {
    invoke<string | null>("detected_call").then(setDetected).catch(() => {});
    const unlisten = listen<string | null>("meeting-detected", (event) => {
      setDetected(event.payload);
    });
    return () => {
      unlisten.then((stop) => stop());
    };
  }, []);

  useEffect(() => {
    const height = panelRef.current?.offsetHeight;
    if (height) {
      getCurrentWindow().setSize(new LogicalSize(300, height + 12)).catch(() => {});
    }
  });

  const recording = state.phase === "recording";
  const mmss =
    elapsed !== null
      ? `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`
      : "";

  const statusBadge = recording ? (
    <span className="ui-badge badge-rec">REC {mmss}</span>
  ) : state.phase === "transcribing" || state.phase === "distilling" ? (
    <span className="ui-badge" style={{ background: "var(--sky-50)", color: "var(--sky-700)" }}>
      Working…
    </span>
  ) : (
    <span className="ui-badge" style={{ background: "var(--surface-sunk)", color: "var(--gray-600)" }}>
      Idle
    </span>
  );

  return (
    <div className="tray-panel" ref={panelRef}>
      <div className="tray-head">
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <span className="tray-tile">
            <Mic size={12} strokeWidth={2} />
          </span>
          <span className="tray-name">Distil</span>
        </div>
        {statusBadge}
      </div>
      <div className="ui-menu-sep" />
      <button
        className="ui-menu-item"
        style={{ fontWeight: 600 }}
        onClick={() => {
          invoke("toggle_recording");
          if (!recording) getCurrentWindow().hide().catch(() => {});
        }}
      >
        <span
          className="meeting-dot"
          data-pulse={recording}
          style={{ width: 8, height: 8, background: "var(--coral-500)" }}
        />
        {recording ? `Stop recording · ${mmss}` : "Start recording"}
      </button>
      {detected && !recording && (
        <>
          <div className="tray-label">Now</div>
          <button
            className="ui-menu-item"
            onClick={() => {
              invoke("record_detected");
              getCurrentWindow().hide().catch(() => {});
            }}
          >
            <span
              style={{
                flex: 1,
                textAlign: "left",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {upcoming ? `${upcoming.title} on ${detected}` : `${detected} call in progress`}
            </span>
            <span className="tray-action">Record</span>
          </button>
        </>
      )}
      {upcoming && !detected && !recording && (
        <>
          <div className="tray-label">Up next</div>
          <button
            className="ui-menu-item"
            onClick={() => {
              invoke("record_upcoming");
              getCurrentWindow().hide().catch(() => {});
            }}
          >
            <span style={{ flex: 1, textAlign: "left" }}>
              {upcoming.title} — {upcoming.startsAtDisplay}
            </span>
            <span className="tray-action">Record</span>
          </button>
        </>
      )}
      {latest && (
        <>
          <div className="tray-label">Latest</div>
          <button
            className="ui-menu-item"
            onClick={() => invoke("open_meeting", { dir: latest.dir })}
          >
            <span
              style={{
                flex: 1,
                textAlign: "left",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {latest.title} — {latestStamp(latest.startedAt)}
            </span>
            <span className="tray-action">Notes</span>
          </button>
        </>
      )}
      <div className="ui-menu-sep" />
      <button className="ui-menu-item" onClick={() => invoke("open_settings")}>
        Settings…
      </button>
      <button className="ui-menu-item" onClick={() => invoke("quit")}>
        Quit Distil
      </button>
    </div>
  );
}
