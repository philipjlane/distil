import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { ArrowDown, Mic } from "lucide-react";
import "./App.css";

type Stream = "me" | "them";
type Line = { startMs: number; endMs: number; text: string };
type TaggedLine = Line & { stream: Stream };
type Status = "preparing" | "live" | "unavailable" | "finished";
type LiveEvent = { stream: Stream; committed: Line[]; provisional: Line[] };
type Snapshot = { status: Status; committed: TaggedLine[]; provisional: TaggedLine[] };
type Provisional = Record<Stream, Line[]>;

type Part = { text: string; provisional: boolean; key: string };
type Block = { stream: Stream; startMs: number; parts: Part[] };

const NO_PROVISIONAL: Provisional = { me: [], them: [] };
const BUSY = ["recording", "stopping", "transcribing", "distilling"];
/** How long the finished panel lingers before it closes itself. */
const CLOSE_AFTER_MS = 4000;
/** Within this many pixels of the bottom counts as following the latest text. */
const FOLLOW_SLACK_PX = 24;

function mmss(ms: number): string {
  const total = Math.floor(ms / 1000);
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

/** Committed lines and each stream's moving tail, interleaved by start time
 * and joined into one block per run of the same speaker. */
function toBlocks(committed: TaggedLine[], provisional: Provisional): Block[] {
  const items = [
    ...committed.map((line) => ({ ...line, provisional: false })),
    ...provisional.me.map((line) => ({ ...line, stream: "me" as const, provisional: true })),
    ...provisional.them.map((line) => ({ ...line, stream: "them" as const, provisional: true })),
  ].sort((a, b) => a.startMs - b.startMs);
  const blocks: Block[] = [];
  for (const item of items) {
    const part = {
      text: item.text,
      provisional: item.provisional,
      key: `${item.stream}-${item.startMs}-${item.endMs}-${item.provisional}`,
    };
    const last = blocks[blocks.length - 1];
    if (last && last.stream === item.stream) {
      last.parts.push(part);
    } else {
      blocks.push({ stream: item.stream, startMs: item.startMs, parts: [part] });
    }
  }
  return blocks;
}

function footerText(phase: string, status: Status): string {
  switch (phase) {
    case "recording":
      if (status === "preparing") return "Preparing the live preview…";
      if (status === "unavailable") return "Live preview unavailable. The recording is unaffected.";
      return "Rough, English-only preview";
    case "stopping":
      return "Stopping…";
    case "transcribing":
      return "Transcribing the full recording…";
    case "distilling":
      return "Writing your notes…";
    case "error":
      return "Something went wrong. See the main window.";
    default:
      return "Done. The notes are in the library.";
  }
}

export function LivePanel() {
  const [status, setStatus] = useState<Status>("preparing");
  const [committed, setCommitted] = useState<TaggedLine[]>([]);
  const [provisional, setProvisional] = useState<Provisional>(NO_PROVISIONAL);
  // The window only opens for a recording, so that is the best first guess.
  const [phase, setPhase] = useState("recording");
  const [following, setFollowing] = useState(true);

  const scroller = useRef<HTMLDivElement | null>(null);
  const followingRef = useRef(true);
  // A committed line is sent once, but the snapshot read at start-up can
  // overlap events that arrived meanwhile, so lines are told apart by place.
  const seen = useRef(new Set<string>());
  const heardStatus = useRef(false);
  const heardTranscript = useRef(false);
  const wasBusy = useRef(false);

  const addCommitted = useCallback((lines: TaggedLine[]) => {
    const fresh = lines.filter((line) => {
      const key = `${line.stream}:${line.startMs}:${line.endMs}`;
      if (seen.current.has(key)) return false;
      seen.current.add(key);
      return true;
    });
    if (fresh.length > 0) setCommitted((all) => [...all, ...fresh]);
  }, []);

  useEffect(() => {
    const unlisten = [
      listen<LiveEvent>("live-transcript", (event) => {
        heardTranscript.current = true;
        const { stream, committed: lines, provisional: tail } = event.payload;
        addCommitted(lines.map((line) => ({ ...line, stream })));
        setProvisional((all) => ({ ...all, [stream]: tail }));
      }),
      listen<Status>("live-status", (event) => {
        heardStatus.current = true;
        if (event.payload === "preparing") {
          // A new recording started while this window was still open.
          seen.current.clear();
          setCommitted([]);
          setProvisional(NO_PROVISIONAL);
        }
        setStatus(event.payload);
      }),
      listen<{ phase: string }>("pipeline-state", (event) => setPhase(event.payload.phase)),
    ];
    // The window opens a moment after the preview starts, so ask what it has
    // said so far. Anything that arrived since is newer and wins.
    Promise.all(unlisten)
      .then(() => invoke<Snapshot | null>("live_snapshot"))
      .then((snapshot) => {
        if (!snapshot) return;
        if (!heardStatus.current) setStatus(snapshot.status);
        addCommitted(snapshot.committed);
        if (!heardTranscript.current) {
          setProvisional({
            me: snapshot.provisional.filter((line) => line.stream === "me"),
            them: snapshot.provisional.filter((line) => line.stream === "them"),
          });
        }
      })
      .catch(() => {});
    invoke<{ phase: string }>("current_state")
      .then((state) => setPhase(state.phase))
      .catch(() => {});
    return () => {
      unlisten.forEach((stop) => stop.then((fn) => fn()));
    };
  }, [addCommitted]);

  // Once the notes are done there is nothing left to watch.
  useEffect(() => {
    if (BUSY.includes(phase)) {
      wasBusy.current = true;
      return;
    }
    if (phase === "idle" && wasBusy.current) {
      const timer = window.setTimeout(() => {
        invoke("close_live").catch(() => {});
      }, CLOSE_AFTER_MS);
      return () => window.clearTimeout(timer);
    }
  }, [phase]);

  const blocks = useMemo(() => toBlocks(committed, provisional), [committed, provisional]);

  // Follow the latest text until the reader scrolls up, and pick up again when
  // they scroll back to the bottom.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && followingRef.current) el.scrollTop = el.scrollHeight;
  }, [blocks]);

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < FOLLOW_SLACK_PX;
    followingRef.current = atBottom;
    setFollowing(atBottom);
  };

  const jumpToLatest = () => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
    followingRef.current = true;
    setFollowing(true);
  };

  const recording = phase === "recording";
  const empty =
    status === "preparing"
      ? "Getting ready…"
      : status === "unavailable"
        ? "No preview for this recording."
        : status === "finished"
          ? "No speech was picked up."
          : "Listening for speech…";

  return (
    <div className="live-panel">
      <div className="live-head">
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <span className="tray-tile">
            <Mic size={12} strokeWidth={2} />
          </span>
          <span className="tray-name">Live transcript</span>
        </div>
        {recording ? (
          <span className="ui-badge badge-rec">
            <span className="meeting-dot" data-pulse="true" style={{ width: 6, height: 6 }} />
            REC
          </span>
        ) : (
          <span className="ui-badge" style={{ background: "var(--surface-sunk)", color: "var(--gray-600)" }}>
            Preview
          </span>
        )}
      </div>

      <div className="live-scroll" ref={scroller} onScroll={onScroll}>
        {blocks.length === 0 ? (
          <p className="live-empty">{empty}</p>
        ) : (
          blocks.map((block) => (
            <div className="live-block" data-stream={block.stream} key={`${block.stream}-${block.startMs}`}>
              <div className="live-who">
                <span>{block.stream === "me" ? "Me" : "Them"}</span>
                <span className="live-time">{mmss(block.startMs)}</span>
              </div>
              <p className="live-text">
                {block.parts.map((part) => (
                  <span className={part.provisional ? "live-prov" : undefined} key={part.key}>
                    {part.text}{" "}
                  </span>
                ))}
              </p>
            </div>
          ))
        )}
        {!following && (
          <div className="live-latest-wrap">
            <button className="ui-btn ui-btn-outline ui-btn-sm live-latest" onClick={jumpToLatest}>
              <ArrowDown size={13} strokeWidth={2} />
              Latest
            </button>
          </div>
        )}
      </div>

      <div className="live-foot" data-warn={status === "unavailable" && recording}>
        {footerText(phase, status)}
      </div>
    </div>
  );
}
