import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Calendar, Check, Mic } from "lucide-react";

type Permissions = { microphone: string; calendar?: string };

const STEPS = [
  {
    key: "microphone" as const,
    pane: "microphone",
    title: "Microphone",
    desc: "Your side of the conversation, as its own stream.",
    Icon: Mic,
  },
  {
    key: "calendar" as const,
    pane: "calendar",
    title: "Calendar",
    desc: "Read locally, no OAuth. Distil offers to record when a meeting starts.",
    Icon: Calendar,
  },
];

export function Onboarding({ onDone }: { onDone: () => void }) {
  const [perms, setPerms] = useState<Permissions | null>(null);

  const refresh = useCallback(() => {
    invoke<Permissions>("permissions_status")
      .then(setPerms)
      .catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const poll = window.setInterval(refresh, 2000);
    return () => window.clearInterval(poll);
  }, [refresh]);

  const granted = (key: (typeof STEPS)[number]["key"]) =>
    (perms?.[key] ?? "") === "granted";
  const allGranted = STEPS.every((s) => granted(s.key));

  // Raise the real prompt. Only fall back to System Settings if that didn't
  // grant it — which is also the point at which Settings starts listing Distil,
  // since the request is what registers the app with TCC.
  const [pending, setPending] = useState<string | null>(null);
  const allow = (step: (typeof STEPS)[number]) => {
    setPending(step.key);
    const openPane = () =>
      invoke("open_privacy_settings", { pane: step.pane }).catch(() => {});
    invoke<Permissions>("request_permission", { pane: step.pane })
      .then((next) => {
        setPerms(next);
        if ((next[step.key] ?? "") !== "granted") openPane();
      })
      .catch(openPane)
      .finally(() => setPending(null));
  };

  return (
    <div className="onboard-wrap">
      <div className="onboard-strip" data-tauri-drag-region />
      <div className="ui-card onboard-card">
        <div className="onboard-head">
          <span className="onboard-tile">
            <Mic size={22} strokeWidth={2} />
          </span>
          <div>
            <div className="app-h1 onboard-title">Welcome to Distil</div>
            <div className="onboard-sub">Meeting notes that never leave your Mac</div>
          </div>
        </div>
        <p className="onboard-intro">
          Distil records your meetings, transcribes them on-device and writes the
          notes with a local model. macOS asks for two permissions first.
        </p>
        <div className="perm-steps">
          {STEPS.map((step) => (
            <div key={step.key} className="perm-step">
              <span className="perm-tile" data-on={granted(step.key)}>
                <step.Icon size={18} strokeWidth={2} />
              </span>
              <span className="perm-copy">
                <span className="perm-title">{step.title}</span>
                <span className="perm-desc">{step.desc}</span>
              </span>
              {granted(step.key) ? (
                <span className="ui-badge badge-teal">
                  <Check size={11} strokeWidth={2.5} /> Granted
                </span>
              ) : (
                <button
                  className="ui-btn ui-btn-outline ui-btn-sm"
                  disabled={pending !== null}
                  onClick={() => allow(step)}
                >
                  {pending === step.key ? "Waiting…" : "Allow"}
                </button>
              )}
            </div>
          ))}
        </div>
        {allGranted ? (
          <button
            className="ui-btn ui-btn-default"
            style={{ width: "100%", marginTop: 22 }}
            onClick={onDone}
          >
            Open Distil
          </button>
        ) : (
          <>
            <p className="onboard-footnote">
              Nothing is uploaded, ever — recordings and notes stay in
              ~/Documents/Distil.
            </p>
            <button className="onboard-skip" onClick={onDone}>
              Skip for now
            </button>
          </>
        )}
      </div>
    </div>
  );
}
