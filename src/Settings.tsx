import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Check, ChevronDown } from "lucide-react";

type WhisperModel = {
  name: string;
  description: string;
  sizeMb: number;
  downloaded: boolean;
  selected: boolean;
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

type Permissions = { microphone: string; calendar?: string };

/// Settings pane name -> the field it reports under in a permissions report.
const PERM_KEYS: Record<string, keyof Permissions> = {
  microphone: "microphone",
  calendar: "calendar",
};

type NotesPrompt = { text: string; isCustom: boolean; defaultText: string };

function Switch({ on, onToggle }: { on: boolean; onToggle: (next: boolean) => void }) {
  return (
    <button className="ui-switch" data-on={on} onClick={() => onToggle(!on)}>
      <span className="knob" />
    </button>
  );
}

export function SettingsPane({
  settings,
  ollamaModels,
  ollamaError,
  onSettingsChange,
  onReloadModels,
}: {
  settings: Settings;
  ollamaModels: OllamaModel[];
  ollamaError: boolean;
  onSettingsChange: (next: Settings) => void;
  onReloadModels: () => void;
}) {
  const [whisperModels, setWhisperModels] = useState<WhisperModel[]>([]);
  const [permissions, setPermissions] = useState<Permissions | null>(null);
  const [autostart, setAutostart] = useState(false);
  const [prompt, setPrompt] = useState<NotesPrompt | null>(null);
  const [promptDraft, setPromptDraft] = useState("");
  const [promptSaved, setPromptSaved] = useState(false);

  const refreshPermissions = useCallback(() => {
    invoke<Permissions>("permissions_status")
      .then(setPermissions)
      .catch(() => setPermissions(null));
  }, []);

  useEffect(() => {
    invoke<WhisperModel[]>("list_whisper_models")
      .then(setWhisperModels)
      .catch(() => {});
    invoke<boolean>("get_autostart").then(setAutostart).catch(() => {});
    invoke<NotesPrompt>("get_notes_prompt")
      .then((p) => {
        setPrompt(p);
        setPromptDraft(p.text);
      })
      .catch(() => {});
    refreshPermissions();
  }, [refreshPermissions, settings.whisperModel]);

  const applyPrompt = (text: string) => {
    invoke<NotesPrompt>("set_notes_prompt", { text })
      .then((p) => {
        setPrompt(p);
        setPromptDraft(p.text);
        setPromptSaved(true);
        window.setTimeout(() => setPromptSaved(false), 1500);
      })
      .catch(() => {});
  };

  const granted = (status: string | undefined) => status === "granted";

  // Same two-step as onboarding: prompt first, because until Distil has asked
  // once macOS won't list it in Privacy & Security for anyone to toggle.
  const allow = (pane: string) => {
    const openPane = () =>
      invoke("open_privacy_settings", { pane }).catch(() => {});
    invoke<Permissions>("request_permission", { pane })
      .then((next) => {
        setPermissions(next);
        if (!granted(next[PERM_KEYS[pane]])) openPane();
      })
      .catch(openPane);
  };

  const permLine = (label: string, status: string | undefined, pane: string) => (
    <div className="perm-line">
      <span>{label}</span>
      {granted(status) ? (
        <span className="ui-badge badge-teal">
          <Check size={11} strokeWidth={2.5} /> Granted
        </span>
      ) : (
        <button
          className="ui-btn ui-btn-outline ui-btn-sm"
          onClick={() => allow(pane)}
        >
          Allow
        </button>
      )}
    </div>
  );

  return (
    <div className="settings-col">
      <div className="page-head">
        <h1 className="app-h1">Settings</h1>
        <div className="page-sub">
          Everything runs on this Mac unless you explicitly opt in.
        </div>
      </div>

      <div className="ui-card section-card">
        <h3 className="app-h2 card-h">Transcription</h3>
        <p className="card-sub">
          On-device with whisper.cpp. Models download automatically on first use.
        </p>
        <div className="model-rows">
          {whisperModels.map((m) => (
            <button
              key={m.name}
              className="model-row"
              data-on={m.selected}
              onClick={() => {
                invoke("set_whisper_model", { model: m.name }).catch(() => {});
                onSettingsChange({ ...settings, whisperModel: m.name });
              }}
            >
              <span className="radio-dot" />
              <span className="model-main">
                <span className="model-name">{m.name}</span>
                <span className="model-desc">
                  {" "}
                  — {m.description} · {m.sizeMb} MB
                </span>
              </span>
              {m.downloaded && <span className="ui-badge badge-teal">Downloaded</span>}
            </button>
          ))}
        </div>
        <div className="switch-row" style={{ marginTop: 16 }}>
          <Switch
            on={settings.diarizeSpeakers}
            onToggle={(enable) => {
              invoke("set_diarize_speakers", { enable }).catch(() => {});
              onSettingsChange({ ...settings, diarizeSpeakers: enable });
            }}
          />
          <div className="switch-main">
            <div className="switch-title">Identify multiple speakers</div>
            <div className="switch-desc">
              Clusters distinct voices into Them 1 / Them 2 / … and lets the
              distiller infer names from the call. First use downloads ~45 MB of
              local models.
            </div>
          </div>
        </div>
      </div>

      <div className="ui-card section-card">
        <h3 className="app-h2 card-h">Distillation</h3>
        <p className="card-sub">Notes are written by a local model via Ollama.</p>
        {ollamaError ? (
          <div className="ui-alert ui-alert-error" style={{ padding: "10px 14px" }}>
            <span style={{ fontSize: 13 }}>
              Ollama isn't reachable — start the Ollama app (or `ollama serve`).
            </span>
          </div>
        ) : (
          <div className="ui-select-wrap" style={{ maxWidth: 320 }}>
            <select
              className="ui-select"
              value={settings.ollamaModel ?? ""}
              onChange={(event) => {
                const model = event.target.value || null;
                invoke("set_ollama_model", { model }).catch(() => {});
                onSettingsChange({ ...settings, ollamaModel: model });
              }}
            >
              <option value="">Auto — largest local model</option>
              {ollamaModels.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                  {m.isCloud
                    ? " · CLOUD"
                    : m.parameterSize
                      ? ` · ${m.parameterSize}B`
                      : ""}
                </option>
              ))}
            </select>
            <span className="chev">
              <ChevronDown size={15} strokeWidth={2} />
            </span>
          </div>
        )}
        <div className="switch-row" style={{ marginTop: 16 }}>
          <Switch
            on={settings.allowCloudModels}
            onToggle={(allow) => {
              invoke("set_allow_cloud_models", { allow })
                .then(() => onReloadModels())
                .catch(() => {});
              onSettingsChange({
                ...settings,
                allowCloudModels: allow,
                ollamaModel:
                  !allow && settings.ollamaModel?.endsWith("-cloud")
                    ? null
                    : settings.ollamaModel,
              });
            }}
          />
          <div className="switch-main">
            <div className="switch-title">Allow cloud Ollama models</div>
            <div className="switch-desc">
              Models badged <span className="ui-badge badge-cloud">CLOUD</span> run
              on Ollama's servers — transcripts sent to them leave this Mac. Never
              chosen automatically.
            </div>
          </div>
        </div>
      </div>

      <div className="ui-card section-card">
        <h3 className="app-h2 card-h" style={{ marginBottom: 14 }}>
          Storage &amp; launch
        </h3>
        <div className="switch-stack">
          <div className="switch-row">
            <Switch
              on={settings.keepAudio}
              onToggle={(keep) => {
                invoke("set_keep_audio", { keep }).catch(() => {});
                onSettingsChange({ ...settings, keepAudio: keep });
              }}
            />
            <div className="switch-main">
              <div className="switch-title">Keep audio after transcription</div>
              <div className="switch-desc">
                Off by default — recordings are the most sensitive artifact. Notes
                live in ~/Documents/Distil as plain Markdown.
              </div>
            </div>
          </div>
          <div className="switch-row">
            <Switch
              on={autostart}
              onToggle={(enable) => {
                setAutostart(enable);
                invoke("set_autostart", { enable }).catch(() => setAutostart(!enable));
              }}
            />
            <div className="switch-main">
              <div className="switch-title">Start Distil at login</div>
              <div className="switch-desc">
                Lives in the menu bar, ready when meetings start.
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="ui-card section-card">
        <h3 className="app-h2 card-h">Notes prompt</h3>
        <p className="card-sub">
          Your standing instruction to the model. {"{transcript}"} marks where the
          transcript goes; clearing the text restores the default.
        </p>
        <textarea
          className="ui-textarea prompt-editor"
          rows={6}
          spellCheck={false}
          value={promptDraft}
          onChange={(event) => setPromptDraft(event.target.value)}
        />
        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <button
            className="ui-btn ui-btn-default ui-btn-sm"
            disabled={prompt !== null && promptDraft === prompt.text}
            onClick={() => applyPrompt(promptDraft)}
          >
            {promptSaved ? "Saved ✓" : "Save prompt"}
          </button>
          {prompt?.isCustom && (
            <button
              className="ui-btn ui-btn-ghost ui-btn-sm"
              onClick={() => applyPrompt("")}
            >
              Reset to default
            </button>
          )}
        </div>
      </div>

      <div className="ui-card section-card">
        <h3 className="app-h2 card-h" style={{ marginBottom: 14 }}>
          Permissions
        </h3>
        <div className="perm-lines">
          {permLine("Microphone", permissions?.microphone, "microphone")}
          {permLine("Calendar (meeting prompts)", permissions?.calendar, "calendar")}
        </div>
      </div>
    </div>
  );
}
