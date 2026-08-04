import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

type WhisperModel = {
  name: string;
  description: string;
  sizeMb: number;
  downloaded: boolean;
  selected: boolean;
};

type OllamaModel = { name: string; parameterSize: number | null };

type Settings = {
  whisperModel: string;
  keepAudio: boolean;
  ollamaUrl: string;
  ollamaModel: string | null;
};

type Permissions = { microphone: string; screenRecording: string; calendar?: string };

export function SettingsPane({
  settings,
  ollamaModels,
  ollamaError,
  onSettingsChange,
}: {
  settings: Settings;
  ollamaModels: OllamaModel[];
  ollamaError: boolean;
  onSettingsChange: (next: Settings) => void;
}) {
  const [whisperModels, setWhisperModels] = useState<WhisperModel[]>([]);
  const [permissions, setPermissions] = useState<Permissions | null>(null);

  const refreshPermissions = useCallback(() => {
    invoke<Permissions>("permissions_status")
      .then(setPermissions)
      .catch(() => setPermissions(null));
  }, []);

  useEffect(() => {
    invoke<WhisperModel[]>("list_whisper_models")
      .then(setWhisperModels)
      .catch(() => {});
    refreshPermissions();
  }, [refreshPermissions, settings.whisperModel]);

  const granted = (status: string | undefined) => status === "granted";

  return (
    <div className="settings">
      <section>
        <h3>Transcription</h3>
        <p className="section-hint">
          Runs on-device with whisper.cpp. A model not yet on disk downloads
          automatically the first time it's used.
        </p>
        {whisperModels.map((m) => (
          <label key={m.name} className="radio-row">
            <input
              type="radio"
              name="whisper"
              checked={m.selected}
              onChange={() => {
                invoke("set_whisper_model", { model: m.name }).catch(() => {});
                onSettingsChange({ ...settings, whisperModel: m.name });
              }}
            />
            <span className="radio-title">{m.name}</span>
            <span className="radio-desc">
              {m.description} · {m.sizeMb} MB
              {m.downloaded ? " · downloaded" : ""}
            </span>
          </label>
        ))}
      </section>

      <section>
        <h3>Distillation</h3>
        <p className="section-hint">
          Notes are written by a local model via Ollama. Cloud-routed Ollama
          models are excluded by design — nothing leaves this Mac.
        </p>
        {ollamaError ? (
          <p className="perm-warn">
            Ollama isn't reachable — start the Ollama app (or `ollama serve`).
          </p>
        ) : (
          <select
            value={settings.ollamaModel ?? ""}
            onChange={(event) => {
              const model = event.target.value || null;
              invoke("set_ollama_model", { model }).catch(() => {});
              onSettingsChange({ ...settings, ollamaModel: model });
            }}
          >
            <option value="">Auto (largest local model)</option>
            {ollamaModels.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
                {m.parameterSize ? ` · ${m.parameterSize}B` : ""}
              </option>
            ))}
          </select>
        )}
      </section>

      <section>
        <h3>Storage</h3>
        <label className="check-row">
          <input
            type="checkbox"
            checked={settings.keepAudio}
            onChange={(event) => {
              invoke("set_keep_audio", { keep: event.target.checked }).catch(() => {});
              onSettingsChange({ ...settings, keepAudio: event.target.checked });
            }}
          />
          <span>
            Keep audio files after transcription
            <em className="radio-desc">
              Off by default — raw recordings are the most sensitive artifact.
              Meetings live in ~/Documents/Distil as plain Markdown.
            </em>
          </span>
        </label>
      </section>

      <section>
        <h3>Permissions</h3>
        <div className="perm-row">
          <span>Microphone</span>
          {granted(permissions?.microphone) ? (
            <em className="perm-ok">granted</em>
          ) : (
            <button onClick={() => invoke("open_privacy_settings", { pane: "microphone" })}>
              {permissions ? `${permissions.microphone} — open System Settings` : "check…"}
            </button>
          )}
        </div>
        <div className="perm-row">
          <span>Screen &amp; system audio</span>
          {granted(permissions?.screenRecording) ? (
            <em className="perm-ok">granted</em>
          ) : (
            <button onClick={() => invoke("open_privacy_settings", { pane: "screen" })}>
              {permissions
                ? `${permissions.screenRecording} — open System Settings`
                : "check…"}
            </button>
          )}
        </div>
        <div className="perm-row">
          <span>Calendar (meeting prompts)</span>
          {granted(permissions?.calendar) ? (
            <em className="perm-ok">granted</em>
          ) : (
            <button onClick={() => invoke("open_privacy_settings", { pane: "calendar" })}>
              {permissions
                ? `${permissions.calendar ?? "unknown"} — open System Settings`
                : "check…"}
            </button>
          )}
        </div>
        <button className="perm-refresh" onClick={refreshPermissions}>
          Re-check
        </button>
      </section>
    </div>
  );
}
