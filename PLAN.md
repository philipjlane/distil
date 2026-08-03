# Distil — v1 Plan

## What it is

Distil is a **local-first meeting notes app for macOS**. It records your meetings
(microphone + system audio), transcribes them on-device, and distils the transcript
into structured notes — summary, decisions, action items. Nothing leaves your Mac:
transcription runs on whisper.cpp, distillation on a local LLM via Ollama, and the
output is plain Markdown files you own.

## v1 user story

1. You join a meeting (Zoom/Meet/anything). Click the distil menu-bar icon → **Start recording**.
2. Distil captures your mic (you) and system audio (everyone else) as **two separate streams**.
3. Click **Stop** when the meeting ends. Distil transcribes both streams locally,
   merges them by timestamp into a `Me:` / `Them:` attributed transcript, then
   distils it into structured notes.
4. A notification fires; the notes open in the library window. Everything lives in
   `~/Documents/Distil/` as Markdown.

The two-stream capture gives us speaker attribution (you vs. everyone else) for free,
without real diarization.

## Architecture

| Component | Choice | Notes |
|---|---|---|
| Shell | **Tauri 2** | Menu-bar (tray) app, `ActivationPolicy::Accessory` (no Dock icon). React + TS + Vite frontend for library/settings windows. |
| Capture | **Swift sidecar** (`distil-capture`) | Bundled as a Tauri sidecar. ScreenCaptureKit audio-only stream for system audio; AVAudioEngine for mic. Emits 16 kHz mono WAV per stream. Needs mic + screen/system-audio recording permissions (TCC). Isolates all Apple-API complexity into a few hundred lines of Swift. |
| Transcription | **whisper.cpp** via whisper-rs (Metal) | Default model: quantized `large-v3-turbo` (~1.6 GB), auto-downloaded on first run into app data dir. Post-meeting transcription in v1, not live. |
| Distillation | **Ollama** over localhost HTTP | Local models **only** — `*-cloud` Ollama models are excluded; they'd silently break the privacy promise. Runtime picks the best locally-installed model; configurable. Structured prompt → TL;DR, key points, decisions, action items (owner + due date when stated), open questions. |
| Orchestration | Rust state machine | `idle → recording → transcribing → distilling → done/error`; progress pushed to UI via Tauri events. |

Pipeline: tray **Start** → sidecar records two WAVs → **Stop** → whisper transcribes both
→ merge by timestamps (`Me`/`Them`) → Ollama distils → write meeting folder → notify.

## Storage — files are the database

```
~/Documents/Distil/
  2026-08-03T14-30 Weekly sync/
    meta.json          # times, duration, models used, app version
    audio-mic.wav      # deleted after transcription unless "keep audio" is on
    audio-system.wav
    transcript.md      # timestamped, Me/Them attributed
    notes.md           # the distilled output — the artifact that matters
```

Library view = directory scan. No database in v1. Markdown means the notes are
greppable, syncable, and readable in Obsidian or anything else forever.

## Privacy stance

Fully local. No telemetry, no accounts, no network calls except localhost Ollama and
the one-time whisper model download (Hugging Face). "Keep audio" is **off** by default —
raw recordings are the most sensitive artifact and the least useful after transcription.

## Milestones

- **M0 — Scaffold**: Tauri 2 tray app boots; Start/Stop toggles state (stub); library
  window renders an empty state gracefully. *Done when `npm run tauri dev` shows a
  working tray menu.*
- **M1 — Capture**: Swift sidecar records both streams to WAVs; permissions onboarding.
  *Done when recording a real call/video yields two intelligible WAVs.*
- **M2 — Transcribe**: whisper-rs integration, model auto-download, merged Me/Them
  `transcript.md`. *Done when the M1 recording produces a readable transcript.*
- **M3 — Distil**: Ollama client, distil prompt, `notes.md`; library shows meetings and
  renders notes. *Done when meeting → notes works end-to-end without a terminal.*
- **M4 — Polish**: settings (models, keep-audio), recording tray-icon state, done
  notification, error surfaces, first-run onboarding.

## Non-goals (v1)

Live/streaming transcription · true multi-speaker diarization (beyond Me/Them) ·
calendar integration / meeting auto-detection · search · in-app note editing ·
Windows/Linux · any cloud or sync.

## Open questions

- **Local LLM default.** Only truly-local model installed today is `qwen3.5:2b` —
  workable but weak for summarization. Recommend pulling a mid-size local model
  (e.g. an 8B-class instruct model); distil should detect what's available and
  suggest, never silently fall back to a `*-cloud` model.
- Whisper model size vs. speed default — tune on real recordings in M2.
- Auto-start recording when a meeting app produces audio — nice v2 candidate.
