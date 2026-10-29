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
| Capture | **Swift sidecar** (`distil-capture`) | Bundled as a Tauri sidecar. ScreenCaptureKit audio-only stream for system audio; AVAudioEngine for raw mic. (Apple voice processing/AEC is deliberately NOT used — in a headless sidecar it produced an all-zero mic stream; bleed is handled at merge time. Headphones give perfect separation regardless.) Emits 16 kHz mono WAV per stream + peak amplitude per stream so silent captures are detected, not transcribed. Needs mic + screen/system-audio recording permissions (TCC). |
| Transcription | **whisper.cpp** via whisper-rs (Metal) | Default model: quantized `large-v3-turbo` (~574 MB), auto-downloaded on first run into app data dir. Post-meeting transcription in v1, not live. Merge-time bleed filter drops mic segments whose text duplicates time-overlapping system segments (backstop for speakers-without-headphones), and whisper noise annotations (`*music*`, `[Applause]`) are stripped. Silence is skipped first: a Silero VAD (via sherpa-onnx, 0.6 MB) finds the speech, whisper sees only that, and segment times are mapped back onto the recording so diarization still lines up. NVIDIA **Parakeet TDT 0.6B v3** (also via sherpa-onnx, CPU) is the faster alternative. One model is loaded per meeting and serves both streams. |
| Distillation | **Ollama** over localhost HTTP | Local models **only** — `*-cloud` Ollama models are excluded; they'd silently break the privacy promise. Runtime picks the best locally-installed model; configurable. Structured prompt → TL;DR, key points, decisions, action items (owner + due date when stated), open questions. |
| Orchestration | Rust state machine | `idle → recording → transcribing → distilling → done/error`; progress pushed to UI via Tauri events. |

Pipeline: tray **Start** → sidecar records two WAVs → **Stop** → whisper transcribes both
→ merge by timestamps (`Me`/`Them`) → Ollama distils → write meeting folder → notify.

## Model selection

Two models, two pickers. Selection lives in `settings.json` in the app data dir from
the moment each pipeline stage exists; the Settings UI is sugar over that file.

- **Transcription (whisper.cpp)** — curated list: `large-v3-turbo` q5 (default) ·
  `small` (speed) · `large-v3` (max accuracy). Picking one downloads it on demand into
  the app data dir; the picker shows what's already on disk.
- **Transcription speed (Oct 2026)**: transcription was the slowest stage, because
  whisper ran over both streams in full, silence included, and reloaded its model
  for each. Now the model loads once per meeting and a **Skip silence** switch (on by
  default) hands whisper only the speech a VAD found. The picker also offers
  `parakeet-tdt-0.6b-v3` (NVIDIA, CC-BY-4.0, about 670 MB int8, 25 European languages
  only) beside the whisper models. Parakeet takes short inputs, so it always works
  from VAD spans. If the VAD model can't be fetched whisper runs over the whole
  stream, and if Parakeet can't load whisper takes over, so a meeting is never
  lost to a download. meta.json records `transcriber` and `skippedSilence`. The
  Parakeet files come from Hugging Face; the 0.6 MB VAD model comes from the
  sherpa-onnx release on GitHub.
  `eval_asr` in `docs/model-eval.md` compares engines on real meetings for speed and
  word error rate before the default changes.
- **Distillation (Ollama)** — populated live from `ollama list`, with `*-cloud` tags
  filtered out. Default = largest local model that fits a RAM heuristic. New options
  appear by running `ollama pull <model>`; the picker just refreshes.
- **Distillation engine (Oct 2026)**: Settings can switch from Ollama to any
  OpenAI-compatible server (oMLX, `mlx_lm.server`, `llama-server`, LM Studio), so
  an MLX runtime can be swapped in. Ollama keeps its native API because only it
  gives a reliable thinking toggle, a settable context window and model sizes for
  auto-pick. A server that isn't on loopback counts as cloud and sits behind the
  same opt-in. Per-model presets (`llm.rs`) apply the sampling and system prompt
  each model card recommends; `docs/model-eval.md` compares models on real
  meetings before one becomes the default.

Rollout: `settings.json` overrides work from M2 (whisper) / M3 (Ollama); the
distillation dropdown ships with M3's library window; the full Settings pane (both
pickers, persistence, download progress) is M4.

Hardware reality check (this machine: M1, 8 GB RAM): whisper `large-v3-turbo` q5
(~0.6 GB) is fine post-meeting; the LLM ceiling is ~4B-class — `qwen3.5:4b` (3.4 GB,
256K context) is the quality pick, installed `qwen3.5:2b` works today. 8B+ models
swap on 8 GB; `gemma4:e4b` (9.6 GB download) does not fit.

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

Local by default, and local means local: no telemetry, no accounts, no network calls
except localhost Ollama and the one-time whisper model download (Hugging Face).
"Keep audio" is **off** by default — raw recordings are the most sensitive artifact
and the least useful after transcription.

One deliberate exception, added at Phil's request (Aug 2026): **cloud Ollama models
as explicit opt-in**. Off by default; enabling the toggle in Settings surfaces
`*-cloud` models badged CLOUD in the picker with consent copy stating that
transcripts sent to them leave the Mac (Ollama's stated policy: transient
processing, zero retention, no training). Cloud models are **never auto-picked** —
auto mode always resolves to the largest local model. Choosing cloud is always a
per-model, user-made decision.

## Milestones

- **M0 — Scaffold**: Tauri 2 tray app boots; Start/Stop toggles state (stub); library
  window renders an empty state gracefully. *Done when `npm run tauri dev` shows a
  working tray menu.*
- **M1 — Capture**: Swift sidecar records both streams to WAVs; permissions onboarding.
  *Done when recording a real call/video yields two intelligible WAVs.*
- **M2 — Transcribe**: whisper-rs integration, model auto-download, merged Me/Them
  `transcript.md`. *Done when the M1 recording produces a readable transcript.*
- **M3 — Distil**: Ollama client, distil prompt, `notes.md`; library shows meetings and
  renders notes, plus a simple distillation-model dropdown (reads `ollama list`).
  *Done when meeting → notes works end-to-end without a terminal.*
- **M4 — Polish**: full Settings pane (both model pickers, keep-audio), recording
  tray-icon state, done notification, error surfaces, first-run onboarding.
- **M5 — Calendar-aware prompts (v1.1)**: EventKit via the Swift helper. Reads the
  macOS Calendar database — whatever accounts are already synced there (Google, iCloud,
  Outlook) — so there is **no OAuth and nothing leaves the machine**; costs one
  Calendar (Full Access) permission. Poll upcoming events; a "real meeting" has other
  attendees or a video-conference link. ~2 min before start: notification + tray item
  "Record: <meeting title>" — one click starts recording with the folder pre-named.
  Meeting title + attendee names go into `meta.json` and into the distil prompt, which
  sharpens action-item owner attribution.
- **M6 — Multi-speaker fidelity (v1.2)**: split "Them" into individual speakers.
  The system stream arrives pre-mixed, so this is diarization: sherpa-onnx
  (segmentation + speaker-embedding models, ~50–100 MB, auto-downloaded, CPU-friendly)
  clusters the system stream into `Them 1..N` time ranges, intersected with whisper
  segments at merge time. A distil-time LLM pass then maps numbered speakers to real
  names — works with or without a calendar attendee list. Evidence hierarchy:
  self-introductions ("it's James here") > repeated address-then-response patterns
  ("Sarah, thoughts?" → Them 3 replies) > anything else; third-person mentions are
  the trap (names of people not in the call) and are discounted, hard when an
  attendee list exists. Conservative assignment: multiple consistent evidence points
  required, one name never maps to two clusters, unsure → keep the number ("Them 2"
  accurate beats "Sarah" wrong). Transcript carries an auditable attribution header
  ("Speakers: Sarah (Them 1, inferred), Them 2 (unidentified)") so inferences are
  visible and trivially hand-correctable in the Markdown. `Me` remains ground truth
  from the mic stream. Settings toggle (adds processing time); honest limit:
  overlapping speech degrades everyone's diarization, including ours. Explicit
  non-approach: joining meetings as a bot for per-participant streams —
  cloud-service territory, rejected.
- **M7: Meeting detection (v1.3)**: notice a call without needing a calendar entry.
  A second sidecar mode, `distil-capture watch-mic`, polls Core Audio's process
  objects every 2 s and reports which processes are capturing input (bundle id +
  pid). It reads metadata about other processes, not audio, so it should need no
  permission of its own. The host feeds each report to a pure, unit-tested
  `Detector` (`meetings.rs`). A known call app (Teams, Zoom, Webex, Slack,
  FaceTime, Discord) seen in two consecutive reports counts as active; when one
  becomes active and nothing is recording, a notification and a tray row offer
  to record it (named after the calendar meeting if one is on, else "Teams call").
  Browsers also open the mic for things that aren't calls, so they only count
  while a calendar meeting is inside its prompt window. During a recording the
  detector remembers every call app it sees; once all of them have been off the
  mic for 15 s it asks whether to stop, or stops by itself if "Stop when the call
  ends" is on (off by default, so a recording never ends unasked). Prompt, not
  auto-start: Distil never begins recording without a click. Both switches live
  in Settings.
  Teams in the calendar: EventKit's location, notes and URL now go to Rust raw, and
  `calendar.rs` finds the video link (previously done in Swift, untestable). It
  recognises `teams.microsoft.com` and `teams.live.com` links, unwraps Outlook
  Safe Links (`*.safelinks.protection.outlook.com/?url=...`) to the link inside,
  and treats a location of "Microsoft Teams Meeting" as a meeting even when
  Outlook left the link only in the body.

## Non-goals (v1)

Live/streaming transcription · true multi-speaker diarization (beyond Me/Them) ·
search · in-app note editing · Windows/Linux · any cloud or sync.
(Calendar prompts graduated from this list to M5.)

## Open questions

- Whisper model size vs. speed default — tune on real recordings in M2.
- Auto-start recording when a meeting app produces audio (no calendar event needed) —
  addressed by M7 as a prompt rather than an auto-start; revisit only if the prompt
  proves too easy to miss.
