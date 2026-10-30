# distil

Local-first meeting notes for macOS. Distil lives in your menu bar, records your
meetings (mic + system audio as separate streams), transcribes them on-device with
whisper.cpp, and distils the transcript into structured notes — TL;DR, decisions,
action items — with a local LLM via Ollama. Notes are plain Markdown files you own,
in `~/Documents/Distil/`. Nothing leaves your Mac unless you explicitly opt in to
cloud Ollama models.

## Features

- **Two-stream capture** — your mic and everyone else's audio recorded separately,
  giving Me/Them speaker attribution for free (plus a bleed filter for
  speakers-without-headphones setups)
- **On-device transcription** — whisper.cpp with Metal, models downloaded on first use
- **Local distillation** — notes written by the largest local Ollama model, or one
  you pick; cloud models are strictly opt-in and clearly badged. Any
  OpenAI-compatible local server (oMLX, mlx-lm, llama.cpp, LM Studio) works too, and
  [docs/model-eval.md](docs/model-eval.md) compares candidate models on your own
  meetings
- **Calendar-aware** — reads your macOS Calendar locally (no OAuth) and offers to
  record when a real meeting is about to start, with the folder pre-named
- **A real library** — browse meetings, rendered notes/transcripts, copy, reveal in
  Finder, delete to Trash

## Build & install

Requirements: macOS 14.2+, Xcode command-line tools, Rust, Node, CMake,
[Ollama](https://ollama.com) with at least one local model (e.g.
`ollama pull qwen3.5:4b`).

```sh
npm install
npm run tauri build        # produces Distil.app + .dmg under src-tauri/target/release/bundle/
```

Copy `Distil.app` to `/Applications` and launch — it appears in the menu bar (no
Dock icon). macOS will ask for Microphone and Calendar up front, and for audio
recording the first time you record — Distil captures system audio with a Core
Audio process tap, so it never asks for Screen Recording. "Start Distil at
login" lives in Settings.

For development: `npm run tauri dev`.

## Architecture

See [PLAN.md](PLAN.md) — Tauri 2 shell, Swift sidecar for capture/EventKit,
whisper-rs, Ollama over localhost, plain-Markdown storage. The plan documents the
full pipeline, milestones (M0–M5, all shipped), and the privacy stance.
