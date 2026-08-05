# Live transcript panel

Date: 2026-08-05
Status: approved, not yet implemented

## Goal

Show a running transcript while a meeting is being recorded, so the user can
glance over and re-read something that was just said. Comparable to Granola's
live transcript.

The live transcript is a **preview**. It is disposable, never persisted, and
never feeds the notes. The authoritative transcript remains the existing
post-recording pass over the full WAV files.

## Decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Purpose | Read along mid-call | Drives the latency and accuracy targets below. |
| Live model | `tiny.en-q8_0` (43 MB) | Barely registers next to a video call. English-only is acceptable for a disposable preview; the final pass still runs `language=auto`. |
| Final pass | Unchanged | Notes quality must not regress. Chunked transcription is worse than a whole-file pass, so the live path is kept entirely separate from it. |
| Placement | Floating always-on-top window | Distil is a menu-bar app; during a call the user is in Zoom/Teams and won't bring the main window forward. |
| Revisions | Commit settled text, only the tail moves | Text already read must not rewrite itself. |
| Capture → transcription contract | Headerless raw PCM file | Avoids reading a WAV mid-write, and is capture-backend agnostic for future non-macOS ports. |

### Rejected alternatives

- **Tail the WAV files directly.** No sidecar change, but a WAV's data-chunk
  length is not finalised until close, so this depends on `AVAudioFile`'s
  undefined flush timing and on-disk layout.
- **`SFSpeechRecognizer` on the Swift side.** Genuinely streaming with
  partial/final semantics for free, but requires Speech Recognition permission.
  Distil just removed the Screen Recording prompt specifically to reduce
  permission burden; adding a third prompt for a preview feature is the wrong
  trade. It is also a different engine from whisper, so live text would not
  resemble the final transcript.
- **Run the user's selected large model live.** Would make the final transcript
  nearly instant on stop, but is heavy on CPU/GPU during a call and introduces
  chunk-boundary artefacts into the authoritative transcript.

## Architecture

`ConvertingWavWriter` already converts every source to 16 kHz mono int16 —
exactly whisper's input format. It gains a second sink writing those same frames
to a headerless PCM file. Because there is no header, there is no length field to
go stale, and a partial read is simply fewer samples.

```
Swift sidecar                    Rust host                     live window
─────────────                    ─────────                     ───────────
ConvertingWavWriter
  ├─▶ audio-mic.wav ─────────────────────────────▶ (post-hoc pass, unchanged)
  └─▶ live-mic.pcm  ──tail──▶ live.rs ──event──▶ panel
                                 │
ConvertingWavWriter              │
  ├─▶ audio-system.wav ──────────┼────────────────▶ (post-hoc pass, unchanged)
  └─▶ live-system.pcm ──tail─────┘
```

## Components

| Component | Responsibility |
| --- | --- |
| `ConvertingWavWriter` (Swift, existing) | Gains a raw PCM tap. No other behaviour change. |
| `live.rs` (Rust, new) | Owns the whisper context and rolling-window logic. Emits `live-transcript`. |
| `live` window (Tauri, new) | Always-on-top panel rendering committed + provisional lines. |
| `models.rs` (existing) | Extract `ensure_model(app, name)` from `ensure_whisper_model`, so the preview model can be fetched independently of the user's selected model. `download_model` already takes an explicit name. |

`tiny.en-q8_0` is already in the `WHISPER_MODELS` catalogue in `lib.rs`, so no
new catalogue entry is needed.

### `live.rs` internals

One `WhisperContext` is created per recording and reused for every tick. The
existing `transcribe()` loads the model afresh on each call, which is fine once
per meeting but would dominate the cost of a preview ticking every few seconds.

Per stream it keeps a `StreamTail`:

- `offset` — bytes consumed from the PCM file so far
- `window` — rolling buffer of roughly the last 20 s of samples
- `committed_ms` — watermark; nothing at or before this is ever re-emitted

## Data flow

Each tick, for each stream:

1. Read bytes past `offset`, convert int16 → f32, append to `window`.
2. Drop samples older than the window length.
3. If the window's RMS is below the silence threshold, skip this tick.
4. Run whisper over the window.
5. Convert segment timestamps to absolute by adding the window start.
6. Split at the commit horizon into committed and provisional.

Step 3 matters more than it appears: in a two-person call each stream is silent
roughly half the time, so it removes about half of all whisper invocations. It
reuses the same instinct as the existing `peakAmplitude` dead-stream check.

### Constants

| Constant | Value | Rationale |
| --- | --- | --- |
| Tick interval | 3 s | Fast enough to read along; slow enough to stay cheap. |
| Window length | 20 s | Must comfortably exceed the commit horizon so committed text had trailing context. |
| Commit horizon | 6 s before window end | Anything frozen had ~14 s of trailing context when whisper saw it, which is what prevents clipped-at-the-seam committed text. |

### Event shape

```ts
{ stream: "me" | "them",
  committed:   [{ startMs, endMs, text }],  // new since last emit — append
  provisional: [{ startMs, endMs, text }] } // the moving tail — replace wholesale
```

Committed lines are append-only and never re-sent, so the panel cannot rewrite
what the user has already read. Provisional is replaced wholesale each tick and
rendered dimmed.

### Attribution

Mic and system are separate files, so "Me" vs "Them" is simply which tail a
segment came from. No diarization runs on the live path. Diarization stays on
the post-hoc pass, where it splits Them into Them 1 / Them 2.

The panel merges the two streams by `startMs` for display.

### On stop

The final provisional flushes to committed. The panel footer switches to the
existing `Transcribing` / `Distilling` phases. The `.pcm` files are deleted
unless the `keepAudio` setting is on.

## Error handling

The invariant: **nothing on the live path can harm a recording.** The recording
is the artefact that cannot be recreated; a preview is not worth risking it.

| Failure | Behaviour |
| --- | --- |
| Preview model not downloaded | Panel shows "preparing…", `models.rs` fetches in the background, recording starts immediately regardless. |
| Whisper error on a tick | Log and skip the tick. Three consecutive failures stop the live loop; the panel reports unavailable. |
| Tick overruns the interval | Single-flight — skip rather than queue, so slow machines degrade to fewer updates instead of a growing backlog. |
| PCM read fails or file absent | Treated as "no new samples". |
| User closes the panel | Live loop stops entirely and reclaims CPU. Recording continues untouched. |

## Testing

The real logic is the windowing and commit split, and it is pure:
`(segments, now, committed_ms) → (newly_committed, provisional)`.

- **Unit tests, no audio or model:** a segment straddling the commit horizon; an
  empty window; the watermark never moving backwards; committed ranges never
  re-emitted.
- **Unit test for the PCM tail reader:** write to a temp file incrementally,
  assert offset tracking and int16 → f32 conversion.
- **`live_e2e.rs`, `#[ignore]`d**, following the existing `transcribe_e2e.rs` and
  `diarize_e2e.rs` pattern: synthesise speech with macOS `say`, write it into a
  PCM file in chunks to simulate live capture, assert text appears and that
  `committed_ms` advances monotonically without rewriting.

## Out of scope

- Persisting or exporting the live transcript.
- Searching, copying or flagging moments during the call.
- Live diarization of multiple remote speakers.
- Any change to the post-recording transcription, diarization or distillation.

## Known risks

- **English-only preview.** A non-English meeting produces a nonsense live
  panel, while the final notes remain correct. Accepted; revisit by switching
  the preview model to `small` if it becomes a real annoyance.
- **Tick cost on lower-end hardware.** Designed around an Apple Silicon machine.
  The single-flight rule and silence skipping are the mitigations; if they prove
  insufficient the tick interval is the tuning knob.
