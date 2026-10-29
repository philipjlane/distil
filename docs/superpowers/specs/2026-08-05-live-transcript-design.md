# Live transcript panel

Date: 2026-08-05
Status: implemented

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

## Implementation notes

Built as designed; the differences and additions are below. The code is
`src-tauri/src/live.rs` (the loop and all the logic), `models::ensure_model`,
the `--live-pcm` option of the sidecar, `src/LivePanel.tsx` and a few lines in
`lib.rs`.

### Differences from the design

- **Whisper hears only the audio after the watermark.** The design runs it over
  the whole 20 s window on every tick. Simulated with real speech, that had
  whisper cut the already committed audio differently each time, which gave
  repeated lines and, worse, lost words: a re-cut line that mostly overlapped
  committed time was dropped as a repeat although its last words were new
  ("...ask what you can do for your country" lost "for your country"). So each
  pass decodes only what follows the last committed text (still never more than
  20 s). Text that commits still had at least the 6 s horizon of audio after it,
  the next text starts at a natural boundary, and a pass costs less. The window
  stays 20 s as the most that is kept.
- **Silence check on the uncommitted audio.** Step 3 measures the RMS of the
  part of the window after the commit watermark, not of the whole 20 s window.
  Otherwise a stream that has just gone quiet stays "loud" for up to 20 s while
  its last words slide out of the window, and the saving the design counts on
  never arrives. When that part is silent, whatever provisional text is on show is
  cleared, because nothing backs it any more. The threshold is an RMS of 0.003
  (about -50 dBFS).
- **The sidecar only writes the PCM files when asked.** `distil-capture
  --out-dir <dir> --live-pcm`; the host passes the flag only when the live
  transcript is on, so switching it off costs no extra disk writes.
- **PCM trouble is a `warning` event, not an `error`.** The host treats any
  `error` event from the sidecar as a failed recording and deletes the meeting
  folder, so a preview problem must not use it. A failure to create or write the
  PCM file is reported once as `{"event": "warning", ...}` and the copy switches
  itself off; the WAV is never touched. Writes use the throwing
  `FileHandle.write(contentsOf:)`, since the older `write(_:)` raises an
  Objective-C exception on a full disk.
- **The panel is a new window per recording, closed for real.** The design said
  closing hides the window. A hidden window shown again takes focus (tao calls
  `makeKeyAndOrderFront`), which would pull focus out of the call at the start
  of the next recording, so the live window is built unfocused each time and
  destroyed when closed. Closing it, or switching the setting off, stops the
  loop. After the notes are written the panel closes itself a few seconds later.
- **Two extra signals for the panel.** A `live-status` event (`preparing`,
  `live`, `unavailable`, `finished`) says what the preview is doing, which the
  design's "preparing" and "unavailable" messages need, and a `live_snapshot`
  command returns everything said so far, because the webview loads a moment
  after the loop starts and would otherwise miss early lines.
- **The last pass decodes again.** On stop the loop reads the final audio, runs
  whisper once more and commits everything, so the last couple of seconds are in
  it. If that decode fails, the provisional text last shown is committed instead.
  The host waits up to 10 s for this before transcription starts, then moves on.
- **The `.pcm` files go as soon as the loop is done.** The host deletes them
  (unless "keep audio" is on) right after the last pass and before the full
  transcription starts, so they are gone whether or not transcription later
  succeeds.

### Choices the design left open

- A line whose middle lies at or before the watermark is a repeat (whisper cuts
  the same audio differently from tick to tick) and is dropped; this is what
  keeps "committed ranges are never re-emitted" true across ticks. A segment
  straddling the commit horizon stays provisional whole, and committing is
  always a prefix, so committed text is never followed by older text. Segment
  times are clamped to the window, since whisper sometimes places a segment past
  the end of its audio.
- Lines over silent audio are dropped before the split: whisper invents "Thank
  you." over silence, and a committed line is permanent.
- One whisper context and one state serve both streams, one tick after another.
  Each decode uses `no_context` and no temperature fallback, so ticks are
  independent and a hard stretch never costs extra passes.
- The loop waits at least a second between passes even when one overruns, so a
  slow machine does less work rather than running flat out.
- A tick fails if either stream's decode fails; three failed ticks in a row stop
  the loop and the panel says the preview is unavailable.
- Settings has a "Show a live transcript while recording" switch (on by
  default).

### Known limits

- A single whisper segment longer than about 14 s never settles while the
  speaker keeps going, and the front of it can slide out of the window before it
  does. In practice tiny.en breaks continuous speech into shorter segments.
- The preview does not filter mic bleed. Without headphones, what the speakers
  play reaches the mic and can show up under Me as well as Them; the real pass
  drops it.
- The live thread runs at normal priority; its cost next to a real call has not
  been measured on a Mac.

### Tests

Unit tests in `live.rs` cover the commit split, the PCM tail reader, the
rolling window, a tick with a scripted decoder, and the loop (failure counting,
cancel, finish, a model that will not load, a decoder that panics). The
`#[ignore]`d `tests/live_e2e.rs` synthesises speech with `say`, feeds it into a
PCM file in chunks on a simulated clock and checks the real model's output:
text appears, committed text only grows, the watermark never moves back, and the
stop commits the last words.
