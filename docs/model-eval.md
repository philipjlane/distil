# Evaluating notes models

`eval_notes` runs candidate models over your own recent meetings with exactly
what the app uses: the same prompt, the same per-model presets, the same
transcript elision and think-stripping. It only reads your meetings. Results go
to `~/Documents/Distil Eval/<timestamp>/`, which the library never scans, and
nothing leaves your Mac unless you point it at a remote server.

## Run it

Ollama 0.40.12 or newer is needed for LFM3. Pull the candidates first (the
script stops and prints any missing `ollama pull` commands):

```bash
ollama pull qwen3.5:4b
```

```bash
ollama pull lfm3:1.7b
```

```bash
ollama pull hf.co/LiquidAI/LFM2-2.6B-Transcript-GGUF:Q4_K_M
```

Then, from `src-tauri/`:

```bash
cargo run --release --example eval_notes
```

That compares those three on your five newest meetings. Quit other heavy apps
first, since on an 8 GB Mac they skew the timings.

Useful variations:

```bash
cargo run --release --example eval_notes -- --limit 10 --models lfm3:1.7b,lfm3:1.7b+think
```

`+think` lets a model reason before answering. The app always asks for no
thinking, so this shows what that costs and whether it helps.

```bash
cargo run --release --example eval_notes -- --engine openai --url http://localhost:8080/v1 --models mlx-community/LFM2-2.6B-Transcript-4bit
```

`--engine openai` talks to any OpenAI-compatible server: oMLX, `mlx_lm.server`,
`llama-server` or LM Studio. All options are listed at the top of
`src-tauri/examples/eval_notes.rs`.

## Read the results

1. **`review.md`**: every candidate's notes per meeting, shuffled and labelled
   A, B, C. Score them before you know which is which.
2. **`key.md`**: which letter was which model.
3. **`report.md`**: speed and format checks per model, then per meeting.
   - *Prefill tok/s* is how fast the model reads the transcript, which is
     most of the wait for notes. *Decode tok/s* is how fast it writes.
   - *Wall s* includes model load on each model's first meeting; *First load s*
     shows how much of that was loading.
   - *Missing TL;DR*, *Over 300 words* and *Hit token cap* flag outputs that
     ignored the prompt's format or ran away.

`raw/` holds each model's notes under its own name, for diffing. The results
folder contains copies of your meeting notes, so delete it when you're done.

## Presets

Models get the sampling settings and system prompt their model cards
recommend (`src-tauri/src/llm.rs`, `preset_for`). The evaluation and the app
share them.

| Model family | Temperature | Other |
|---|---|---|
| LFM3 | 0.1 | top_k 50, repeat penalty 1.1 |
| LFM2-2.6B-Transcript | 0.3 | Liquid's "expert meeting analyst" system prompt |
| Everything else | 0.3 | none |

# Evaluating transcription

`eval_asr` runs the app's own transcriber over your recordings: the same whisper
settings, the same silence skipping and the same Parakeet set-up. It only reads
your recordings. Results go to `~/Documents/Distil Eval/<timestamp> asr/`, which
the library never scans, and nothing leaves your Mac apart from the first-time
model downloads.

## Keep some audio

Distil deletes a meeting's recordings once it has transcribed them, so there is
nothing to test on unless **Keep audio after transcription** is on in Settings.
Turn it on, record a few real meetings (calls with a few voices are the hard case
that matters), and turn it off again if you like. The recordings are the most
sensitive thing Distil makes, so delete the ones you no longer need.

## Run it

From `src-tauri/`:

```bash
cargo run --release --example eval_asr
```

That compares three engines on your three newest meetings that still have audio.
Each meeting contributes its microphone and system streams as separate files.

- `whisper:large-v3-turbo-q5_0`: how the app used to transcribe.
- `whisper:large-v3-turbo-q5_0+vad`: the same model, with silence skipped first. This
  is the app's default now.
- `parakeet`: NVIDIA Parakeet TDT 0.6B v3.

Models the app hasn't downloaded yet are fetched into its models folder
(`~/Library/Application Support/com.philipjlane.distil/models`) the first time:
about 0.7 GB for Parakeet. Quit other heavy apps first, since on an 8 GB Mac they
skew the timings.

Useful variations:

```bash
cargo run --release --example eval_asr -- --limit 1 --engines whisper:small,whisper:small+vad,parakeet
```

`--engines` takes `whisper:<model>` (any name from the whisper.cpp repository on
Hugging Face, without `ggml-` and `.bin`), the same with `+vad` on the end, and
`parakeet`.

```bash
cargo run --release --example eval_asr -- --wav ~/Desktop/standup.wav,~/Desktop/call.wav
```

`--wav` uses recordings you name instead of meetings. They must be 16 kHz mono
(`ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav` converts). All options are listed at
the top of `src-tauri/examples/eval_asr.rs`.

## Read the results

`report.md` has a summary per engine, then a row per file.

- *RTF* (real-time factor) is transcribe time divided by audio time. 0.1 means an
  hour of audio takes six minutes; lower is faster. Transcribe time includes
  skipping silence where an engine does it.
- *Load s* is the time to load the model, once per engine. The app pays it once per
  meeting. The first file may also include one-off warm-up.
- *WER* is the word error rate: the share of the correct words that were
  substituted, dropped or inserted. It needs the correct text in a `<name>.txt`
  beside the WAV (`audio-mic.txt` in a meeting's folder). The quickest way to make
  one is to copy the best engine's file from `raw/`, remove the timestamps and
  correct it by hand. A few minutes of speech is plenty. Case and punctuation are
  ignored, but numbers written as digits and as words ("3" and "three") count as
  different words, so judge the WER of one engine against another rather than as an
  absolute.

`raw/<file>/<engine>.md` holds each engine's transcript with timestamps, for
reading side by side. A good check on the `+vad` rows is the quiet stretches of a
recording: without silence skipping, whisper can invent text there. The results
folder contains transcripts of your meetings, so delete it when you're done.

## Parakeet and languages

Parakeet TDT 0.6B v3 knows 25 European languages: Bulgarian, Croatian, Czech,
Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian,
Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian,
Slovak, Slovenian, Spanish, Swedish and Ukrainian. It picks the language itself.
For anything else it does not report an error, it just transcribes badly, so
keep whisper for meetings in other languages.

Parakeet is by NVIDIA and is licensed
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Distil downloads the
int8 ONNX conversion that the sherpa-onnx project publishes
([csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8)),
made from [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
The silence detector is the MIT licensed Silero VAD, in the copy that sherpa-onnx
ships.
