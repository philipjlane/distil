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
