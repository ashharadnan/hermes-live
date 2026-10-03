# Local Live Voice Assistant for Hermes Agent

> **THIS PROJECT IS NOT HERMES READY YET.** Do not wire it into Hermes process
> management as-is — it runs standalone today; Hermes/PM integration is future
> work (see Roadmap).

A fully local, half-duplex-but-interruptible voice assistant: Moonshine STT ->
any OpenAI-compatible LLM -> any OpenAI-compatible TTS, wired over HTTP so
nothing shares GPU code. Built on pipecat 1.12.0.

**Status: working.** Verified live runs with clean metrics, correct voice, and
correct playback speed. **Barge-in verified** (user speech mid-reply cancels
in-flight LLM/TTS in milliseconds, repeatable) and **multi-turn context
verified** (the bot recalls prior turns, including ones interrupted mid-reply).
**Mid-reply starvation gaps fixed** via sentence-batched TTS requests (see
Status of the lane experiment in Roadmap).

## Architecture

Three independent pieces; the bot is a pure HTTP client to the servers (zero
coupling to their code). Swap any server for another OpenAI-compatible one by
editing `config.yaml` — no code changes.

| Piece | Runs | Configured via |
|---|---|---|
| LLM server (llama.cpp used here) | anywhere reachable over HTTP | `llm:` section |
| TTS server (Chatterbox Turbo used here) | anywhere reachable over HTTP | `tts:` section |
| pipecat bot (this repo) | host with mic access, CPU only | `audio:` / `vad:` sections |

How the pipeline flows (`bot.py`): Silero VAD on the mic segments user speech ->
Moonshine transcribes each segment on CPU (ONNX, no GPU) -> the LLM server
streams the reply with thinking forced off -> the TTS server returns PCM audio,
resampled to the configured output rate. User speech during bot playback raises
an InterruptionFrame that cancels in-flight LLM/TTS output.

## Quick start

1. Start your LLM server (its OpenAI-compatible endpoint goes in
   `config.yaml` under `llm.base_url`).
2. Start your TTS server (endpoint under `tts.base_url`). Note the rate it
   emits on `/v1/audio/speech` — it becomes `audio.out_sample_rate`.
3. Environment — two options:
   - **No setup needed when running as a Hermes process manager (PM) job** — the
     PM venv reads `pyproject.toml` and handles dependencies automatically.
   - **Standalone run:** create a project venv if you want this isolated from
     Hermes —
     ```bash
     cp config_template.yaml config.yaml   # then edit endpoints/models/voice
     uv sync   # or: pip install .
     python bot.py
     ```
   The mic requires a host-native python — e.g. from WSL, run the Windows venv's
   `python.exe`; a WSL-side python cannot open the audio device. The bot itself
   is CPU-only (STT included).

## Configuration

**`config.yaml` is the single source of truth.** No parameters live in code.
`config.py` loads it into read-only namespaces (`CFG.llm.*`, `CFG.tts.*`,
`CFG.audio.*`, `CFG.stt.*`, `CFG.vad.*`) — write attempts raise. Edit the yaml,
restart the bot, that's it.

What you'll typically point at your setup: LLM endpoint/model/`enable_thinking`
(the flag the LLM honours — see gotchas) /system prompt; TTS endpoint/model/
voice reference/timeout/**aggregation mode** (`tts.aggregation_mode`: sentence
|token|batch, with `tts.batch_lead`/`batch_carry` for batch); audio in/out
sample rates; STT model cache location; VAD thresholds. A committed
`config_template.yaml` documents every key with its default.

## Measured performance (live, co-resident GPU servers, warm, low contention)

Reference setup those numbers came from:

- **CPU:** AMD Ryzen 5 5600X (6C/12T)
- **GPU:** AMD Radeon RX 6800 XT 16 GB (ROCm on the Windows host) — both servers
  co-resident (llama.cpp ~6.3 GB + Chatterbox ~5.9 GB VRAM of 16.4 GB total)
- **RAM:** 32 GB
- **LLM:** llama.cpp ROCm build (llama-server) serving gemma-4-12B IQ3_XXS
  GGUF (~4.6 GB on disk)
- **TTS:** Chatterbox-TTS-amd (Turbo model, ROCm build)
- **STT:** Moonshine small-streaming ONNX on CPU (bot process)
- **Pipeline:** `/v1/audio/speech` lane at `audio.out_sample_rate` from
  config.yaml (both the non-streaming 44.1 kHz lane and the server-streaming
  24 kHz lane were measured — no first-byte gain from streaming, slower
  big-text completion; the live config sits on the streaming 24 kHz lane);
  TTS requests are **sentence-batched** (`batch(2,2)`: two sentences per
  request); metrics are pipecat's per-leg TTFB/TTFA

| Metric | Value |
|---|---|
| Moonshine STT TTFB | 0.41–1.07 s |
| LLM TTFB | 0.05–0.77 s (warm server cache reaches ~0.1 s) |
| TTS first byte (per batch) | 1.26–3.82 s |
| TTFA (first audible sample) | 1.22–3.90 s |
| First spoken word after user stops | ~2.8–3.9 s |

Under GPU contention the TTS TTFA alone blew up to 21 s — keep measurement runs
clean. First-run note: Moonshine's model downloads once on first use (into
`stt.cache_dir`); the TTS engine may need a one-time long warmup (10–26 s) on
its first synthesis after boot.

## Testing

Requires the `test` extra: `uv pip install -e ".[test]"` (or
`pip install -e ".[test]"`).

```bash
.venv/Scripts/python.exe -m pytest tests/ -q     # or .venv/bin/python
```

All tests run offline — no LLM/TTS servers or mic needed. Prereqs: config.yaml
copied from the template (Quick start) and Moonshine weights in `model_cache/`
(warm; a cold cache downloads them on first run, ~8 s). The STT tests
transcribe `tests/data/hello.wav` (16 kHz mono reference clip).

## Known gotchas

- **Audio out rate must match the server.** `audio.out_sample_rate` MUST equal
  the rate the TTS server actually emits on `/v1/audio/speech` (check its
  config — some servers resample non-streaming output to a configured rate,
  while streaming branches often emit engine-native PCM). The PCM is
  headerless; a mismatch plays slow and pitch-dropped.
- **Thinking must stay off for models that emit thinking blocks** — otherwise
  multiple seconds of silence before speech. Flags ride every request via
  `extra_body` in `build_chat_completion_params`
  (`services/llamacpp_llm.py`); models without thinking blocks simply ignore
  unknown chat-template kwargs.
- **`tts.stop_frame_timeout_s: 15`** covers TTS first-byte latency under
  GPU load; pipecat's 3.0 s default aborts a TTS context before audio arrives.
- **Custom TTS voices need registration.** pipecat 1.12 validates the `voice`
  client-side against OpenAI's built-in `VALID_VOICES` table; local
  reference-clip voices are registered into that table at init
  (`services/chatterbox_tts.py`).
- **Streaming-WAV caveat (Chatterbox fork).** Raw streaming wav
  (`stream=true` + `response_format=wav`) ships header sizes of `0xFFFFFFFF` —
  broken duration if written straight to file. Use pcm for streaming lanes;
  if you must save streamed wav, rewrite a correct 44-byte header from the
  collected payload first.
- **Engine lead-in silence.** The engine renders ~50–105 ms of sub-threshold
  audio at the head of every synthesis (both lanes) — the first syllable
  starts faint. Perceptible only on some output devices (Bluetooth, notably);
  on wired output it is a non-issue, so no trimming is applied.

## Layout

- `bot.py` — pipeline assembly + runner
- `pyproject.toml` — dependency source of truth (PM venv installs from it;
  standalone users run `uv sync` / `pip install .`)
- `config.py` — read-only yaml loader (zero parameters here)
- `config.yaml` — every endpoint, model/voice name, sample rate, VAD knob
  (gitignored; copied from the template)
- `config_template.yaml` — committed template with defaults + key comments
- `services/llamacpp_llm.py` — generic OpenAI-compatible LLM client, thinking
  flag forced off on every request
- `services/chatterbox_tts.py` — generic OpenAI-shaped TTS client, PCM
  streaming, custom voice registered client-side, aggregation-mode selection
  (sentence/token/batch)
- `services/sentence_batch.py` — sentence-batching TTS aggregator (release one
  group of `lead`/`carry` sentences per request — kills mid-reply starvation
  gaps; 9 dedicated tests)
- `tests/` — offline test suite: config loader gate, LLM/TTS/STT services, VAD
  state machine, pipeline assembly invariants, frame flow, bot contracts,
  sentence-batch rhythm/loss/lifecycle

## Roadmap

1. **Moonshine accuracy** — check transcription quality; tune VAD thresholds or
   swap variant if weak.
2. Later: remote transports (Discord/WebRTC), spoken-prose rewrite layer, wake
   word, duplex turn-taking.

Lane experiment (settled 2026-10-03, for the record): server-side streaming-ON
(24 kHz native) + sentence `batch(2,2)` = shipped config — sentence-level
starvation gaps eliminated at TTFA 1.5–2.5s. Token aggregation measured
architecturally wrong for request-per-call engines (one request per word, each
paying engine startup); uniform pairs beat both lead=1/carry=2 (slow TTFA +
first-sentence gap) and lead=2/carry=1 (one early gap) in a live A/B.