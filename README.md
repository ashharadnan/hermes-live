# Local Live Voice Assistant for Hermes Agent

A fully local, half-duplex-but-interruptible voice assistant: Moonshine STT ->
any OpenAI-compatible LLM -> any OpenAI-compatible TTS, wired over HTTP so
nothing shares GPU code. Built on pipecat 1.12.0.

**Status: working.** Verified multi-turn live runs with clean metrics, correct
voice, and correct playback speed. Remaining: a barge-in test, multi-turn memory
sanity, and an optional faster TTS lane (see Roadmap).

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
3. Set up the environment and run:
   ```bash
   pip install -r requirements.txt   # into the project venv
   cp config_template.yaml config.yaml   # then edit endpoints/models/voice
   python bot.py
   ```
   The mic requires a host-native python — e.g. from WSL, run the Windows venv's
   `python.exe`; a WSL-side python cannot open the audio device. The bot itself
   is CPU-only (STT included).

## Configuration

**`config.yaml` is the single source of truth.** No parameters live in code.
`config.py` loads it into read-only namespaces (`CFG.llm.*`, `CFG.tts.*`,
`CFG.audio.*`, `CFG.vad.*`) — write attempts raise. Edit the yaml, restart the
bot, that's it.

What you'll typically point at your setup: LLM endpoint/model/`enable_thinking`
(the flag the LLM honors — see gotchas) /system prompt; TTS model/voice
reference clip/timeout; audio in/out sample rates; VAD thresholds. A committed
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
- **Pipeline:** non-streaming `/v1/audio/speech` lane, 44.1 kHz PCM; metrics are
  pipecat's per-leg TTFB/TTFA

| Metric | Value |
|---|---|
| Moonshine STT TTFB | 0.44–0.85 s |
| LLM TTFB | 0.26–0.30 s |
| TTS first byte | 1.78–2.23 s |
| TTFA (first audible sample) | 1.88–2.42 s |
| First spoken word after user stops | ~2.8–3.4 s |

Under GPU contention the TTS TTFA alone blew up to 21 s — keep measurement runs
clean. First-run note: Moonshine's model downloads once on first use; the TTS
engine may need a one-time long warmup on its first synthesis.

## Known gotchas

- **Audio out rate must match the server.** `audio.out_sample_rate` MUST equal
  the rate the TTS server actually emits on `/v1/audio/speech` (check its
  config — some servers resample non-streaming output). The PCM is headerless;
  a mismatch plays slow and pitch-dropped.
- **Thinking must stay off for models that emit thinking blocks** — otherwise tens of seconds of silence before speech. Flags ride every
  request via `extra_body` in `build_chat_completion_params`
  (`services/llamacpp_llm.py`); models without thinking blocks simply ignore
  unknown chat-template kwargs.
- **`tts.stop_frame_timeout_s: 15`** covers TTS first-byte latency under
  GPU load; pipecat's 3.0 s default aborts a TTS context before audio arrives.
- **Custom TTS voices need registration.** pipecat 1.12 validates the `voice`
  client-side against OpenAI's built-in `VALID_VOICES` table; local
  reference-clip voices are registered into that table at init
  (`services/chatterbox_tts.py`).

## Layout

- `bot.py` — pipeline assembly + runner
- `config.py` — read-only yaml loader (zero parameters here)
- `config.yaml` — every endpoint, model/voice name, sample rate, VAD knob
  (gitignored; copied from the template)
- `config_template.yaml` — committed template with defaults + key comments
- `services/llamacpp_llm.py` — generic OpenAI-compatible LLM client, thinking
  flag forced off on every request
- `services/chatterbox_tts.py` — generic OpenAI-shaped TTS client, PCM
  streaming, custom voice registered client-side

## Roadmap

1. **Barge-in test** — speak over the bot mid-reply; verify InterruptionFrame
   cancels in-flight TTS cleanly.
2. **Multi-turn sanity** — confirm the bot remembers earlier turns within a
   session.
3. **Optional faster TTS lane** — enable server-side streaming for a faster
   first byte; on the Chatterbox fork that means `openai_stream_by_default:
   true` + restart, with pipecat's output rate adjusted to the streaming
   branch's rate (engine-native 24 kHz there).
4. **Moonshine accuracy** — check transcription quality; tune VAD thresholds or
   swap variant if weak.
5. Later: remote transports (Discord/WebRTC), spoken-prose rewrite layer, wake
   word, duplex turn-taking.