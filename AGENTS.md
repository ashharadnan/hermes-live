# AGENTS.md — guidance for coding agents in this repo

Standalone pipecat voice pipeline: Moonshine STT -> OpenAI-compatible LLM ->
OpenAI-compatible TTS. The bot is a pure HTTP client; the servers are
independent and live outside this repo. All endpoints/models/voices come from
config.yaml - swap servers without touching code.

## Project type

Python 3.12, pipecat 1.12.0. Runs in the project venv (`.venv/`) with a
host-native python — when developing from WSL against a Windows venv, invoke
`.venv/Scripts/python.exe` (a WSL-side python cannot open the mic).

## Architecture invariants (do not break)

- Pipeline order: `transport.input() -> stt -> context_aggregator.user() -> llm -> tts -> transport.output() -> context_aggregator.assistant()`
- VAD rides the USER aggregator (`LLMUserAggregatorParams(vad_analyzer=...)`), not
  TransportParams — 1.12 wires turn detection + interruption there
- Thinking-off flags (e.g. `chat_template_kwargs`) must ride every LLM request as
  `extra_body`; the SDK rejects unknown top-level kwargs
- `audio.out_sample_rate` must equal the rate the TTS server actually emits on
  `/v1/audio/speech` (check the server's config); mismatch = pitch-dropped,
  slowed audio on headerless PCM
- Custom TTS voices must be registered into pipecat's client-side
  `VALID_VOICES` table (see `services/chatterbox_tts.py`)
- Services in `services/` are generic OpenAI-compatible clients, not
  model/vendor-specific: class names + log lines must stay implementation-neutral

## Config rule (hard requirement)

ALL tunables live in `config.yaml` — nothing hardcoded in source files. Source
code references `CFG.*` only (`from config import CFG`); `config.py` is a pure
read-only loader. `config.yaml` is gitignored; `config_template.yaml` ships and
gets copied to `config.yaml` on first run. After editing config.yaml, restart
the bot.

## Commands

```bash
cd <project-root>
[ -f config.yaml ] || cp config_template.yaml config.yaml

# run (servers must be up first; see preflight below)
.venv/Scripts/python.exe bot.py        # Windows venv from WSL
# or: .venv/bin/python bot.py          # POSIX venv

# lint (round-scoped: only files touched this round; ruff must be available)
ruff check --select E,F,W,I,N,UP,B,SIM,PLW config.py bot.py services/*.py

# compile gate
.venv/Scripts/python.exe -m py_compile config.py bot.py services/*.py
```

Server preflight (a listening port is not a working service):
- LLM server — `GET <base_url>/models` (connection refused = not started)
- TTS server — `POST <base_url>/audio/speech` with a short input and
  `response_format=pcm`; note some OpenAI-shaped servers 404 `/v1/models` even
  when healthy

The user owns the server processes (bat files / services of their choice);
never start or kill them unasked. A TTS engine may need a one-time long warmup
(10-26 s) on its first synthesis after boot — never report it as pipeline
latency.

## Code style

Google docstrings; ASCII-only; comments only where strictly necessary (single
line); meaningful names; no `print()` debug leftovers (loguru only); zero magic
numbers (they belong in config.yaml). Ruff-clean required.

## Verification workflow (before declaring done)

1. Delete any stray probe/scratch files created during the round
2. ruff (round-scoped) + py_compile (above)
3. Import the bot as a module in the project venv
   (`importlib.util.spec_from_file_location("bot", "bot.py")`) to catch wiring
   regressions without starting the pipeline
4. Live restart + read service startup log lines (endpoints, voice, VAD params)
   to confirm values load from config.yaml

## Known gotchas

- pipecat docs lag the installed package: verify class/module names against the
  installed tree before writing code (aggregators live in
  `...aggregators.llm_response_universal`; services have no
  `.create_context_aggregator()`; `PipelineParams` has no `allow_interruptions`)
- `TTSService.stop_frame_timeout_s` (default 3.0 s) aborts TTS contexts when
  first byte arrives late under GPU load; our value 15 s covers it. The
  "completed with no audio" error means timeout, not engine failure — triage
  with direct endpoint probes before touching the pipeline
- Never run engine probes while a live measurement is in flight; GPU contention
  inflates every TTS number (measured: TTFA 2 s warm -> 21 s under contention)
- Probe scripts must live on paths visible to the venv's python — a Windows
  venv python resolves WSL paths like `/home/...` against the drive root
- Moonshine downloads model weights on first use

## Milestone log

Current next steps: barge-in verification, multi-turn context sanity, optional
server-side streaming lane (faster first byte; adjust pipecat's output rate to
the streaming branch's rate), Moonshine accuracy tuning.