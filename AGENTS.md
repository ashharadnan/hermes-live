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

# lint — select set is pinned in pyproject [tool.ruff.lint], so a plain
# run is exactly the gate; round-scope only files touched where useful
ruff check .

# test suite (offline; see Testing below)
.venv/Scripts/python.exe -m pytest tests/ -q     # or .venv/bin/python

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

## Testing protocol

Offline suite in `tests/` — no LLM/TTS servers, no mic. Needs the `test`
extra (`uv pip install -e ".[test]"` or `pip install -e ".[test]"`), a copied
config.yaml, and `model_cache/` warm (Moonshine loads ~1 s warm; cold cache
downloads once).

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Coverage per module: `test_config` (loader gate: sections/keys/types, read-only
guard, error paths, template structure parity), `test_llm_service` (settings
pinning, thinking-off flag in extra_body), `test_tts_service` (VALID_VOICES
gate, pcm pass-through via stubbed HTTP), `test_stt_service` (real Moonshine
transcription ground truth vs `tests/data/hello.wav`), `test_vad` (state
machine with stubbed model), `test_pipeline_assembly` (AGENTS.md invariants:
order, VAD placement, shared context, rates from config), `test_frame_flow`
(user-turn aggregation, LLM chunk streaming), `test_bot_contract` (import
gate, cache redirect, neutral naming, ast-scan for hardcoded tunables).

Extending the suite — conventions every new test must follow:

- **Never hit the network.** Stub the OpenAI client at `service._client`
  (MagicMock/AsyncMock shaped like AsyncOpenAI); the voice-gate test must also
  assert zero request attempts on rejection paths.
- **Fresh-clone safety lives in conftest only.** `collect_ignore` switches all
  config-touching modules off when config.yaml is absent. Do NOT add per-test
  skip guards for missing config — module-level imports crash before any skip
  fires (learned the hard way: six dead guards).
- **Mock shapes match installed 1.12, verified against source:**
  `get_chat_completions` is a coroutine returning a stream (drive with
  `stream = await svc.get_chat_completions(ctx)`); chunk->LLMTextFrame
  conversion happens in `svc._process_context` (capture by stubbing
  `svc.push_frame`); TTS `run_tts` needs a fake where `r.iter_bytes` is an
  async GENERATOR (`AsyncMock(side_effect=async_gen)`, not
  `return_value=`).
- **VAD state tests stub the model** (`vad._model` = callable returning
  `[confidence]` plus `reset_states()`) — the real Silero is speech-trained
  and reads synthetic tones as silence. Use a FRESH analyzer per scenario:
  the volume tracker's smoothed state persists across calls.
- **loguru capture:** the sink callable receives the formatted STRING.
- **Timeouts:** pytest.ini carries a 120 s blanket; add tighter per-test
  `@pytest.mark.timeout` only for slow real-inference tests (STT).
- **Zero hardcoded tunables in tests too** — derive expected values from
  `CFG.*`; the bot-contract AST scan is the pattern for enforcing it.

## Continuous integration

`.github/workflows/ci.yml` mirrors this file's verification workflow on every
push to main and every PR, as two jobs on ubuntu-latest / Python 3.12:

- `lint` — `ruff check .` only. No project install (ruff reads its pin and
  select set from `pyproject.toml`), so this job never touches portaudio.
- `test` — `py_compile` + bot import gate + `pytest tests/ -q` after
  `pip install -e ".[test]"`.

What CI must add that a local run does not (all encoded in the workflow, not
the repo):

- `cp config_template.yaml config.yaml` — config.yaml is gitignored; conftest
  switches the whole suite off without it (Fresh-clone safety above).
- `portaudio19-dev` via apt — PyAudio (pulled by `pipecat-ai[local]`) ships
  Windows wheels only, so the sdist build needs portaudio headers on Linux.
- `actions/cache` on `model_cache/` — Moonshine downloads ONNX weights once
  (~8 s cold); a prior run's cache makes later runs load warm.

## Code style

Google docstrings; ASCII-only; comments only where strictly necessary (single
line); meaningful names; no `print()` debug leftovers (loguru only); zero magic
numbers (they belong in config.yaml). Ruff-clean required.

## Verification workflow (before declaring done)

1. Delete any stray probe/scratch files created during the round
2. `ruff check .` + py_compile + offline suite (`pytest tests/ -q`, above)
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