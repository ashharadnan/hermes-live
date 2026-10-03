"""Shared test setup: config-gated collection, Moonshine cache redirect, import path."""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

# Fresh-clone gate: config.yaml only exists after the Quick-start cp step.
# Every project module imports `config` at module top level, so a missing
# config.yaml is a COLLECTION error, not a runtime skip -- switch the whole
# suite off here instead of sprinkling per-test guards that never run.
if not (_PROJECT_ROOT / "config.yaml").is_file():
    collect_ignore = [
        "test_bot_contract.py",
        "test_config.py",
        "test_frame_flow.py",
        "test_llm_service.py",
        "test_pipeline_assembly.py",
        "test_stt_service.py",
        "test_tts_service.py",
        "test_vad.py",
    ]

# Make project-root modules (config, bot, services) importable from tests/.
sys.path.insert(0, str(_PROJECT_ROOT))

# Derive the Moonshine cache redirect from config.yaml exactly like bot.py
# does, so the two can never silently disagree. Only reachable when
# config.yaml exists (collect_ignore above otherwise skips the modules that
# would construct Moonshine).
try:
    from config import CFG

    _cache_dir = CFG.stt.cache_dir
except (FileNotFoundError, ImportError):
    _cache_dir = "model_cache"

# Moonshine resolves its model cache at service construction; set it before
# any pipecat import. bot.py apply-setdefaults the same derived value.
os.environ.setdefault(
    "MOONSHINE_VOICE_CACHE",
    str(_PROJECT_ROOT.resolve() / _cache_dir),
)

from loguru import logger  # noqa: E402

logger.remove()
