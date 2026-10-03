"""Import-level gate for bot.py: wiring must import without starting a pipeline.

Also pins the two runtime contracts the pipeline depends on:
  - MOONSHINE_VOICE_CACHE redirect to project model_cache/
  - AGENTS.md 'services stay implementation-neutral' log strings
"""

import os
from pathlib import Path

import pytest

import bot
from config import CFG


class TestModuleImport:
    def test_bot_module_imports_without_running(self):
        assert hasattr(bot, "main")
        assert callable(bot.main)

    def test_moonshine_cache_env_redirect_set(self):
        # bot.py sets the env var at import time via setdefault, resolving the
        # cache dir from CFG.stt.cache_dir.
        expected_cache = (
            Path(bot.__file__).parent / CFG.stt.cache_dir
        ).resolve()
        actual = os.environ.get("MOONSHINE_VOICE_CACHE")
        assert actual is not None
        assert Path(actual) == expected_cache

    def test_stt_cache_dir_is_project_relative(self):
        assert not Path(CFG.stt.cache_dir).is_absolute()


class TestImplementationNeutrality:
    def test_llm_service_class_named_by_endpoint_role_not_model(self):
        # AGENTS.md invariant: class names stay implementation-neutral. The
        # class name may encode the SERVER (llamacpp, chatterbox) but never
        # the MODEL. This test pins the current reality.
        assert "LlamacppLLMService" in dir(bot)
        assert "gemma" not in bot.LlamacppLLMService.__name__.lower()
        assert "chatterbox" in bot.ChatterboxTTSService.__name__.lower()

    def test_config_yaml_is_the_only_tunables_home(self):
        # Zero-magic-numbers rule: spot-check that no executable code embeds
        # endpoints, rates, or the timeout. Docstrings/comments may mention
        # them, so scan statements via ast, not raw text. The forbidden rates
        # derive from config.yaml so a rate change re-arms the scan without
        # editing this file.
        import ast

        import yaml

        root = Path(bot.__file__).parent
        with open(root / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        forbidden = {
            "8004",  # TTS port, legacy default
            "18080",  # LLM port, legacy default
            str(cfg["audio"]["in_sample_rate"]),
            str(cfg["audio"]["out_sample_rate"]),
            str(cfg["tts"]["stop_frame_timeout_s"]),
            str(cfg["vad"]["confidence"]),
            str(cfg["vad"]["start_secs"]),
            str(cfg["vad"]["stop_secs"]),
            str(cfg["vad"]["min_volume"]),
        }
        for name in [
            "bot.py",
            "config.py",
            "services/llamacpp_llm.py",
            "services/chatterbox_tts.py",
        ]:
            source = (root / name).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and (
                    str(node.value) in forbidden
                ):
                    pytest.fail(
                        f"{name}:{node.lineno} hardcodes {node.value!r}; "
                        "move it to config.yaml"
                    )
            if name == "services/chatterbox_tts.py":
                # The TTS timeout must flow from config through the ctor.
                assert "CFG.tts.stop_frame_timeout_s" in source
