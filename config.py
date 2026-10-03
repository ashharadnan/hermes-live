"""Config loader: parses config.yaml into read-only nested CFG namespaces.

Usage: CFG.llm.base_url, CFG.tts.voice, CFG.audio.in_sample_rate.
"""

from pathlib import Path

import yaml


class _Section:
    """Read-only attribute view over a dict of config.yaml."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            object.__setattr__(
                self, key, _Section(value) if isinstance(value, dict) else value
            )

    def __setattr__(self, name, value) -> None:
        raise AttributeError("Config is read-only; edit config.yaml instead")


def _load_config() -> _Section:
    """Load config.yaml from this file's directory."""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.yaml not found (expected {config_path})")
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(
            f"config.yaml must be a YAML mapping, got {type(data).__name__}"
        )
    return _Section(data)


CFG = _load_config()
