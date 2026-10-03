"""Config loader gate: every section, key, read-only guard, and error path."""

from pathlib import Path

import pytest
import yaml

import config
from config import CFG, _Section

# Every section and key that tests rely on downstream; the loader must expose
# them all as attributes with values of sane types.
REQUIRED_SECTIONS = {
    "llm": ["base_url", "model", "api_key", "enable_thinking", "system_prompt"],
    "tts": ["base_url", "model", "voice", "stop_frame_timeout_s"],
    "audio": ["in_sample_rate", "out_sample_rate"],
    "stt": ["cache_dir"],
    "vad": ["confidence", "start_secs", "stop_secs", "min_volume"],
}

REQUIRED_TYPES = {
    ("llm", "base_url"): str,
    ("llm", "model"): str,
    ("llm", "enable_thinking"): bool,
    ("tts", "stop_frame_timeout_s"): float,
    ("audio", "in_sample_rate"): int,
    ("audio", "out_sample_rate"): int,
    ("vad", "confidence"): float,
    ("vad", "start_secs"): float,
}


def test_every_section_and_key_present_with_type():
    for section, keys in REQUIRED_SECTIONS.items():
        sec = getattr(CFG, section)
        assert isinstance(sec, _Section)
        for key in keys:
            value = getattr(sec, key)
            type_key = (section, key)
            if type_key in REQUIRED_TYPES:
                expected = REQUIRED_TYPES[type_key].__name__
                assert isinstance(value, REQUIRED_TYPES[type_key]), (
                    f"CFG.{section}.{key} should be {expected}, "
                    f"got {type(value).__name__}"
                )


def test_read_only_guard_blocks_attribute_write_at_every_level():
    with pytest.raises(AttributeError):
        CFG.new_key = 1  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError):
        CFG.llm.base_url = "http://changed"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(AttributeError):
        CFG.audio.in_sample_rate = 8000  # pyright: ignore[reportAttributeAccessIssue]


def test_failed_write_leaves_value_uncorrupted():
    original = CFG.llm.base_url
    with pytest.raises(AttributeError):
        CFG.llm.base_url = "http://hijacked"
    # The failed write must not have landed through any side door.
    assert CFG.llm.base_url == original


def test_unknown_key_access_raises():
    with pytest.raises(AttributeError):
        _ = CFG.llm.nonexistent_key  # noqa: B018
    with pytest.raises(AttributeError):
        _ = CFG.nonexistent_section  # noqa: B018


def test_missing_config_file_raises(tmp_path):
    import importlib.util
    import shutil

    shutil.copy(config.__file__, tmp_path / "config.py")
    spec = importlib.util.spec_from_file_location(
        "config_missing", tmp_path / "config.py"
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(FileNotFoundError):
        spec.loader.exec_module(module)


def test_non_mapping_yaml_raises(tmp_path):
    import importlib.util
    import shutil

    shutil.copy(config.__file__, tmp_path / "config.py")
    (tmp_path / "config.yaml").write_text("- just\n- a\n- list\n")
    spec = importlib.util.spec_from_file_location(
        "config_yaml_list", tmp_path / "config.py"
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(TypeError):
        spec.loader.exec_module(module)


def test_section_wraps_nested_dicts_transitively():
    sec = _Section({"a": {"b": {"c": [1, 2, 3]}}})
    assert sec.a.b.c == [1, 2, 3]
    # Lists stay plain values, not wrapped sections.
    assert isinstance(sec.a.b.c, list)


def _shape(data: dict, prefix: str = ""):
    """Flatten a nested mapping into dotted-keys leaf paths."""
    keys = set()
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys |= _shape(value, path)
        else:
            keys.add(path)
    return keys


def test_template_and_live_config_share_structure():
    """config.yaml is gitignored and evolves; the template is the starter copy.
    Values may drift (voice/model renames), so the gate enforces SECTION+KEY
    structure equality, not value equality.

    Fresh-clone note: config.yaml only exists after the Quick-start cp step.
    Without it the whole suite is skipped at conftest collection time.
    """
    live_path = Path(config.__file__).parent / "config.yaml"
    if not live_path.is_file():
        pytest.skip("config.yaml not present yet (run cp config_template.yaml first)")
    with open(live_path, encoding="utf-8") as f:
        live = yaml.safe_load(f)
    with open(
        Path(config.__file__).parent / "config_template.yaml", encoding="utf-8"
    ) as f:
        template = yaml.safe_load(f)

    live_keys = _shape(live)
    template_keys = _shape(template)
    assert live_keys == template_keys, (
        f"structure drift: config.yaml-only={sorted(live_keys - template_keys)}, "
        f"template-only={sorted(template_keys - live_keys)}"
    )
    # Type classes must also agree (a string where a float lives is drift).
    for key in sorted(live_keys):
        lv, tv = live, template
        for part in key.split("."):
            lv, tv = lv[part], tv[part]
        assert type(lv) is type(tv), (
            f"{key}: type drift {type(lv).__name__} vs {type(tv).__name__}"
        )


def test_new_live_config_key_without_template_fails_with_clear_diff(tmp_path):
    """A key added to config.yaml only must fail this module's parity gate
    with a message that names the offending key."""
    root = Path(config.__file__).parent
    live_path = root / "config.yaml"
    if not live_path.is_file():
        pytest.skip("config.yaml not present yet (run cp config_template.yaml first)")
    with open(live_path, encoding="utf-8") as f:
        live = yaml.safe_load(f)
    live["new_section"] = {"key": "value"}
    drifted_path = tmp_path / "drifted.yaml"
    drifted_path.write_text(yaml.safe_dump(live), encoding="utf-8")

    with open(root / "config_template.yaml", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    diff = _shape(yaml.safe_load(drifted_path.read_text(encoding="utf-8"))) - _shape(
        template
    )
    assert diff == {"new_section.key"}, (
        f"parity gate must flag template-missing keys, got {diff}"
    )
