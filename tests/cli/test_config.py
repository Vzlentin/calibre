from __future__ import annotations

import pytest

from calibre.cli.config import load_config_from_mapping


def _config() -> dict[str, object]:
    return {
        "config_schema": "1.0",
        "dataset": {"adapter": "unit_cli", "path": "ignored"},
        "tasks": [{"model": "stub_model", "horizon": 1, "config": {"backend": "stub"}}],
        "origins": {"start": "2024-02-04", "end": "2024-02-04", "freq": "W-SUN"},
        "output": {},
        "execution": {"backend": "local"},
    }


def test_invalid_method_raises_at_parse_time() -> None:
    config = _config()
    config["conformal"] = {"method": "bogus"}

    with pytest.raises(ValueError, match="conformal.method"):
        load_config_from_mapping(config)


def test_invalid_mode_raises_at_parse_time() -> None:
    config = _config()
    config["conformal"] = {"method": "aci", "mode": "rolling"}

    with pytest.raises(ValueError, match="conformal.mode"):
        load_config_from_mapping(config)


def test_unknown_key_raises_at_parse_time() -> None:
    config = _config()
    config["unexpected"] = True

    with pytest.raises(ValueError, match="unexpected"):
        load_config_from_mapping(config)
