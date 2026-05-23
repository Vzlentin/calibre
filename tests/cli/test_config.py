from __future__ import annotations

import pytest

from calibre.cli.config import load_config_from_mapping


def _base_config() -> dict:
    return {
        "config_schema": "1.0",
        "dataset": {"adapter": "stub", "path": "stub://data"},
        "tasks": [{"model": "AutoARIMA", "horizon": 4}],
        "origins": {"start": "2024-01-01", "end": "2024-03-31", "freq": "W"},
    }


def test_invalid_method_raises_at_parse_time():
    cfg = _base_config()
    cfg["conformal"] = {"method": "invalid_method", "coverage": 0.9}
    with pytest.raises(ValueError, match="mscp.*aci|method"):
        load_config_from_mapping(cfg)


def test_invalid_mode_raises_at_parse_time():
    cfg = _base_config()
    cfg["conformal"] = {"method": "mscp", "mode": "not_a_valid_mode"}
    with pytest.raises(ValueError, match="perhorizon.*cumulative|mode"):
        load_config_from_mapping(cfg)


def test_unknown_key_raises_at_parse_time():
    cfg = _base_config()
    cfg["conformal"] = {"method": "mscp", "totally_unknown_key": True}
    with pytest.raises(ValueError):
        load_config_from_mapping(cfg)


def test_valid_conformal_config_parses_correctly():
    cfg = _base_config()
    cfg["conformal"] = {"method": "aci", "coverage": 0.95, "gamma": 0.1}
    result = load_config_from_mapping(cfg)
    assert result.conformal is not None
    assert result.conformal.method == "aci"
    assert result.conformal.coverage == 0.95


def test_invalid_execution_backend_raises():
    cfg = _base_config()
    cfg["execution"] = {"backend": "quantum_gpu"}
    with pytest.raises(ValueError):
        load_config_from_mapping(cfg)
