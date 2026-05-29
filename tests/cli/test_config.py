from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from calibre.cli.config import load_config_from_mapping

_VALID: dict[str, Any] = {
    "config_schema": "1.0",
    "dataset": {"adapter": "vn2", "path": "ignored", "period": 2, "currency": "EUR"},
    "tasks": [{"model": "SeasonalNaive", "horizon": 2, "config": {"backend": "statsforecast"}}],
    "conformal": {"method": "mscp", "coverage": 0.9, "mode": "perhorizon"},
    "origins": {"start": "2024-01-29", "end": "2024-02-05", "freq": "W-MON"},
    "output": {"streaming": False},
    "execution": {"backend": "local", "seed": 42},
}


def _config(**overrides: Any) -> dict[str, Any]:
    data = copy.deepcopy(_VALID)
    data.update(overrides)
    return data


def test_valid_mapping_round_trips() -> None:
    config = load_config_from_mapping(_config(), source_path="mem://c.yaml")

    assert config.conformal is not None
    assert config.conformal.method == "mscp"
    assert config.source_path == "mem://c.yaml"


def test_invalid_method_raises_at_parse_time() -> None:
    with pytest.raises(ValidationError, match="method"):
        load_config_from_mapping(_config(conformal={"method": "bogus"}))


def test_invalid_mode_raises_at_parse_time() -> None:
    with pytest.raises(ValidationError, match="mode"):
        load_config_from_mapping(_config(conformal={"method": "mscp", "mode": "bogus"}))


def test_unknown_key_raises_at_parse_time() -> None:
    with pytest.raises(ValidationError, match="conformal_window"):
        load_config_from_mapping(_config(conformal={"method": "mscp", "conformal_window": 10}))


def test_dataset_extra_keys_collected_into_options() -> None:
    config = load_config_from_mapping(_config())

    assert config.dataset.period == 2
    assert config.dataset.options == {"currency": "EUR"}


def test_origins_coerces_string_timestamps_and_validates_order() -> None:
    with pytest.raises(ValueError, match="origins.end must be greater than or equal"):
        load_config_from_mapping(
            _config(origins={"start": "2024-02-05", "end": "2024-01-29", "freq": "W-MON"})
        )
