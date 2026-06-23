"""Tests for CLI YAML config loading and validation."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from calibre.cli.config import load_config_from_mapping
from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig
from calibre.conformal.partitions import GLOBAL_PARTITION, global_partition, series_partition
from calibre.core.forecast_frame import UNIQUE_ID

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


def _order_conformal_config(**order_conformal: Any) -> dict[str, Any]:
    # order_conformal cannot co-reside with conformal yet, so drop the default
    # diagnostic block from the base fixture for these scenarios.
    data = copy.deepcopy(_VALID)
    data.pop("conformal", None)
    data["order_conformal"] = order_conformal
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


def test_conformal_partition_defaults_to_global_runtime_key() -> None:
    config = load_config_from_mapping(_config(conformal={"method": "mscp"}))

    assert config.conformal is not None
    runtime_config = config.conformal.to_runtime_config()

    assert runtime_config.partition_key({UNIQUE_ID: "A"}) == GLOBAL_PARTITION


def test_conformal_partition_series_uses_unique_id_runtime_key() -> None:
    config = load_config_from_mapping(
        _config(conformal={"method": "mscp", "partition": "series", "max_partitions": 10})
    )

    assert config.conformal is not None
    assert config.conformal.max_partitions == 10
    runtime_config = config.conformal.to_runtime_config()

    assert runtime_config.partition_key({UNIQUE_ID: "A"}) == "A"


def test_invalid_partition_raises_at_parse_time() -> None:
    with pytest.raises(ValidationError, match="partition"):
        load_config_from_mapping(_config(conformal={"method": "mscp", "partition": "bogus"}))


def test_coherent_draws_spread_threads_to_runtime_config() -> None:
    config = load_config_from_mapping(
        _config(
            conformal={
                "method": "aci",
                "spread": "coherent_draws",
                "draw_count": 64,
                "draw_seed": 5,
            }
        )
    )
    assert config.conformal is not None
    runtime_config = config.conformal.to_runtime_config()
    assert runtime_config.spread == "coherent_draws"
    assert runtime_config.draw_count == 64
    assert runtime_config.draw_seed == 5


def test_coherent_draws_spread_rejects_non_none_reconciliation() -> None:
    with pytest.raises(ValidationError, match="coherent_draws"):
        load_config_from_mapping(
            _config(
                conformal={"method": "aci", "spread": "coherent_draws"},
                reconciliation={"strategy": "wls_struct"},
            )
        )


def test_invalid_spread_raises_at_parse_time() -> None:
    with pytest.raises(ValidationError, match="spread"):
        load_config_from_mapping(_config(conformal={"method": "mscp", "spread": "bogus"}))


def test_unknown_key_raises_at_parse_time() -> None:
    with pytest.raises(ValidationError, match="conformal_window"):
        load_config_from_mapping(_config(conformal={"method": "mscp", "conformal_window": 10}))


def test_execution_chunk_size_plumbs_into_execution_options() -> None:
    config = load_config_from_mapping(_config(execution={"backend": "local", "chunk_size": 32}))

    assert config.execution.chunk_size == 32
    assert config.execution.to_execution_options(freq="W-MON").chunk_size == 32


def test_execution_chunk_size_defaults_to_256() -> None:
    config = load_config_from_mapping(_config())

    assert config.execution.chunk_size == 256
    assert config.execution.to_execution_options(freq="W-MON").chunk_size == 256


def test_execution_chunk_size_rejects_non_positive() -> None:
    with pytest.raises(ValidationError, match="chunk_size"):
        load_config_from_mapping(_config(execution={"backend": "local", "chunk_size": 0}))


def test_dataset_extra_keys_collected_into_options() -> None:
    config = load_config_from_mapping(_config())

    assert config.dataset.period == 2
    assert config.dataset.options == {"currency": "EUR"}


def test_origins_coerces_string_timestamps_and_validates_order() -> None:
    with pytest.raises(ValueError, match="origins.end must be greater than or equal"):
        load_config_from_mapping(
            _config(origins={"start": "2024-02-05", "end": "2024-01-29", "freq": "W-MON"})
        )


# --- order_conformal decision-runtime config (U1) --------------------------


def test_order_conformal_round_trips_to_runtime_config() -> None:
    config = load_config_from_mapping(
        _order_conformal_config(
            coverage=0.74,
            calibration_window=5000,
            buffer_max=0.0,
            protection_period=3,
        )
    )

    assert config.conformal is None
    assert config.order_conformal is not None
    runtime_config = config.order_conformal.to_runtime_config()

    assert isinstance(runtime_config, CumulativeConformalRiskConfig)
    assert runtime_config.coverage == pytest.approx(0.74)
    assert runtime_config.calibration_window == 5000
    assert runtime_config.buffer_max == pytest.approx(0.0)
    assert runtime_config.protection_period == 3
    assert runtime_config.method_name == "capped_crc"
    assert runtime_config.weight_decay is None
    assert runtime_config.partition_key is global_partition


def test_order_conformal_partition_series_uses_series_runtime_key() -> None:
    config = load_config_from_mapping(_order_conformal_config(coverage=0.74, partition="series"))

    assert config.order_conformal is not None
    runtime_config = config.order_conformal.to_runtime_config()

    assert runtime_config.partition_key is series_partition
    assert runtime_config.partition_key({UNIQUE_ID: "A"}) == "A"


def test_order_conformal_unknown_key_raises_at_parse_time() -> None:
    with pytest.raises(ValidationError, match="coherage"):
        load_config_from_mapping(_order_conformal_config(coherage=0.74))


def test_order_conformal_invalid_partition_raises_at_parse_time() -> None:
    with pytest.raises(ValidationError, match="partition"):
        load_config_from_mapping(_order_conformal_config(coverage=0.74, partition="bogus"))


def test_conformal_and_order_conformal_both_set_rejected_with_s2_message() -> None:
    data = copy.deepcopy(_VALID)
    data["conformal"] = {"method": "mscp", "coverage": 0.9}
    data["order_conformal"] = {"coverage": 0.74}

    with pytest.raises(ValidationError, match="co-residence.*lands in S2"):
        load_config_from_mapping(data)


def test_order_conformal_base_column_defaults_to_none() -> None:
    config = load_config_from_mapping(_order_conformal_config(coverage=0.74))

    assert config.order_conformal is not None
    assert config.order_conformal.base_column is None


def test_order_conformal_base_column_carried_verbatim() -> None:
    config = load_config_from_mapping(_order_conformal_config(coverage=0.74, base_column="q_0p59"))

    assert config.order_conformal is not None
    assert config.order_conformal.base_column == "q_0p59"
    assert config.order_conformal.to_runtime_config().base_column == "q_0p59"
