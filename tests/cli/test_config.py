"""Tests for CLI YAML config loading and validation."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from calibre.cli.config import TaskConfig, load_config_from_mapping
from calibre.conformal.partitions import GLOBAL_PARTITION
from calibre.core.forecast_frame import UNIQUE_ID


def test_censoring_fit_gate_defaults_false_and_plumbs_into_model_config() -> None:
    default = TaskConfig(model="lightgbm.LGBMRegressor", horizon=3)
    assert default.censoring_fit is False
    assert default.resolved_model_config()["censoring_fit"] is False

    enabled = TaskConfig(model="lightgbm.LGBMRegressor", horizon=3, censoring_fit=True)
    assert enabled.resolved_model_config()["censoring_fit"] is True


def test_censoring_fit_inside_config_dict_does_not_enable_gate() -> None:
    # The gate is a task-root peer of model/horizon, not a config key. A stray
    # key inside config: does not flip the typed gate (it stays the default).
    task = TaskConfig(
        model="lightgbm.LGBMRegressor",
        horizon=3,
        config={"censoring_fit": True},
    )
    assert task.censoring_fit is False
    assert task.resolved_model_config()["censoring_fit"] is False


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


def test_order_conformal_maps_six_knobs_to_runtime_config() -> None:
    config = load_config_from_mapping(
        _config(
            conformal=None,
            order_conformal={
                "coverage": 0.74,
                "protection_period": 4,
                "calibration_window": 1000,
                "partition": "global",
                "base_column": "q_0p833",
                "buffer_max": 0.0,
            },
        )
    )

    assert config.order_conformal is not None
    runtime_config = config.order_conformal.to_runtime_config()

    assert runtime_config.coverage == 0.74
    assert runtime_config.protection_period == 4
    assert runtime_config.calibration_window == 1000
    assert runtime_config.base_column == "q_0p833"
    assert runtime_config.buffer_max == 0.0
    assert runtime_config.partition_key({UNIQUE_ID: "A"}) == GLOBAL_PARTITION


def test_order_conformal_winner_knob_subset_builds_implied_config() -> None:
    """The six-knob winner-shaped subset builds exactly the config it implies.

    This asserts the subset's mapped fields, not winner-runtime equivalence: the
    deployed winner also injects a tuned base column, protection period, and
    weight_decay/method the static block cannot supply (those need runtime
    injection paths this config surface does not expose). The block-implied
    config keeps the runtime defaults for those unset knobs.
    """
    config = load_config_from_mapping(
        _config(
            conformal=None,
            order_conformal={
                "coverage": 0.74,
                "calibration_window": 5000,
                "buffer_max": 0.0,
            },
        )
    )

    assert config.order_conformal is not None
    runtime_config = config.order_conformal.to_runtime_config()

    assert runtime_config.coverage == 0.74
    assert runtime_config.calibration_window == 5000
    assert runtime_config.buffer_max == 0.0
    # Unset knobs keep the cumulative-risk runtime defaults (not winner values).
    assert runtime_config.weight_decay == 0.85
    assert runtime_config.weighted_quantile_mode == "empirical"
    assert runtime_config.resolved_base_column == "y_hat"


def test_order_conformal_series_partition_uses_unique_id_runtime_key() -> None:
    config = load_config_from_mapping(
        _config(conformal=None, order_conformal={"partition": "series"})
    )

    assert config.order_conformal is not None
    runtime_config = config.order_conformal.to_runtime_config()

    assert runtime_config.partition_key({UNIQUE_ID: "A"}) == "A"


def test_order_conformal_unknown_key_raises_at_parse_time() -> None:
    with pytest.raises(ValidationError, match="weight_decay"):
        load_config_from_mapping(
            _config(conformal=None, order_conformal={"coverage": 0.5, "weight_decay": 0.9})
        )


def test_order_conformal_out_of_range_coverage_rejected_at_parse_time() -> None:
    # The bound (0 < coverage < 1) is enforced at parse time via the Field
    # constraint, so `calibre validate` rejects it without ever building the
    # runtime config (parity with protection_period/calibration_window).
    with pytest.raises(ValidationError, match="coverage"):
        load_config_from_mapping(_config(conformal=None, order_conformal={"coverage": 1.0}))


def test_order_conformal_runtime_config_also_guards_coverage() -> None:
    # Defense-in-depth: the runtime CumulativeConformalRiskConfig keeps its own
    # 0 < coverage < 1 invariant, so a coverage that bypasses the CLI surface is
    # still rejected when the config object is constructed directly.
    from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig

    with pytest.raises(ValueError, match="coverage"):
        CumulativeConformalRiskConfig(coverage=1.0)


def test_order_conformal_and_conformal_together_rejected() -> None:
    with pytest.raises(ValidationError, match="one runtime slot"):
        load_config_from_mapping(
            _config(
                conformal={"method": "mscp"},
                order_conformal={"coverage": 0.74},
            )
        )


def test_ordering_coverage_inherits_from_order_conformal_when_unset() -> None:
    """An omitted ordering.coverage is back-filled from order_conformal.coverage."""
    config = load_config_from_mapping(
        _config(
            conformal=None,
            order_conformal={"coverage": 0.74},
            ordering={"policy": "rs", "params": {"unique_id": "1_10"}},
        )
    )

    assert config.ordering is not None
    assert config.ordering.coverage == 0.74


def test_ordering_coverage_inherits_order_conformal_default_when_both_omitted() -> None:
    """Omitting coverage on BOTH blocks back-fills ordering from order_conformal's default.

    order_conformal and ordering carry different defaults (0.5 vs 0.9), so an
    omit-both rs decision-bound config must inherit order_conformal's effective
    default rather than collide the two and reject.
    """
    config = load_config_from_mapping(
        _config(
            conformal=None,
            order_conformal={"protection_period": 2},
            ordering={"policy": "rs", "params": {"unique_id": "1_10"}},
        )
    )

    assert config.order_conformal is not None
    assert config.ordering is not None
    assert config.ordering.coverage == config.order_conformal.coverage == 0.5


def test_ordering_coverage_matching_order_conformal_parses() -> None:
    """An explicit ordering.coverage equal to order_conformal.coverage is accepted."""
    config = load_config_from_mapping(
        _config(
            conformal=None,
            order_conformal={"coverage": 0.74},
            ordering={"policy": "rs", "coverage": 0.74, "params": {"unique_id": "1_10"}},
        )
    )

    assert config.ordering is not None
    assert config.ordering.coverage == 0.74


def test_ordering_coverage_mismatch_rejected_at_parse_time() -> None:
    """An explicit ordering.coverage that disagrees with order_conformal raises."""
    with pytest.raises(ValidationError, match="must equal order_conformal.coverage"):
        load_config_from_mapping(
            _config(
                conformal=None,
                order_conformal={"coverage": 0.74},
                ordering={"policy": "rs", "coverage": 0.9, "params": {"unique_id": "1_10"}},
            )
        )


def test_ordering_quantile_mode_exempt_from_coverage_sync() -> None:
    """Quantile mode reads a q_<p> column, not the bound, so coverage is unchecked."""
    config = load_config_from_mapping(
        _config(
            conformal=None,
            order_conformal={"coverage": 0.74},
            ordering={"policy": "rs", "quantile": 0.8, "params": {"unique_id": "1_10"}},
        )
    )

    assert config.ordering is not None
    # Not back-filled and not rejected: the default coverage stands.
    assert config.ordering.coverage == 0.9
    assert config.ordering.quantile == 0.8


def test_ordering_coverage_unchanged_without_order_conformal() -> None:
    """Without an order_conformal block, ordering.coverage is left as configured."""
    config = load_config_from_mapping(
        _config(
            conformal=None,
            ordering={"policy": "rs", "coverage": 0.9, "params": {"unique_id": "1_10"}},
        )
    )

    assert config.ordering is not None
    assert config.ordering.coverage == 0.9


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
