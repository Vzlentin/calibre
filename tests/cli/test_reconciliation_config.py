from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from calibre.cli.commands import run_config
from calibre.cli.config import ReconciliationConfig, load_config_from_mapping
from calibre.execution.backend import BackendResult
from calibre.execution.ledger import InMemoryLedger
from calibre.reconciliation import (
    BottomUpReconciler,
    NixtlaHierarchicalIntervalPhase,
    NixtlaReconciler,
    NoOpReconciler,
)

_M5_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "m5"

_VALID: dict[str, Any] = {
    "config_schema": "1.0",
    "dataset": {"adapter": "vn2", "path": "ignored", "period": 2},
    "tasks": [{"model": "SeasonalNaive", "horizon": 2, "config": {"backend": "statsforecast"}}],
    "origins": {"start": "2024-01-29", "end": "2024-02-05", "freq": "W-MON"},
    "output": {"streaming": False},
    "execution": {"backend": "local", "seed": 42},
}


def _config(**overrides: Any) -> dict[str, Any]:
    data = copy.deepcopy(_VALID)
    data.update(overrides)
    return data


def test_omitted_section_defaults_to_noop() -> None:
    config = load_config_from_mapping(_config())
    assert config.reconciliation is None
    # The run path falls back to a no-op reconciler when the section is absent.
    assert isinstance(ReconciliationConfig().to_reconciler(), NoOpReconciler)


def test_strategy_none_round_trips_to_noop() -> None:
    config = load_config_from_mapping(_config(reconciliation={"strategy": "none"}))
    assert config.reconciliation is not None
    assert config.reconciliation.strategy == "none"
    assert isinstance(config.reconciliation.to_reconciler(), NoOpReconciler)


@pytest.mark.parametrize(
    "strategy",
    ["ols", "wls_struct", "mint_shrink", "wls_var", "erm"],
)
def test_nixtla_strategy_resolves(strategy: str) -> None:
    config = load_config_from_mapping(_config(reconciliation={"strategy": strategy}))
    reconciler = config.reconciliation.to_reconciler()
    assert isinstance(reconciler, NixtlaReconciler)
    assert reconciler.strategy == strategy


def test_bottom_up_strategy_resolves_to_native_reconciler() -> None:
    config = load_config_from_mapping(_config(reconciliation={"strategy": "bottom_up"}))
    assert isinstance(config.reconciliation.to_reconciler(), BottomUpReconciler)


@pytest.mark.parametrize("strategy", ["mint", "mint_cov", "top_down", "bogus"])
def test_unknown_strategy_value_raises_listing_valid_choices(strategy: str) -> None:
    with pytest.raises(ValidationError, match="strategy"):
        load_config_from_mapping(_config(reconciliation={"strategy": strategy}))


def test_unknown_key_under_reconciliation_is_forbidden() -> None:
    with pytest.raises(ValidationError, match="bogus_knob"):
        load_config_from_mapping(_config(reconciliation={"strategy": "none", "bogus_knob": 1}))


def test_weighting_key_under_reconciliation_is_forbidden() -> None:
    with pytest.raises(ValidationError, match="weighting"):
        load_config_from_mapping(_config(reconciliation={"strategy": "ols", "weighting": "ols"}))


def test_run_config_passes_bundle_hierarchy_into_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_config threads the bundle hierarchy + resolved reconciler into the engine."""
    captured: dict[str, Any] = {}

    class _FakeEngine:
        def __init__(self, **kwargs: Any) -> None:
            captured["reconciliation"] = kwargs["reconciliation"]

        def execute(self, tasks: Any, actuals: Any, origins: Any) -> BackendResult:
            return BackendResult(ledger=InMemoryLedger())

        def close(self) -> None:
            return None

    monkeypatch.setattr("calibre.cli.commands.BackendEngine", _FakeEngine)

    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "m5", "path": str(_M5_FIXTURE)},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 1,
                    "config": {"backend": "statsforecast", "season_length": 7},
                }
            ],
            "origins": {"start": "2011-01-30", "end": "2011-01-30", "freq": "D"},
            "reconciliation": {"strategy": "ols"},
            "execution": {"backend": "local", "seed": 42},
        }
    )
    run_config(config)

    options = captured["reconciliation"]
    assert isinstance(options.reconciler, NixtlaReconciler)
    assert options.reconciler.strategy == "ols"
    assert options.hierarchy_index is not None
    assert "unique_id" in options.hierarchy_index.frame.columns


def test_hierarchical_intervals_section_resolves_to_fused_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeEngine:
        def __init__(self, **kwargs: Any) -> None:
            captured["reconciliation"] = kwargs["reconciliation"]
            captured["hierarchical_intervals"] = kwargs["hierarchical_intervals"]

        def execute(self, tasks: Any, actuals: Any, origins: Any) -> BackendResult:
            captured["tasks"] = tasks
            return BackendResult(ledger=InMemoryLedger())

        def close(self) -> None:
            return None

    monkeypatch.setattr("calibre.cli.commands.BackendEngine", _FakeEngine)

    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "m5", "path": str(_M5_FIXTURE)},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 1,
                    "config": {"backend": "statsforecast", "season_length": 7},
                }
            ],
            "origins": {"start": "2011-01-30", "end": "2011-01-30", "freq": "D"},
            "hierarchical_intervals": {
                "method": "nixtla_conformal",
                "coverage": 0.9,
                "strategy": "ols",
            },
            "execution": {"backend": "local", "seed": 42},
        }
    )
    run_config(config)

    reconciliation = captured["reconciliation"]
    fused = captured["hierarchical_intervals"]
    assert isinstance(fused.phase, NixtlaHierarchicalIntervalPhase)
    assert fused.phase.options.strategy == "ols"
    assert reconciliation.hierarchy_index is not None
    assert reconciliation.reconciler is None
    assert len(captured["tasks"].local) > 2


def test_hierarchical_intervals_unknown_key_is_forbidden() -> None:
    with pytest.raises(ValidationError, match="bogus_knob"):
        load_config_from_mapping(
            _config(
                conformal=None,
                hierarchical_intervals={
                    "method": "nixtla_conformal",
                    "strategy": "bottom_up",
                    "bogus_knob": True,
                },
            )
        )


def test_hierarchical_intervals_rejects_conformal_double_calibration() -> None:
    with pytest.raises(ValidationError, match="cannot be combined with conformal"):
        load_config_from_mapping(
            _config(
                conformal={"method": "aci", "coverage": 0.9},
                hierarchical_intervals={"method": "nixtla_conformal"},
            )
        )


def test_hierarchical_intervals_rejects_point_reconciliation() -> None:
    with pytest.raises(ValidationError, match="non-none reconciliation"):
        load_config_from_mapping(
            _config(
                conformal=None,
                reconciliation={"strategy": "ols"},
                hierarchical_intervals={"method": "nixtla_conformal"},
            )
        )


def test_hierarchical_intervals_rejects_duplicate_task_model_names() -> None:
    with pytest.raises(ValidationError, match="requires unique task model names"):
        load_config_from_mapping(
            _config(
                conformal=None,
                tasks=[
                    {
                        "model": "SeasonalNaive",
                        "horizon": 2,
                        "config": {"backend": "statsforecast", "season_length": 7},
                    },
                    {
                        "model": "SeasonalNaive",
                        "horizon": 2,
                        "config": {"backend": "statsforecast", "season_length": 14},
                    },
                ],
                hierarchical_intervals={"method": "nixtla_conformal"},
            )
        )


def test_hierarchical_intervals_requires_dataset_hierarchy() -> None:
    config = load_config_from_mapping(
        _config(
            conformal=None,
            dataset={"adapter": "vn2", "path": "benchmarks/vn2/fixture", "period": 0},
            hierarchical_intervals={"method": "nixtla_conformal"},
        )
    )

    with pytest.raises(ValueError, match="requires a dataset hierarchy"):
        run_config(config)
