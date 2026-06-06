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
from calibre.reconciliation import BottomUpReconciler, MinTReconciler, NoOpReconciler

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


def test_strategy_bottom_up_resolves() -> None:
    config = load_config_from_mapping(_config(reconciliation={"strategy": "bottom_up"}))
    assert isinstance(config.reconciliation.to_reconciler(), BottomUpReconciler)


def test_strategy_mint_shrinkage_resolves_with_weighting() -> None:
    config = load_config_from_mapping(
        _config(reconciliation={"strategy": "mint", "weighting": "shrinkage"})
    )
    reconciler = config.reconciliation.to_reconciler()
    assert isinstance(reconciler, MinTReconciler)
    assert reconciler.weighting == "shrinkage"


def test_unknown_strategy_value_raises_listing_valid_choices() -> None:
    with pytest.raises(ValidationError, match="strategy"):
        load_config_from_mapping(_config(reconciliation={"strategy": "bogus"}))


def test_unknown_key_under_reconciliation_is_forbidden() -> None:
    with pytest.raises(ValidationError, match="bogus_knob"):
        load_config_from_mapping(_config(reconciliation={"strategy": "none", "bogus_knob": 1}))


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
            "reconciliation": {"strategy": "bottom_up"},
            "execution": {"backend": "local", "seed": 42},
        }
    )
    run_config(config)

    options = captured["reconciliation"]
    assert isinstance(options.reconciler, BottomUpReconciler)
    assert options.hierarchy is not None
    assert "unique_id" in options.hierarchy.columns
