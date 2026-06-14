"""Tests for the committed M5 benchmark configs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from calibre.cli.commands import run_config
from calibre.cli.config import load_config
from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.core.forecast_frame import (
    CONFORMAL_MODE,
    H,
    interval_column_names,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "smoke.yaml"
_FULL_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "full.yaml"
_FULL_WLS_STRUCT_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "full-wls-struct.yaml"
_FULL_CUMULATIVE_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "full-cumulative.yaml"
_SMOKE_CUMULATIVE_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "smoke-cumulative.yaml"
_M5_README = _REPO_ROOT / "benchmarks" / "m5" / "README.md"


def _first_finite_higher_quantile_count(alpha: float) -> int:
    count = 1
    while alpha <= 1.0 / (count + 1):
        count += 1
    return count


def test_m5_smoke_config_parses_as_source_cli_config() -> None:
    config = load_config(_SMOKE_CONFIG)

    assert config.dataset.adapter == "m5"
    assert config.dataset.path == "tests/fixtures/m5"
    assert config.dataset.options["phase"] == "evaluation"
    assert config.output.ledger_path == "results/m5/smoke/forecast-ledger.parquet"


def test_m5_full_config_uses_series_partition_handoff() -> None:
    config = load_config(_FULL_CONFIG)

    assert config.dataset.adapter == "m5"
    assert config.dataset.path == "data/m5"
    assert config.dataset.options["phase"] == "evaluation"
    assert config.conformal is not None
    assert config.conformal.method == "mscp"
    assert config.conformal.mode == "perhorizon"
    assert config.conformal.coverage == 0.9
    assert config.conformal.calibration_window == 10
    assert config.conformal.partition == "series"
    assert config.conformal.max_partitions == 1_000_000
    assert config.reconciliation is not None
    assert config.reconciliation.strategy == "bottom_up"
    assert config.output.ledger_path == "results/m5/full-mscp-bottom-up/forecast-ledger.parquet"
    assert config.output.streaming is True
    assert config.execution.backend == "auto"


def test_m5_smoke_config_executes_source_run_path(tmp_path: Path) -> None:
    ledger_path = tmp_path / "results" / "m5" / "smoke" / "forecast-ledger.parquet"
    config = load_config(_SMOKE_CONFIG)
    config = config.model_copy(
        update={"output": config.output.model_copy(update={"ledger_path": str(ledger_path)})}
    )

    result = run_config(config)
    ledger = result.ledger.to_df()

    assert ledger_path.exists()
    assert not ledger.empty
    assert set(ledger[H]) == {1}
    written = pd.read_parquet(ledger_path)
    assert not written.empty
    assert len(written) == len(ledger)
    assert set(written[H]) == {1}


def test_m5_full_wls_struct_config_uses_series_partition_handoff() -> None:
    config = load_config(_FULL_WLS_STRUCT_CONFIG)

    assert config.dataset.adapter == "m5"
    assert config.dataset.path == "data/m5"
    assert config.dataset.options["phase"] == "evaluation"
    assert config.conformal is not None
    assert config.conformal.method == "mscp"
    assert config.conformal.mode == "perhorizon"
    assert config.conformal.coverage == 0.9
    assert config.conformal.calibration_window == 10
    assert config.conformal.partition == "series"
    assert config.conformal.max_partitions == 1_000_000
    assert config.reconciliation is not None
    assert config.reconciliation.strategy == "wls_struct"
    assert config.output.ledger_path == "results/m5/full-mscp-wls-struct/forecast-ledger.parquet"
    assert config.output.streaming is True
    assert config.execution.backend == "auto"


@pytest.mark.parametrize("config_path", [_FULL_CONFIG, _FULL_WLS_STRUCT_CONFIG])
def test_m5_full_origin_window_meets_mscp_horizon_invariant(config_path: Path) -> None:
    config = load_config(config_path)
    assert config.conformal is not None

    runtime_config = config.conformal.to_runtime_config()
    horizon = config.tasks[0].horizon
    assert runtime_config.resolved_quantile_rule == "higher"
    first_ready_count = _first_finite_higher_quantile_count(runtime_config.alpha)
    minimum_origins = first_ready_count + 2 * (horizon - 1)

    assert horizon == 28
    assert first_ready_count == 10
    assert len(config.origins.to_list()) == 64
    assert len(config.origins.to_list()) >= minimum_origins


def test_m5_runbook_keeps_fixture_out_of_statistical_acceptance() -> None:
    text = _M5_README.read_text(encoding="utf-8")

    assert "CI smoke fixture" in text
    assert "not statistical evidence for M5 coverage" in text
    assert "full run is a local acceptance run only" in text
    assert "input-side node-history materialization" in text
    assert "projected node-history rows" in text
    assert "coverage-by-node.parquet" in text
    assert "report.md" in text
    assert "full-population marginal coverage" in text
    assert "per-node outliers are emitted and counted as diagnostics only" in text
    assert "lazy aggregate actual resolution" in text


def test_m5_full_cumulative_config_selects_engine_internal_cumulative_mode() -> None:
    config = load_config(_FULL_CUMULATIVE_CONFIG)

    assert config.dataset.adapter == "m5"
    assert config.dataset.path == "data/m5"
    assert config.dataset.options["phase"] == "evaluation"
    assert config.conformal is not None
    assert config.conformal.method == "mscp"
    assert config.conformal.mode == "cumulative"
    assert config.conformal.protection_period == 28
    assert config.conformal.coverage == 0.9
    assert config.conformal.calibration_window == 10
    assert config.conformal.partition == "series"
    assert config.conformal.max_partitions == 1_000_000
    assert config.reconciliation is not None
    assert config.reconciliation.strategy == "bottom_up"
    assert config.tasks[0].horizon == 28
    assert config.output.ledger_path == "results/m5/full-mscp-cumulative/forecast-ledger.parquet"
    assert config.output.streaming is True
    assert config.execution.backend == "auto"

    runtime_config = config.conformal.to_runtime_config()
    # Constructing the runtime config proves the cumulative mscp + protection
    # invariant (SymmetricIntervalConfig.__post_init__) is satisfied.
    assert isinstance(runtime_config, SymmetricIntervalConfig)
    assert runtime_config.mode == "cumulative"
    assert runtime_config.protection_period == 28


def test_m5_smoke_cumulative_config_parses_as_source_cli_config() -> None:
    config = load_config(_SMOKE_CUMULATIVE_CONFIG)

    assert config.dataset.adapter == "m5"
    assert config.dataset.path == "tests/fixtures/m5"
    assert config.conformal is not None
    assert config.conformal.mode == "cumulative"
    assert config.conformal.protection_period == 2
    assert config.tasks[0].horizon == 2


def test_m5_smoke_cumulative_config_executes_cumulative_path(tmp_path: Path) -> None:
    ledger_path = tmp_path / "results" / "m5" / "smoke-cumulative" / "forecast-ledger.parquet"
    config = load_config(_SMOKE_CUMULATIVE_CONFIG)
    config = config.model_copy(
        update={"output": config.output.model_copy(update={"ledger_path": str(ledger_path)})}
    )
    assert config.conformal is not None
    protection_period = config.conformal.protection_period
    assert protection_period == 2
    lower_col, upper_col = interval_column_names(config.conformal.coverage)

    result = run_config(config)
    ledger = result.ledger.to_df()

    assert ledger_path.exists()
    assert not ledger.empty
    written = pd.read_parquet(ledger_path)
    assert len(written) == len(ledger)

    # The cumulative branch executed (not perhorizon).
    assert set(ledger[CONFORMAL_MODE]) == {"cumulative"}

    # The cumulative bound is written only on the terminal-H row (H ==
    # protection_period) of each (uid, model, origin) group, never on earlier-H
    # rows — directly asserting _apply_cumulative's terminal-row contract.
    terminal = ledger[ledger[H] == protection_period]
    earlier = ledger[ledger[H] < protection_period]
    assert earlier[lower_col].isna().all()
    assert earlier[upper_col].isna().all()

    # Once the per-series calibrator has warmed up, at least one terminal-H row
    # carries a populated, finite bound — proving the cumulative apply emits an
    # interval, not just NaN scaffolding.
    populated = terminal[terminal[lower_col].notna()]
    assert not populated.empty
    for _, row in populated.iterrows():
        assert row[lower_col] <= row[upper_col]

    # The incomplete-window deferral contract (a group with fewer than
    # protection_period resolved horizons gets no bound) is not reproducible at
    # smoke shape — horizon == protection_period makes every window complete — so
    # it is covered by the runtime's own unit tests, not asserted here.
