from __future__ import annotations

from pathlib import Path

import pandas as pd

from calibre.cli.commands import run_config
from calibre.cli.config import load_config
from calibre.core.forecast_frame import H

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "smoke.yaml"
_FULL_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "full.yaml"


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


def test_m5_full_config_parses_without_partition_prerequisite() -> None:
    config = load_config(_FULL_CONFIG)

    assert config.dataset.adapter == "m5"
    assert config.dataset.path == "data/m5"
    assert config.dataset.options["phase"] == "evaluation"
    assert config.conformal is not None
    assert config.conformal.method == "mscp"
    assert config.conformal.mode == "perhorizon"
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


def test_m5_full_origin_window_meets_mscp_horizon_invariant() -> None:
    config = load_config(_FULL_CONFIG)
    assert config.conformal is not None

    runtime_config = config.conformal.to_runtime_config()
    horizon = config.tasks[0].horizon
    assert runtime_config.resolved_quantile_rule == "higher"
    first_ready_count = _first_finite_higher_quantile_count(runtime_config.alpha)
    minimum_origins = first_ready_count + (horizon - 1)

    assert horizon == 28
    assert first_ready_count == 10
    assert len(config.origins.to_list()) >= minimum_origins
