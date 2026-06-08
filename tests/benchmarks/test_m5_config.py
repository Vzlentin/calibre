from __future__ import annotations

from math import isfinite
from pathlib import Path

from calibre.cli.commands import run_config
from calibre.cli.config import load_config
from calibre.conformal.numerics import finite_sample_radius
from calibre.core.forecast_frame import H

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "smoke.yaml"
_FULL_CONFIG = _REPO_ROOT / "benchmarks" / "m5" / "config" / "full.yaml"


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


def test_m5_full_origin_window_meets_mscp_horizon_invariant() -> None:
    config = load_config(_FULL_CONFIG)
    assert config.conformal is not None

    runtime_config = config.conformal.to_runtime_config()
    alpha = 1.0 - runtime_config.coverage
    first_finite_count = next(
        count
        for count in range(1, 128)
        if isfinite(
            finite_sample_radius(
                [1.0] * count,
                alpha,
                0.0,
                runtime_config.resolved_quantile_rule,
            )
        )
    )
    horizon = config.tasks[0].horizon
    minimum_origins = first_finite_count + (horizon - 1)

    assert horizon == 28
    assert first_finite_count == 10
    assert len(config.origins.to_list()) >= minimum_origins
