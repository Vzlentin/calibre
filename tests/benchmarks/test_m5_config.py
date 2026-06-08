from __future__ import annotations

from pathlib import Path

from calibre.cli.commands import run_config
from calibre.cli.config import load_config
from calibre.conformal.calibrators import RollingQuantileCalibrator
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
    calibrator = RollingQuantileCalibrator(
        calibration_window=runtime_config.calibration_window,
        quantile_rule=runtime_config.resolved_quantile_rule,
    )
    first_ready_count = None
    for count in range(1, runtime_config.calibration_window + 1):
        calibrator.update(1.0)
        if calibrator.ready(alpha=runtime_config.alpha):
            first_ready_count = count
            break

    horizon = config.tasks[0].horizon
    assert first_ready_count is not None
    minimum_origins = first_ready_count + (horizon - 1)

    assert horizon == 28
    assert first_ready_count == 10
    assert len(config.origins.to_list()) >= minimum_origins
