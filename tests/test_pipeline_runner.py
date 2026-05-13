"""Integration tests for calibre.execution.runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.execution.backend import BackendResult
from calibre.execution.dataset import DatasetBundle
from calibre.execution.ledger import ForecastLedger as Ledger
from calibre.execution.runner import PipelineResult, run_backtest, run_forecast

_MODEL_CONFIGS = [{"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 52}]
_SERIES_FILTER = ["0_126"]
_HORIZON = 4
_PERIOD = 0


@pytest.fixture(scope="module")
def dataset_bundle(data_dir: Path) -> DatasetBundle:
    from benchmarks.vn2.dataset import VN2DatasetAdapter

    return VN2DatasetAdapter().load(data_dir, period=_PERIOD)


@pytest.fixture(scope="module")
def backtest_result(data_dir: Path) -> PipelineResult:
    return run_backtest(
        data_dir=data_dir,
        period=_PERIOD,
        model_configs=_MODEL_CONFIGS,
        horizon=_HORIZON,
        origins=2,
        series_filter=_SERIES_FILTER,
        freq="W-MON",
    )


def test_run_backtest_smoke(backtest_result: PipelineResult) -> None:
    """run_backtest returns a PipelineResult with a non-empty ledger."""
    assert isinstance(backtest_result, PipelineResult)
    assert isinstance(backtest_result.ledger, Ledger)
    ledger_df = backtest_result.ledger.to_df()
    assert not ledger_df.empty, "Ledger DataFrame should not be empty"


def test_run_backtest_scores(backtest_result: PipelineResult) -> None:
    """run_backtest result has non-empty scores with at least one metric column."""
    scores = backtest_result.scores
    assert scores is not None
    assert isinstance(scores, pd.DataFrame)
    assert not scores.empty, "Scores DataFrame should not be empty"
    assert "mae" in scores.columns, f"Expected 'mae' column, got: {list(scores.columns)}"


def test_run_forecast_returns_backend_result(data_dir: Path) -> None:
    """run_forecast returns a BackendResult with non-empty ledger and y_hat values."""
    result = run_forecast(
        data_dir=data_dir,
        period=_PERIOD,
        model_configs=_MODEL_CONFIGS,
        horizon=_HORIZON,
        series_filter=_SERIES_FILTER,
        freq="W-MON",
    )
    assert isinstance(result, BackendResult)
    assert isinstance(result.ledger, Ledger)
    df = result.ledger.to_df()
    assert not df.empty, "Forecast Ledger DataFrame should not be empty"
    assert "y_hat" in df.columns, f"Expected 'y_hat' column, got: {list(df.columns)}"
    assert df["y_hat"].notna().any(), "At least some y_hat values should be non-null"


def test_run_backtest_with_conformal_config_enriches_ledger(data_dir: Path) -> None:
    conformal_config = SymmetricIntervalConfig(
        method="aci",
        coverage=0.9,
        calibration_window=4,
        gamma=0.05,
    )
    result = run_backtest(
        data_dir=data_dir,
        period=_PERIOD,
        model_configs=_MODEL_CONFIGS,
        horizon=_HORIZON,
        origins=2,
        series_filter=_SERIES_FILTER,
        freq="W-MON",
        conformal_runtime_factory=lambda: SymmetricIntervalRuntime(conformal_config),
    )
    lower_col, upper_col = conformal_config.interval_columns
    ledger_df = result.ledger.to_df()
    assert lower_col in ledger_df.columns
    assert upper_col in ledger_df.columns
    assert "coverage" in result.scores.columns
    assert "mean_interval_width" in result.scores.columns


def test_run_backtest_with_mscp_config_enriches_ledger(data_dir: Path) -> None:
    conformal_config = SymmetricIntervalConfig(
        method="mscp",
        coverage=0.9,
        calibration_window=4,
    )
    result = run_backtest(
        data_dir=data_dir,
        period=_PERIOD,
        model_configs=_MODEL_CONFIGS,
        horizon=_HORIZON,
        origins=2,
        series_filter=_SERIES_FILTER,
        freq="W-MON",
        conformal_runtime_factory=lambda: SymmetricIntervalRuntime(conformal_config),
    )
    lower_col, upper_col = conformal_config.interval_columns
    ledger_df = result.ledger.to_df()
    assert lower_col in ledger_df.columns
    assert upper_col in ledger_df.columns
    assert ledger_df["conformal_method"].eq("mscp").all()


def test_run_backtest_accepts_dataset_bundle(dataset_bundle: DatasetBundle) -> None:
    result = run_backtest(
        data_dir=dataset_bundle,
        model_configs=_MODEL_CONFIGS,
        horizon=_HORIZON,
        origins=1,
        series_filter=_SERIES_FILTER,
        freq="W-MON",
    )
    assert isinstance(result, PipelineResult)
    assert not result.ledger.to_df().empty
