"""Integration tests for calibre.execution.runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.core.forecast_frame import UNIQUE_ID, Y_HAT, H, Y
from calibre.execution.backend import BackendResult
from calibre.execution.data_loading import load_period
from calibre.execution.dataset import DatasetBundle
from calibre.execution.ledger import Ledger
from calibre.execution.runner import PipelineResult, _resolve_bundle, run_backtest, run_forecast
from calibre.execution.vn2_adapter import VN2DatasetAdapter

_MODEL_CONFIGS = [{"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 52}]
_SERIES_FILTER = ["0_126"]
_HORIZON = 4
_PERIOD = 0


@pytest.fixture(scope="module")
def dataset_bundle(data_dir: Path) -> DatasetBundle:
    return VN2DatasetAdapter().load(data_dir, period=_PERIOD)


def test_resolve_bundle_carries_vn2_hierarchy(data_dir: Path) -> None:
    bundle = _resolve_bundle(data_dir, period=_PERIOD)
    assert bundle.hierarchy is not None
    assert not bundle.hierarchy.empty


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
    assert isinstance(backtest_result, PipelineResult)
    assert isinstance(backtest_result.ledger, Ledger)
    ledger_df = backtest_result.ledger.to_df()
    # 1 series x 2 origins x horizon 4 = 8 forecast rows.
    assert len(ledger_df) == 2 * _HORIZON
    assert ledger_df[UNIQUE_ID].unique().tolist() == _SERIES_FILTER
    assert sorted(ledger_df[H].unique()) == [1, 2, 3, 4]


def test_run_backtest_scores(backtest_result: PipelineResult) -> None:
    scores = backtest_result.scores
    assert not scores.empty
    assert "mae" in scores.columns

    # The reported MAE must equal the mean absolute error recomputed directly
    # from the resolved (y, y_hat) rows in the ledger — i.e. scores genuinely
    # reflect ledger content rather than being fabricated.
    resolved = backtest_result.ledger.to_df().dropna(subset=[Y, Y_HAT])
    assert not resolved.empty
    expected_mae = (
        resolved.assign(_ae=(resolved[Y] - resolved[Y_HAT]).abs())
        .groupby([UNIQUE_ID, H])["_ae"]
        .mean()
        .rename("mae")
        .sort_index()
    )
    actual_mae = scores.set_index([UNIQUE_ID, H])["mae"].sort_index()
    pd.testing.assert_series_equal(actual_mae, expected_mae)


def test_run_forecast_returns_backend_result(data_dir: Path) -> None:
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

    # Single latest origin, one series, horizon 4.
    assert df[H].tolist() == [1, 2, 3, 4]
    assert df[UNIQUE_ID].unique().tolist() == _SERIES_FILTER
    y_hat = df[Y_HAT]
    assert y_hat.notna().all()
    assert (y_hat >= 0).all()

    # SeasonalNaive copies observations one season back, so every forecast value
    # must be a value actually observed in this series' history.
    history = load_period(data_dir, _PERIOD)
    observed = history.loc[history[UNIQUE_ID] == _SERIES_FILTER[0], Y].dropna().to_numpy()
    for value in y_hat:
        assert np.min(np.abs(observed - value)) < 1e-6


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
    assert "coverage" in result.scores.columns
    assert "mean_interval_width" in result.scores.columns

    # Symmetric ACI intervals must bracket the point forecast: lo <= y_hat <= hi.
    enriched = ledger_df.dropna(subset=[lower_col, upper_col])
    assert not enriched.empty
    assert (enriched[lower_col] <= enriched[Y_HAT]).all()
    assert (enriched[Y_HAT] <= enriched[upper_col]).all()


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
    # MSCP is split-conformal: with only 2 origins it never fills the
    # calibration window, so the interval columns are present-but-NaN here.
    # The method marker is what proves the MSCP path actually ran.
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
