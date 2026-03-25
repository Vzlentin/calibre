"""Integration tests for calibre.pipeline.runner."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from calibre.engine.ledger import Ledger
from calibre.pipeline.runner import PipelineResult, run_backtest, run_forecast

_MODEL_CONFIGS = [
    {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 52}
]
_SERIES_FILTER = ["0_126"]
_HORIZON = 4
_WEEK = 0


def _find_data_dir() -> Path:
    """Locate the data/ directory, handling git worktrees."""
    candidate = Path(__file__).parent.parent / "data"
    if candidate.is_dir():
        return candidate
    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=Path(__file__).parent,
            text=True,
        ).strip()
        project_root = Path(common_dir).parent
        candidate = project_root / "data"
        if candidate.is_dir():
            return candidate
    except subprocess.CalledProcessError:
        pass
    raise FileNotFoundError(
        f"Cannot find data/ directory. Tried {Path(__file__).parent.parent / 'data'}"
    )


@pytest.fixture(scope="module")
def data_dir() -> Path:
    return _find_data_dir()


@pytest.fixture(scope="module")
def backtest_result(data_dir: Path) -> PipelineResult:
    return run_backtest(
        data_dir=data_dir,
        week=_WEEK,
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


def test_run_forecast_returns_ledger(data_dir: Path) -> None:
    """run_forecast returns a Ledger with non-empty to_df() and y_hat values."""
    ledger = run_forecast(
        data_dir=data_dir,
        week=_WEEK,
        model_configs=_MODEL_CONFIGS,
        horizon=_HORIZON,
        series_filter=_SERIES_FILTER,
        freq="W-MON",
    )
    assert isinstance(ledger, Ledger)
    df = ledger.to_df()
    assert not df.empty, "Forecast Ledger DataFrame should not be empty"
    assert "y_hat" in df.columns, f"Expected 'y_hat' column, got: {list(df.columns)}"
    assert df["y_hat"].notna().any(), "At least some y_hat values should be non-null"
