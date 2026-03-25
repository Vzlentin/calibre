"""End-to-end pipeline runner for backtesting and forward forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from calibre.contracts.forecast_frame import DS
from calibre.engine.backend import BackendEngine
from calibre.engine.ledger import Ledger
from calibre.engine.scoring import compute_metrics
from calibre.metrics import mae, rmse, smape, wape
from calibre.pipeline.loading import load_week
from calibre.pipeline.tasks import build_tasks

_DEFAULT_METRICS: list[Callable] = [mae, rmse, smape, wape]


@dataclass(frozen=True)
class PipelineResult:
    ledger: Ledger
    scores: pd.DataFrame | None
    sales: pd.DataFrame

    def to_parquet(self, path: str | Path) -> None:
        """Export ledger predictions to parquet."""
        self.ledger.to_parquet(str(path))


def _derive_origins(sales: pd.DataFrame, n: int, horizon: int) -> list[pd.Timestamp]:
    """Return last N origin timestamps that have horizon-worth of actuals after them."""
    all_dates = sorted(sales[DS].unique())
    return list(all_dates[-(n + horizon): -horizon])


def run_backtest(
    data_dir: str | Path,
    week: int,
    model_configs: list[dict],
    horizon: int,
    origins: list[pd.Timestamp] | int,
    metrics: list[Callable] | None = None,
    series_filter: list[str] | None = None,
    freq: str = "W",
    engine: Any = None,
) -> PipelineResult:
    """End-to-end backtest pipeline.

    Steps:
    1. Load sales data for the given week.
    2. Build forecast tasks (optionally filtered to a subset of series).
    3. Resolve origins: if an int N is given, derive the last N origin timestamps
       that still have horizon-worth of actuals ahead of them.
    4. Execute the backend engine to produce a Ledger.
    5. Compute aggregate metrics over resolved rows.
    6. Return a PipelineResult.
    """
    sales = load_week(data_dir, week)
    tasks = build_tasks(sales, model_configs, horizon, series_filter)

    if isinstance(origins, int):
        origins = _derive_origins(sales, origins, horizon)

    ledger = BackendEngine(freq=freq, engine=engine).execute(tasks, sales, origins)

    if metrics is None:
        metrics = _DEFAULT_METRICS

    ledger_df = ledger.to_df()
    scores: pd.DataFrame | None
    if ledger_df.empty:
        scores = pd.DataFrame()
    else:
        scores = compute_metrics(ledger_df, metrics)

    return PipelineResult(ledger=ledger, scores=scores, sales=sales)


def run_forecast(
    data_dir: str | Path,
    week: int,
    model_configs: list[dict],
    horizon: int,
    series_filter: list[str] | None = None,
    freq: str = "W",
    engine: Any = None,
) -> Ledger:
    """Forward-looking forecast. Single origin = latest date in sales. No scoring."""
    sales = load_week(data_dir, week)
    tasks = build_tasks(sales, model_configs, horizon, series_filter)

    latest_origin = pd.Timestamp(sales[DS].max())
    origins = [latest_origin]

    return BackendEngine(freq=freq, engine=engine).execute(tasks, sales, origins)
