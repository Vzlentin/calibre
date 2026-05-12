"""End-to-end pipeline runner for backtesting and forward forecasting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from calibre.conformal import ConformalPolicyConfig
from calibre.contracts.forecast_frame import DS
from calibre.engine.backend import BackendEngine, BackendResult
from calibre.engine.ledger import ForecastLedger, OrderLedger
from calibre.eval.metrics import compute_metrics
from calibre.metrics import mae, rmse, smape, wape
from calibre.order.config import OrderPolicyConfig
from calibre.pipeline.dataset import DatasetBundle
from calibre.pipeline.loading import load_period
from calibre.pipeline.tasks import build_tasks

_DEFAULT_METRICS: list[Callable] = [mae, rmse, smape, wape]


@dataclass(frozen=True)
class PipelineResult:
    ledger: ForecastLedger
    scores: pd.DataFrame
    sales: pd.DataFrame
    order_ledger: OrderLedger | None = None

    def to_parquet(self, path: str | Path) -> None:
        """Export ledger predictions to parquet."""
        self.ledger.to_parquet(str(path))


def _derive_origins(all_dates: list, n: int, horizon: int) -> list[pd.Timestamp]:
    """Return last N origin timestamps that have horizon-worth of actuals after them."""
    if len(all_dates) < n + horizon:
        raise ValueError(
            f"Not enough dates to derive {n} origins with horizon {horizon}: "
            f"need at least {n + horizon} dates, got {len(all_dates)}"
        )
    return [pd.Timestamp(d) for d in all_dates[-(n + horizon) : -horizon]]


def _resolve_bundle(data_dir: DatasetBundle | str | Path, period: int | None) -> DatasetBundle:
    if isinstance(data_dir, DatasetBundle):
        return data_dir
    if period is None:
        raise ValueError("period is required when data_dir is not a DatasetBundle")
    return DatasetBundle(
        history=load_period(data_dir, period),
        future_x=None,
        costs={},
        hierarchy=None,
        censoring=None,
    )


def run_backtest(
    data_dir: DatasetBundle | str | Path,
    period: int | None = None,
    *,
    model_configs: list[dict],
    horizon: int,
    origins: list[pd.Timestamp] | int,
    metrics: list[Callable] | None = None,
    series_filter: list[str] | None = None,
    freq: str = "W",
    engine: Any = None,
    conformal_config: ConformalPolicyConfig | None = None,
    order_config: OrderPolicyConfig | None = None,
) -> PipelineResult:
    """End-to-end backtest pipeline.

    Steps:
    1. Load sales data for the given period.
    2. Build forecast tasks (optionally filtered to a subset of series).
    3. Resolve origins: if an int N is given, derive the last N origin timestamps
       that still have horizon-worth of actuals ahead of them.
    4. Execute the backend engine to produce a Ledger.
    5. Compute aggregate metrics over resolved rows.
    6. Return a PipelineResult.
    """
    bundle = _resolve_bundle(data_dir, period)
    sales = bundle.history
    tasks = build_tasks(sales, model_configs, horizon, series_filter)

    if isinstance(origins, int):
        all_dates = sorted(sales[DS].unique())
        origins = _derive_origins(all_dates, origins, horizon)

    result = BackendEngine(
        freq=freq,
        engine=engine,
        conformal_config=conformal_config,
        order_config=order_config,
    ).execute(tasks, sales, origins)

    ledger = result.ledger

    if metrics is None:
        metrics = _DEFAULT_METRICS

    ledger_df = ledger.to_df()
    interval_bounds = conformal_config.interval_columns if conformal_config is not None else None
    scores = compute_metrics(ledger_df, metrics, interval_bounds=interval_bounds)

    return PipelineResult(
        ledger=ledger, scores=scores, sales=sales, order_ledger=result.order_ledger
    )


def run_forecast(
    data_dir: DatasetBundle | str | Path,
    period: int | None = None,
    *,
    model_configs: list[dict],
    horizon: int,
    series_filter: list[str] | None = None,
    freq: str = "W",
    engine: Any = None,
    conformal_config: ConformalPolicyConfig | None = None,
    order_config: OrderPolicyConfig | None = None,
) -> BackendResult:
    """Forward-looking forecast. Single origin = latest date in sales. No scoring."""
    bundle = _resolve_bundle(data_dir, period)
    sales = bundle.history
    tasks = build_tasks(sales, model_configs, horizon, series_filter)

    latest_origin = sales[DS].max()
    origins = [latest_origin]

    return BackendEngine(
        freq=freq,
        engine=engine,
        conformal_config=conformal_config,
        order_config=order_config,
    ).execute(tasks, sales, origins)
