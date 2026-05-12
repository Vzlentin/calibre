from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import fugue.api as fa
import numpy as np
import pandas as pd

from calibre.conformal.runtime import (
    ConformalPolicyConfig,
    ConformalRuntime,
    build_conformal_runtime,
)
from calibre.contracts.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    is_quantile_column,
)
from calibre.engine.ledger import ForecastLedger, OrderLedger
from calibre.eval.metrics import compute_row_errors, resolve_actuals
from calibre.models.registry import get_scope, resolve_adapter
from calibre.order.config import OrderPolicyConfig, apply_order_policy
from calibre.tasks.forecast_task import ForecastTask


def _finalize_preds(preds: pd.DataFrame, origin: pd.Timestamp, model_name: str) -> pd.DataFrame:
    preds[FORECAST_ORIGIN] = origin
    preds[MODEL_NAME] = model_name
    preds[Y] = np.nan
    extras = [c for c in preds.columns if is_quantile_column(c) and c not in REQUIRED_COLUMNS]
    return preds[REQUIRED_COLUMNS + extras]


@dataclass
class BackendResult:
    """Result returned by BackendEngine.execute()."""

    ledger: ForecastLedger
    order_ledger: OrderLedger | None = None


class BackendEngine:
    def __init__(
        self,
        freq: str = "W",
        metrics: list[Callable] | None = None,
        engine: Any = None,
        conformal_config: ConformalPolicyConfig | None = None,
        order_config: OrderPolicyConfig | None = None,
    ) -> None:
        self.freq = freq
        self.metrics = metrics
        self.engine = engine
        self.conformal_config = conformal_config
        self.order_config = order_config

    def execute(
        self,
        tasks: list[ForecastTask],
        actuals: pd.DataFrame,
        origins: list[pd.Timestamp],
    ) -> BackendResult:
        ledger = ForecastLedger()
        order_ledger = OrderLedger() if self.order_config is not None else None
        conformal_runtime = (
            build_conformal_runtime(self.conformal_config)
            if self.conformal_config is not None
            else None
        )

        # Split tasks once: local (per-series Fugue) vs global (joint direct)
        parallel_tasks: list[ForecastTask] = []
        direct_tasks: list[ForecastTask] = []
        for task in tasks:
            if get_scope(task.model_config) == "local":
                parallel_tasks.append(task)
            else:
                direct_tasks.append(task)

        # Group parallel tasks by uid for efficient Fugue dispatch
        tasks_by_uid: dict[str, list[ForecastTask]] = {}
        for task in parallel_tasks:
            tasks_by_uid.setdefault(task.unique_id, []).append(task)

        for origin in origins:
            if conformal_runtime is not None:
                self._resolve_ledger(ledger, actuals, origin, conformal_runtime)

            local_preds = self._run_parallel(tasks_by_uid, origin)
            global_preds = self._run_direct(direct_tasks, origin)

            origin_preds = pd.concat(
                [df for df in [local_preds, global_preds] if not df.empty],
                ignore_index=True,
            )

            if conformal_runtime is not None and not origin_preds.empty:
                origin_preds = conformal_runtime.apply(origin_preds)

            if self.order_config is not None and not origin_preds.empty:
                order_result = apply_order_policy(origin_preds, self.order_config)
                order_ledger.append(order_result)  # type: ignore[union-attr]

            if not origin_preds.empty:
                ledger.append(origin_preds)

            self._resolve_ledger(ledger, actuals, origin, conformal_runtime)

        return BackendResult(ledger=ledger, order_ledger=order_ledger)

    def _resolve_ledger(
        self,
        ledger: ForecastLedger,
        actuals: pd.DataFrame,
        origin: pd.Timestamp,
        conformal_runtime: ConformalRuntime | None,
    ) -> None:
        current = ledger.to_df()
        if current.empty:
            return

        updated, newly_resolved = resolve_actuals(current, actuals, origin)
        if newly_resolved.empty:
            return

        if conformal_runtime is not None:
            newly_resolved = conformal_runtime.observe(newly_resolved)
            if NONCONFORMITY_SCORE not in updated.columns:
                updated[NONCONFORMITY_SCORE] = np.nan
            updated.loc[newly_resolved.index, NONCONFORMITY_SCORE] = newly_resolved[
                NONCONFORMITY_SCORE
            ]

        scored = compute_row_errors(newly_resolved)
        for col in ("error", "abs_error", "pct_error"):
            if col not in updated.columns:
                updated[col] = np.nan
            updated.loc[scored.index, col] = scored[col]

        ledger.update_resolved(updated)

    def _run_parallel(
        self,
        tasks_by_uid: dict[str, list[ForecastTask]],
        origin: pd.Timestamp,
    ) -> pd.DataFrame:
        """Run per-series adapters in parallel via Fugue, one partition per uid."""
        uids = list(tasks_by_uid.keys())
        if not uids:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        uid_df = pd.DataFrame({UNIQUE_ID: uids})
        freq = self.freq

        def _process_partition(df: pd.DataFrame) -> pd.DataFrame:
            uid = df[UNIQUE_ID].iloc[0]
            results: list[pd.DataFrame] = []

            for task in tasks_by_uid[uid]:
                history = task.history[task.history[DS] < origin]
                if history.empty:
                    continue

                task_future_x = task.future_x
                if task_future_x is not None and not task_future_x.empty:
                    task_future_x = task_future_x[task_future_x[UNIQUE_ID] == uid]

                origin_task = ForecastTask(
                    history=history,
                    horizon=task.horizon,
                    model_config={**task.model_config, "freq": freq},
                    forecast_origin=origin,
                    future_x=task_future_x,
                )

                adapter = resolve_adapter(origin_task.model_config)
                adapter.fit(origin_task)
                preds = adapter.predict(origin_task)

                results.append(_finalize_preds(preds, origin, origin_task.model_name))

            if not results:
                return pd.DataFrame(columns=REQUIRED_COLUMNS)

            return pd.concat(results, ignore_index=True)

        schema = (
            f"{UNIQUE_ID}:str,{DS}:datetime,{Y}:double,"
            f"{Y_HAT}:double,{H}:long,"
            f"{FORECAST_ORIGIN}:datetime,{MODEL_NAME}:str"
        )

        result = fa.transform(
            uid_df,
            _process_partition,
            schema=schema,
            partition={"by": UNIQUE_ID},
            engine=self.engine,
        )

        return pd.DataFrame(result)

    def _run_direct(
        self,
        tasks: list[ForecastTask],
        origin: pd.Timestamp,
    ) -> pd.DataFrame:
        """Run global (multi-series) adapters directly, without Fugue partitioning."""
        if not tasks:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        all_preds: list[pd.DataFrame] = []

        for task in tasks:
            history = task.history[task.history[DS] < origin]
            if history.empty:
                continue

            origin_task = ForecastTask(
                history=history,
                horizon=task.horizon,
                model_config={**task.model_config, "freq": self.freq},
                forecast_origin=origin,
                future_x=task.future_x,
            )

            adapter = resolve_adapter(origin_task.model_config)
            adapter.fit(origin_task)
            preds = adapter.predict(origin_task)

            all_preds.append(_finalize_preds(preds, origin, origin_task.model_name))

        if not all_preds:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        return pd.concat(all_preds, ignore_index=True)
