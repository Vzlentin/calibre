from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import fugue.api as fa
import numpy as np
import pandas as pd

from calibre.conformal.runtime import ConformalPolicyConfig, ConformalRuntime
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
)
from calibre.engine.ledger import ForecastLedger, OrderLedger
from calibre.engine.scoring import compute_row_errors, resolve_actuals
from calibre.models.registry import resolve_adapter
from calibre.order.config import OrderPolicyConfig, apply_order_policy
from calibre.tasks.forecast_task import ForecastTask


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
            ConformalRuntime(self.conformal_config) if self.conformal_config is not None else None
        )

        tasks_by_uid: dict[str, list[ForecastTask]] = {}
        for task in tasks:
            tasks_by_uid.setdefault(task.unique_id, []).append(task)

        for origin in origins:
            if conformal_runtime is not None:
                self._resolve_ledger(ledger, actuals, origin, conformal_runtime)

            origin_preds = self._run_origin(tasks_by_uid, origin)
            if conformal_runtime is not None and not origin_preds.empty:
                origin_preds = conformal_runtime.apply(origin_preds)

            # Apply order policy AFTER conformal (needs interval columns)
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

    def _run_origin(
        self,
        tasks_by_uid: dict[str, list[ForecastTask]],
        origin: pd.Timestamp,
    ) -> pd.DataFrame:
        uids = list(tasks_by_uid.keys())
        if not uids:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        uid_df = pd.DataFrame({UNIQUE_ID: uids})
        freq = self.freq

        def _process_partition(df: pd.DataFrame) -> pd.DataFrame:
            uid = df[UNIQUE_ID].iloc[0]
            results: list[pd.DataFrame] = []

            for task in tasks_by_uid[uid]:
                history = task.history[task.history["ds"] < origin]
                if history.empty:
                    continue

                origin_task = ForecastTask(
                    unique_id=task.unique_id,
                    history=history,
                    horizon=task.horizon,
                    model_config={**task.model_config, "freq": freq},
                    forecast_origin=origin,
                    future_x=task.future_x,
                )

                adapter = resolve_adapter(origin_task.model_config)
                adapter.fit(origin_task)
                preds = adapter.predict(origin_task)

                preds[UNIQUE_ID] = uid
                preds[FORECAST_ORIGIN] = origin
                preds[MODEL_NAME] = origin_task.model_name
                preds[Y] = np.nan

                results.append(preds[[UNIQUE_ID, DS, Y, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME]])

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
