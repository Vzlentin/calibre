from __future__ import annotations

import base64
import logging
import pickle
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import fugue.api as fa
import numpy as np
import pandas as pd

from calibre.conformal.runtime import (
    ConformalRuntime,
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
    build_symmetric_interval_runtime,
    deserialize_calibration_state,
    serialize_calibration_state,
)
from calibre.core.forecast_frame import (
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
    validate_actuals_frame,
    validate_forecast_frame,
)
from calibre.core.forecast_task import ForecastTask
from calibre.core.forecast_task_io import ForecastTaskRef
from calibre.core.metrics import observe_forecast_duration
from calibre.core.seeding import Seed, seed_model_config, set_seed
from calibre.core.tracing import span
from calibre.evaluation.forecast_metrics import compute_row_errors, resolve_actuals
from calibre.execution.ledger import ForecastLedger, OrderLedger
from calibre.forecasting.adapter_registry import get_scope, resolve_adapter
from calibre.ordering.policy_config import OrderPolicyConfig, apply_order_policy
from calibre.storage.state import RUNTIME_PARTITION, ConformalStateStore

_MODEL_CONFIG_PAYLOAD = "model_config_payload"
_HISTORY_URI = "history_uri"
_FUTURE_X_URI = "future_x_uri"
_HAS_FUTURE_X = "has_future_x"
logger = logging.getLogger(__name__)


def _finalize_preds(preds: pd.DataFrame, origin: pd.Timestamp, model_name: str) -> pd.DataFrame:
    preds[FORECAST_ORIGIN] = origin
    preds[MODEL_NAME] = model_name
    preds[Y] = np.nan
    extras = [c for c in preds.columns if is_quantile_column(c) and c not in REQUIRED_COLUMNS]
    return preds[REQUIRED_COLUMNS + extras]


def _coerce_forecast_frame_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    for col in (UNIQUE_ID, MODEL_NAME):
        if col in result.columns:
            result[col] = result[col].astype("object")
    for col in (DS, FORECAST_ORIGIN):
        if col in result.columns:
            result[col] = pd.to_datetime(result[col]).astype("datetime64[ns]")
    for col in (Y, Y_HAT):
        if col in result.columns:
            result[col] = result[col].astype("float64")
    if H in result.columns:
        result[H] = result[H].astype("int64")
    return result


def _encode_model_config(model_config: dict) -> str:
    return base64.b64encode(pickle.dumps(model_config)).decode("ascii")


def _decode_model_config(payload: str) -> dict:
    return pickle.loads(base64.b64decode(payload.encode("ascii")))  # noqa: S301


@dataclass(frozen=True)
class _TaskDispatchRecord:
    unique_id: str
    model_config_payload: str
    horizon: int
    history_uri: str
    future_x_uri: str
    has_future_x: bool

    @classmethod
    def from_task(
        cls, task: ForecastTask, base_uri: str, model_config: dict
    ) -> _TaskDispatchRecord:
        ref = ForecastTask(
            history=task.history,
            horizon=task.horizon,
            model_config=model_config,
            forecast_origin=task.forecast_origin,
            future_x=task.future_x,
        ).to_uri(base_uri)
        return cls(
            unique_id=ref.unique_id,
            model_config_payload=_encode_model_config(ref.model_config),
            horizon=ref.horizon,
            history_uri=ref.history_uri,
            future_x_uri=ref.future_x_uri or "",
            has_future_x=ref.future_x_uri is not None,
        )


def _dispatch_records_to_frame(
    records: list[_TaskDispatchRecord],
    origin: pd.Timestamp,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                UNIQUE_ID: record.unique_id,
                _MODEL_CONFIG_PAYLOAD: record.model_config_payload,
                H: int(record.horizon),
                _HISTORY_URI: record.history_uri,
                _FUTURE_X_URI: record.future_x_uri,
                _HAS_FUTURE_X: bool(record.has_future_x),
                FORECAST_ORIGIN: origin,
            }
            for record in records
        ]
    )


def _process_task_ref_partition(df: pd.DataFrame) -> pd.DataFrame:
    uid = str(df[UNIQUE_ID].iloc[0])
    results: list[pd.DataFrame] = []

    for _, row in df.iterrows():
        origin = pd.Timestamp(row[FORECAST_ORIGIN])
        model_config = _decode_model_config(str(row[_MODEL_CONFIG_PAYLOAD]))
        future_x_uri = str(row[_FUTURE_X_URI]) if bool(row[_HAS_FUTURE_X]) else None
        task = ForecastTaskRef(
            unique_id=uid,
            model_config=model_config,
            horizon=int(row[H]),
            forecast_origin=origin,
            history_uri=str(row[_HISTORY_URI]),
            future_x_uri=future_x_uri,
        ).materialize()

        history = task.history[task.history[DS] < origin]
        if history.empty:
            continue

        task_future_x = task.future_x
        if task_future_x is not None and not task_future_x.empty:
            task_future_x = task_future_x[task_future_x[UNIQUE_ID] == uid]

        origin_task = ForecastTask(
            history=history,
            horizon=task.horizon,
            model_config=model_config,
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
        conformal_runtime: ConformalRuntime | None = None,
        conformal_config: SymmetricIntervalConfig | None = None,
        order_config: OrderPolicyConfig | None = None,
        streaming_output: str | None = None,
        streaming_order_output: str | None = None,
        seed: int | None = None,
        run_id: UUID | None = None,
        conformal_state_store: ConformalStateStore | None = None,
    ) -> None:
        self.freq = freq
        self.metrics = metrics
        self.engine = engine
        if conformal_runtime is not None and conformal_config is not None:
            raise ValueError("Pass either conformal_runtime or conformal_config, not both")
        self.conformal_config = conformal_config
        self.conformal_runtime = (
            conformal_runtime
            if conformal_runtime is not None
            else build_symmetric_interval_runtime(conformal_config)
            if conformal_config is not None
            else None
        )
        self.order_config = order_config
        self.streaming_output = streaming_output
        self.streaming_order_output = streaming_order_output
        self.seed: Seed | None = set_seed(seed) if seed is not None else None
        self.run_id = run_id
        self.conformal_state_store = conformal_state_store

    def execute(
        self,
        tasks: list[ForecastTask],
        actuals: pd.DataFrame,
        origins: list[pd.Timestamp],
    ) -> BackendResult:
        validate_actuals_frame(actuals)
        ledger = ForecastLedger()
        if self.streaming_output is not None:
            ledger.stream_to(self.streaming_output)
        order_ledger = OrderLedger() if self.order_config is not None else None
        if order_ledger is not None and self.streaming_order_output is not None:
            order_ledger.stream_to(self.streaming_order_output)
        self._restore_conformal_state()
        conformal_runtime = self.conformal_runtime

        try:
            # Split tasks once: local (per-series Fugue) vs global (joint direct)
            parallel_tasks: list[ForecastTask] = []
            direct_tasks: list[ForecastTask] = []
            for task in tasks:
                if get_scope(task.model_config) == "local":
                    parallel_tasks.append(task)
                else:
                    direct_tasks.append(task)

            with tempfile.TemporaryDirectory(prefix="calibre-backend-") as temp_dir:
                parallel_records = self._materialize_dispatch_records(
                    parallel_tasks,
                    str(Path(temp_dir) / "local"),
                )
                direct_records = self._materialize_dispatch_records(
                    direct_tasks,
                    str(Path(temp_dir) / "global"),
                )

                for origin in origins:
                    origin_started = time.perf_counter()
                    with span("backtest", origin=str(origin)):
                        if conformal_runtime is not None:
                            self._resolve_ledger(ledger, actuals, origin, conformal_runtime)

                        local_preds = self._run_parallel(parallel_records, origin)
                        global_preds = self._run_direct(direct_records, origin)

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
                            try:
                                validate_forecast_frame(origin_preds)
                            except ValueError as exc:
                                raise ValueError(
                                    f"Invalid forecast frame at origin {origin}: {exc}"
                                ) from exc
                            ledger.append(origin_preds)

                        self._resolve_ledger(ledger, actuals, origin, conformal_runtime)
                    duration = time.perf_counter() - origin_started
                    observe_forecast_duration("mixed", "origin", duration)
                    logger.info(
                        "completed origin",
                        extra={
                            "origin": origin,
                            "phase": "origin",
                            "duration_ms": round(duration * 1000.0, 3),
                        },
                    )
        finally:
            ledger.close()
            if order_ledger is not None:
                order_ledger.close()

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
        self._persist_conformal_state(conformal_runtime)

    def _restore_conformal_state(self) -> None:
        if (
            self.run_id is None
            or self.conformal_state_store is None
            or self.conformal_config is None
        ):
            return
        state = self.conformal_state_store.get(self.run_id, RUNTIME_PARTITION)
        if state is None:
            return
        self.conformal_runtime = SymmetricIntervalRuntime.from_state(
            self.conformal_config,
            serialize_calibration_state(state),
        )

    def _persist_conformal_state(self, conformal_runtime: ConformalRuntime | None) -> None:
        if self.run_id is None or self.conformal_state_store is None or conformal_runtime is None:
            return
        state = deserialize_calibration_state(
            serialize_calibration_state(conformal_runtime.get_diagnostics())
        )
        self.conformal_state_store.upsert(self.run_id, RUNTIME_PARTITION, state)

    def _materialize_dispatch_records(
        self,
        tasks: list[ForecastTask],
        base_dir: str,
    ) -> list[_TaskDispatchRecord]:
        records: list[_TaskDispatchRecord] = []
        for idx, task in enumerate(tasks):
            model_config = {
                **seed_model_config(task.model_config, self.seed),
                "freq": self.freq,
            }
            task_base = str(Path(base_dir) / str(idx))
            records.append(_TaskDispatchRecord.from_task(task, task_base, model_config))
        return records

    def _run_parallel(
        self,
        records: list[_TaskDispatchRecord],
        origin: pd.Timestamp,
    ) -> pd.DataFrame:
        """Run per-series adapters in parallel via Fugue, one partition per uid."""
        if not records:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        task_df = _dispatch_records_to_frame(records, origin)

        schema = (
            f"{UNIQUE_ID}:str,{DS}:datetime,{Y}:double,"
            f"{Y_HAT}:double,{H}:long,"
            f"{FORECAST_ORIGIN}:datetime,{MODEL_NAME}:str"
        )

        result = fa.transform(
            task_df,
            _process_task_ref_partition,
            schema=schema,
            partition={"by": UNIQUE_ID},
            engine=self.engine,
        )

        return _coerce_forecast_frame_dtypes(fa.as_pandas(result))

    def _run_direct(
        self,
        records: list[_TaskDispatchRecord],
        origin: pd.Timestamp,
    ) -> pd.DataFrame:
        """Run global (multi-series) adapters directly, without Fugue partitioning."""
        if not records:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        all_preds: list[pd.DataFrame] = []

        for record in records:
            task = ForecastTaskRef(
                unique_id=record.unique_id,
                model_config=_decode_model_config(record.model_config_payload),
                horizon=record.horizon,
                forecast_origin=origin,
                history_uri=record.history_uri,
                future_x_uri=record.future_x_uri if record.has_future_x else None,
            ).materialize()
            history = task.history[task.history[DS] < origin]
            if history.empty:
                continue

            origin_task = ForecastTask(
                history=history,
                horizon=task.horizon,
                model_config=task.model_config,
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
