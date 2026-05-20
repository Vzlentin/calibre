from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import numpy as np
import pandas as pd

from calibre.conformal.runtime import (
    ConformalRuntime,
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
    build_symmetric_interval_runtime,
    parse_state_ref,
    to_json_safe_state,
)
from calibre.core.forecast_frame import (
    CALIBRATION_STATE_REF,
    CONFORMAL_ALPHA,
    CONFORMAL_METHOD,
    CONFORMAL_MODE,
    CONFORMAL_PARTITION,
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
from calibre.core.forecast_task import ForecastTask, ForecastTaskRef
from calibre.core.metrics import observe_forecast_duration, set_conformal_coverage
from calibre.core.seeding import Seed, seed_model_config, set_seed
from calibre.core.tracing import span
from calibre.evaluation.forecast_metrics import compute_row_errors, resolve_actuals
from calibre.execution.ledger import ForecastLedger, OrderLedger
from calibre.forecasting.adapter_registry import get_scope, resolve_adapter
from calibre.ordering.policy_config import OrderPolicyConfig, apply_order_policy
from calibre.storage.state import RUNTIME_PARTITION, ConformalStateStore

logger = logging.getLogger(__name__)
_REMOTE_PROCESS_TASK_REF: Any | None = None


def _finalize_preds(preds: pd.DataFrame, origin: pd.Timestamp, model_name: str) -> pd.DataFrame:
    preds[FORECAST_ORIGIN] = origin
    preds[MODEL_NAME] = model_name
    preds[Y] = np.nan
    extras = [c for c in preds.columns if is_quantile_column(c) and c not in REQUIRED_COLUMNS]
    return preds[REQUIRED_COLUMNS + extras]


def _fit_predict_task(task: ForecastTask) -> pd.DataFrame:
    adapter = resolve_adapter(task.model_config)
    model_name = task.model_name
    uid = task.unique_id
    origin = task.forecast_origin

    fit_started = time.perf_counter()
    adapter.fit(task)
    logger.info(
        "completed adapter fit",
        extra={
            "origin": origin,
            "model_name": model_name,
            "unique_id": uid,
            "phase": "fit",
            "duration_ms": round((time.perf_counter() - fit_started) * 1000.0, 3),
        },
    )

    predict_started = time.perf_counter()
    preds = adapter.predict(task)
    logger.info(
        "completed adapter predict",
        extra={
            "origin": origin,
            "model_name": model_name,
            "unique_id": uid,
            "phase": "predict",
            "duration_ms": round((time.perf_counter() - predict_started) * 1000.0, 3),
        },
    )
    return preds


def _coerce_forecast_frame_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    for col in (
        UNIQUE_ID,
        MODEL_NAME,
        CALIBRATION_STATE_REF,
        CONFORMAL_PARTITION,
        CONFORMAL_METHOD,
        CONFORMAL_MODE,
    ):
        if col in result.columns:
            result[col] = result[col].astype("object")
    for col in (DS, FORECAST_ORIGIN):
        if col in result.columns:
            result[col] = pd.to_datetime(result[col]).astype("datetime64[ns]")
    for col in (Y, Y_HAT, NONCONFORMITY_SCORE, CONFORMAL_ALPHA):
        if col in result.columns:
            result[col] = result[col].astype("float64")
    if H in result.columns:
        result[H] = result[H].astype("int64")
    return result


def _empty_forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def _process_task_ref(
    ref: ForecastTaskRef,
    origin: pd.Timestamp,
    *,
    local_scope: bool,
) -> pd.DataFrame:
    """Materialize and execute one URI-backed task ref.

    This function is intentionally conformal- and order-blind so all mutable
    conformal state stays on the driver.
    """
    task = ForecastTaskRef(
        unique_id=ref.unique_id,
        model_config=dict(ref.model_config),
        horizon=ref.horizon,
        forecast_origin=origin,
        history_uri=ref.history_uri,
        future_x_uri=ref.future_x_uri,
    ).materialize()

    history = task.history[task.history[DS] < origin]
    if history.empty:
        return _empty_forecast_frame()

    future_x = task.future_x
    if local_scope and future_x is not None and not future_x.empty:
        future_x = future_x[future_x[UNIQUE_ID] == ref.unique_id]

    origin_task = ForecastTask(
        history=history,
        horizon=task.horizon,
        model_config=task.model_config,
        forecast_origin=origin,
        future_x=future_x,
    )
    preds = _fit_predict_task(origin_task)
    return _finalize_preds(preds, origin, origin_task.model_name)


def _get_remote_process_task_ref(ray: Any) -> Any:
    global _REMOTE_PROCESS_TASK_REF
    if _REMOTE_PROCESS_TASK_REF is None:
        _REMOTE_PROCESS_TASK_REF = ray.remote(_process_task_ref)
    return _REMOTE_PROCESS_TASK_REF


def _concat_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _empty_forecast_frame()
    return _coerce_forecast_frame_dtypes(pd.concat(non_empty, ignore_index=True))


@dataclass
class BackendResult:
    """Result returned by BackendEngine.execute()."""

    ledger: ForecastLedger
    order_ledger: OrderLedger | None = None


@dataclass(frozen=True)
class ExecutionOptions:
    freq: str = "W"
    backend: Literal["local", "ray", "auto"] = "auto"
    ray_address: str | None = None
    ray_threshold: int = 10
    max_concurrency: int | None = None
    seed: int | None = None
    metrics: list[Callable] | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"local", "ray", "auto"}:
            raise ValueError("backend must be 'local', 'ray', or 'auto'")
        if self.ray_threshold < 1:
            raise ValueError("ray_threshold must be at least 1")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")


@dataclass(frozen=True)
class LedgerOutputOptions:
    forecast_path: str | None = None
    order_path: str | None = None
    streaming: bool = False


@dataclass(frozen=True)
class ConformalOptions:
    runtime: ConformalRuntime | None = None
    config: SymmetricIntervalConfig | None = None
    run_id: UUID | None = None
    state_store: ConformalStateStore | None = None
    initial_ledger: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.runtime is not None and self.config is not None:
            raise ValueError("Pass either conformal runtime or config, not both")


_DEFAULT_EXECUTION = ExecutionOptions()
_DEFAULT_OUTPUT = LedgerOutputOptions()
_DEFAULT_CONFORMAL = ConformalOptions()


class BackendEngine:
    def __init__(
        self,
        *,
        execution: ExecutionOptions = _DEFAULT_EXECUTION,
        output: LedgerOutputOptions = _DEFAULT_OUTPUT,
        conformal: ConformalOptions = _DEFAULT_CONFORMAL,
        order: OrderPolicyConfig | None = None,
    ) -> None:
        self.execution = execution
        self.output = output
        self.conformal = conformal
        self.order_config = order
        self.freq = execution.freq
        self.seed: Seed | None = set_seed(execution.seed) if execution.seed is not None else None
        self.conformal_config = conformal.config
        self.conformal_runtime = (
            conformal.runtime
            if conformal.runtime is not None
            else build_symmetric_interval_runtime(conformal.config)
            if conformal.config is not None
            else None
        )
        self.streaming_output = output.forecast_path if output.streaming else None
        self.streaming_order_output = output.order_path if output.streaming else None
        self.run_id = conformal.run_id
        self.conformal_state_store = conformal.state_store
        self.initial_ledger = (
            conformal.initial_ledger.copy() if conformal.initial_ledger is not None else None
        )

    def execute(
        self,
        tasks: list[ForecastTask],
        actuals: pd.DataFrame,
        origins: list[pd.Timestamp],
    ) -> BackendResult:
        """Run all origins and return the final batch result."""
        result: BackendResult | None = None
        for origin_result in self.iter_origins(tasks, actuals, origins):
            result = origin_result
        if result is not None:
            return result
        return BackendResult(
            ledger=ForecastLedger(),
            order_ledger=OrderLedger() if self.order_config is not None else None,
        )

    def iter_origins(
        self,
        tasks: list[ForecastTask],
        actuals: pd.DataFrame,
        origins: list[pd.Timestamp],
    ) -> Iterator[BackendResult]:
        """Yield the cumulative backend result after each completed origin."""
        validate_actuals_frame(actuals)
        ledger = ForecastLedger()
        if self.streaming_output is not None:
            ledger.stream_to(self.streaming_output)
        if self.initial_ledger is not None and not self.initial_ledger.empty:
            ledger.append(_coerce_forecast_frame_dtypes(self.initial_ledger))
        order_ledger = OrderLedger() if self.order_config is not None else None
        if order_ledger is not None and self.streaming_order_output is not None:
            order_ledger.stream_to(self.streaming_order_output)
        self._restore_conformal_state()
        self._advance_issued_count_from_initial_ledger()
        conformal_runtime = self.conformal_runtime

        try:
            parallel_tasks: list[ForecastTask] = []
            direct_tasks: list[ForecastTask] = []
            for task in tasks:
                if get_scope(task.model_config) == "local":
                    parallel_tasks.append(task)
                else:
                    direct_tasks.append(task)

            with tempfile.TemporaryDirectory(prefix="calibre-backend-") as temp_dir:
                parallel_refs = self._materialize_task_refs(
                    parallel_tasks,
                    str(Path(temp_dir) / "local"),
                )
                direct_refs = self._materialize_task_refs(
                    direct_tasks,
                    str(Path(temp_dir) / "global"),
                )

                completed_origins = self._completed_initial_origins()
                for origin in origins:
                    origin = pd.Timestamp(origin)
                    if origin in completed_origins:
                        logger.info(
                            "skipping resumed origin",
                            extra={"origin": origin, "phase": "resume"},
                        )
                        yield BackendResult(ledger=ledger, order_ledger=order_ledger)
                        continue
                    origin_started = time.perf_counter()
                    with span("backtest", origin=str(origin)):
                        self._execute_origin(
                            ledger=ledger,
                            order_ledger=order_ledger,
                            actuals=actuals,
                            origin=origin,
                            conformal_runtime=conformal_runtime,
                            parallel_refs=parallel_refs,
                            direct_refs=direct_refs,
                        )
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
                    yield BackendResult(ledger=ledger, order_ledger=order_ledger)
        finally:
            ledger.close()
            if order_ledger is not None:
                order_ledger.close()

    def shutdown_owned_ray(self) -> None:
        """Shutdown a local Ray runtime this engine started."""
        ray = getattr(self, "_ray", None)
        if not getattr(self, "_owns_ray_runtime", False) or ray is None:
            return
        ray.shutdown()
        self._owns_ray_runtime = False
        self._ray = None

    def close(self) -> None:
        self.shutdown_owned_ray()

    def _execute_origin(
        self,
        *,
        ledger: ForecastLedger,
        order_ledger: OrderLedger | None,
        actuals: pd.DataFrame,
        origin: pd.Timestamp,
        conformal_runtime: ConformalRuntime | None,
        parallel_refs: list[ForecastTaskRef],
        direct_refs: list[ForecastTaskRef],
    ) -> None:
        if conformal_runtime is not None:
            self._resolve_ledger(ledger, actuals, origin, conformal_runtime)

        local_preds = self._run_local_scope(parallel_refs, origin)
        global_preds = self._run_global_scope(direct_refs, origin)

        origin_preds = _concat_prediction_frames([local_preds, global_preds])

        if conformal_runtime is not None and not origin_preds.empty:
            origin_preds = conformal_runtime.apply(origin_preds)

        if self.order_config is not None and not origin_preds.empty:
            order_result = apply_order_policy(origin_preds, self.order_config)
            order_ledger.append(order_result)  # type: ignore[union-attr]

        if not origin_preds.empty:
            try:
                validate_forecast_frame(origin_preds)
            except ValueError as exc:
                raise ValueError(f"Invalid forecast frame at origin {origin}: {exc}") from exc
            ledger.append(origin_preds)

        self._resolve_ledger(ledger, actuals, origin, conformal_runtime)

    def _resolve_ledger(
        self,
        ledger: ForecastLedger,
        actuals: pd.DataFrame,
        origin: pd.Timestamp,
        conformal_runtime: ConformalRuntime | None,
    ) -> None:
        current = ledger.resolution_frame()
        if current.empty:
            return

        updated, newly_resolved = resolve_actuals(current, actuals, origin)
        if newly_resolved.empty:
            return

        if conformal_runtime is not None:
            newly_resolved = conformal_runtime.observe(newly_resolved)
            self._record_conformal_coverage(newly_resolved, conformal_runtime)
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
        self.conformal_runtime = SymmetricIntervalRuntime.from_state(self.conformal_config, state)

    def _persist_conformal_state(self, conformal_runtime: ConformalRuntime | None) -> None:
        if self.run_id is None or self.conformal_state_store is None or conformal_runtime is None:
            return
        state = to_json_safe_state(conformal_runtime.get_resume_state())
        self.conformal_state_store.upsert(self.run_id, RUNTIME_PARTITION, state)

    def _record_conformal_coverage(
        self,
        resolved: pd.DataFrame,
        conformal_runtime: ConformalRuntime,
    ) -> None:
        lower_col, upper_col = conformal_runtime.interval_columns
        if lower_col not in resolved.columns or upper_col not in resolved.columns:
            return
        comparable = resolved.dropna(subset=[Y, lower_col, upper_col, MODEL_NAME])
        if comparable.empty:
            return
        mode = getattr(conformal_runtime, "mode", "unknown")
        for model_name, group in comparable.groupby(MODEL_NAME, sort=False):
            covered = (group[Y] >= group[lower_col]) & (group[Y] <= group[upper_col])
            set_conformal_coverage(str(model_name), mode, float(covered.mean()))

    def _advance_issued_count_from_initial_ledger(self) -> None:
        """Recover issued-origin accounting from a resumed ledger snapshot."""
        runtime = self.conformal_runtime
        if (
            runtime is None
            or self.initial_ledger is None
            or self.initial_ledger.empty
            or CALIBRATION_STATE_REF not in self.initial_ledger.columns
        ):
            return

        max_issued_count = 0
        for state_ref in self.initial_ledger[CALIBRATION_STATE_REF].dropna().astype(str):
            parsed = parse_state_ref(state_ref) if state_ref else None
            if parsed is None:
                continue
            max_issued_count = max(max_issued_count, parsed.issued_count + 1)

        if (
            isinstance(runtime, SymmetricIntervalRuntime)
            and max_issued_count > runtime._issued_count
        ):
            runtime._issued_count = max_issued_count

    def _completed_initial_origins(self) -> set[pd.Timestamp]:
        if (
            self.initial_ledger is None
            or self.initial_ledger.empty
            or FORECAST_ORIGIN not in self.initial_ledger.columns
        ):
            return set()
        origins = pd.to_datetime(self.initial_ledger[FORECAST_ORIGIN], errors="coerce")
        return {pd.Timestamp(origin) for origin in origins.dropna().unique()}

    def _materialize_task_refs(
        self,
        tasks: list[ForecastTask],
        base_dir: str,
    ) -> list[ForecastTaskRef]:
        refs: list[ForecastTaskRef] = []
        for idx, task in enumerate(tasks):
            model_config = {
                **seed_model_config(task.model_config, self.seed),
                "freq": self.freq,
            }
            task_base = str(Path(base_dir) / str(idx))
            ref = ForecastTask(
                history=task.history,
                horizon=task.horizon,
                model_config=model_config,
                forecast_origin=task.forecast_origin,
                future_x=task.future_x,
            ).to_uri(task_base)
            refs.append(ref)
        return refs

    def _run_local_scope(
        self,
        refs: list[ForecastTaskRef],
        origin: pd.Timestamp,
    ) -> pd.DataFrame:
        if not refs:
            return _empty_forecast_frame()
        if self._should_use_ray(len(refs)):
            return self._run_refs_on_ray(refs, origin, local_scope=True)
        return _concat_prediction_frames(
            [_process_task_ref(ref, origin, local_scope=True) for ref in refs]
        )

    def _run_global_scope(
        self,
        refs: list[ForecastTaskRef],
        origin: pd.Timestamp,
    ) -> pd.DataFrame:
        """Run global multi-series adapters on the driver."""
        if not refs:
            return _empty_forecast_frame()
        return _concat_prediction_frames(
            [_process_task_ref(ref, origin, local_scope=False) for ref in refs]
        )

    def _should_use_ray(self, task_count: int) -> bool:
        if self.execution.backend == "local":
            return False
        if self.execution.backend == "ray":
            return True
        return task_count >= self.execution.ray_threshold

    def _ensure_ray(self) -> Any:
        cached_ray = getattr(self, "_ray", None)
        if cached_ray is not None and cached_ray.is_initialized():
            return cached_ray

        import ray

        if ray.is_initialized():
            self._ray = ray
            return ray

        if self.execution.ray_address is not None:
            ray.init(address=self.execution.ray_address, ignore_reinit_error=True)
            self._owns_ray_runtime = False
        else:
            ray.init(include_dashboard=False, ignore_reinit_error=True)
            self._owns_ray_runtime = True
        self._ray = ray
        return ray

    def _run_refs_on_ray(
        self,
        refs: list[ForecastTaskRef],
        origin: pd.Timestamp,
        *,
        local_scope: bool,
    ) -> pd.DataFrame:
        ray = self._ensure_ray()
        remote_process = _get_remote_process_task_ref(ray)
        concurrency = self.execution.max_concurrency or len(refs)
        frames: list[pd.DataFrame] = []

        for start in range(0, len(refs), concurrency):
            chunk = refs[start : start + concurrency]
            object_refs = [
                remote_process.remote(ref, origin, local_scope=local_scope) for ref in chunk
            ]
            frames.extend(ray.get(object_refs))

        return _concat_prediction_frames(frames)
