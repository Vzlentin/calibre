from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from typing import Any, Literal
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from calibre.conformal.runtime import (
    ConformalRuntime,
    PartitionedConformalRuntime,
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
from calibre.core.forecast_task import ForecastTask, ForecastTaskRef, TaskGroups
from calibre.core.metrics import (
    observe_forecast_duration,
    set_conformal_coverage,
    set_conformal_coverage_drift,
)
from calibre.core.seeding import Seed, seed_model_config, set_seed
from calibre.core.tracing import span
from calibre.evaluation.forecast_metrics import compute_row_errors, resolve_actuals
from calibre.execution.io import join_uri, rm
from calibre.execution.ledger import (
    InMemoryLedger,
    InMemoryOrderLedger,
    Ledger,
    OrderLedger,
    StreamingLedger,
    StreamingOrderLedger,
)
from calibre.execution.ray_runtime import RayRuntimeHandle, acquire_ray_runtime
from calibre.execution.threading import cap_threaded_config
from calibre.forecasting.adapter_base import PredictionResult
from calibre.forecasting.adapter_registry import resolve_adapter
from calibre.forecasting.cache import ModelArtifactCache
from calibre.ordering.policy_config import OrderPolicy, apply_order_policy
from calibre.reconciliation.protocols import Reconciler, ReconciliationContext
from calibre.reconciliation.summing import build_summing_matrix
from calibre.storage.state import RUNTIME_PARTITION, ConformalStateStore

logger = logging.getLogger(__name__)


def _finalize_preds(preds: pd.DataFrame, origin: pd.Timestamp, model_name: str) -> pd.DataFrame:
    preds[FORECAST_ORIGIN] = origin
    preds[MODEL_NAME] = model_name
    preds[Y] = np.nan
    extras = [c for c in preds.columns if is_quantile_column(c) and c not in REQUIRED_COLUMNS]
    return preds[REQUIRED_COLUMNS + extras]


def fit_predict_task(
    task: ForecastTask,
    cache: ModelArtifactCache | None = None,
    *,
    collect_fitted_values: bool = False,
) -> PredictionResult:
    adapter = resolve_adapter(task.model_config)
    model_name = task.model_name
    uid = task.unique_id
    origin = task.forecast_origin

    fit_started = time.perf_counter()
    fit_ran, artifact_key = adapter.fit_with_cache(
        task,
        cache,
        collect_fitted_values=collect_fitted_values,
    )
    logger.info(
        "completed adapter fit" if fit_ran else "restored adapter from cache",
        extra={
            "origin": origin,
            "model_name": model_name,
            "unique_id": uid,
            "phase": "fit",
            "cache_hit": not fit_ran,
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
    fitted_values = adapter.fitted_values(task) if collect_fitted_values else None
    return PredictionResult(
        forecast=preds,
        fitted_values=fitted_values,
        artifact_key=artifact_key,
    )


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


def _empty_prediction_result() -> PredictionResult:
    return PredictionResult(forecast=_empty_forecast_frame())


def _process_task_ref(
    ref: ForecastTaskRef,
    origin: pd.Timestamp,
    local_scope: bool,
    collect_fitted_values: bool,
) -> PredictionResult:
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
        task_group=ref.task_group,
    ).materialize()

    history = task.history[task.history[DS] < origin]
    if history.empty:
        return _empty_prediction_result()

    future_x = task.future_x
    if local_scope and future_x is not None and not future_x.empty:
        future_x = future_x[future_x[UNIQUE_ID] == ref.unique_id]

    origin_task = ForecastTask(
        history=history,
        horizon=task.horizon,
        model_config=task.model_config,
        forecast_origin=origin,
        future_x=future_x,
        task_group=task.task_group,
    )
    result = fit_predict_task(origin_task, collect_fitted_values=collect_fitted_values)
    return PredictionResult(
        forecast=_finalize_preds(result.forecast, origin, origin_task.model_name),
        fitted_values=result.fitted_values,
        artifact_key=result.artifact_key,
    )


def _process_global_panel(
    refs: list[ForecastTaskRef],
    model_config: dict,
    origin: pd.Timestamp,
    collect_fitted_values: bool,
) -> PredictionResult:
    """Fit one global adapter for a config over the full multi-SKU panel."""
    histories: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []
    horizon: int | None = None
    task_group: str | None = None

    for ref in refs:
        task = ref.materialize()
        history = task.history[task.history[DS] < origin]
        if not history.empty:
            histories.append(history)
        if task.future_x is not None and not task.future_x.empty:
            future_frames.append(task.future_x)
        if horizon is None:
            horizon = task.horizon
        if task_group is None:
            task_group = task.task_group

    if not histories or horizon is None:
        return _empty_prediction_result()

    panel = pd.concat(histories, ignore_index=True).drop_duplicates([UNIQUE_ID, DS])
    future_x = (
        pd.concat(future_frames, ignore_index=True).drop_duplicates([UNIQUE_ID, DS])
        if future_frames
        else None
    )
    origin_task = ForecastTask(
        history=panel,
        horizon=horizon,
        model_config=dict(model_config),
        forecast_origin=origin,
        future_x=future_x,
        task_group=task_group,
    )
    result = fit_predict_task(origin_task, collect_fitted_values=collect_fitted_values)
    return PredictionResult(
        forecast=_finalize_preds(result.forecast, origin, origin_task.model_name),
        fitted_values=result.fitted_values,
        artifact_key=result.artifact_key,
    )


def _concat_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _empty_forecast_frame()
    return _coerce_forecast_frame_dtypes(pd.concat(non_empty, ignore_index=True))


def _concat_prediction_results(results: list[PredictionResult]) -> PredictionResult:
    if not results:
        return _empty_prediction_result()
    forecasts = _concat_prediction_frames([result.forecast for result in results])
    fitted_parts = [
        result.fitted_values
        for result in results
        if result.fitted_values is not None and not result.fitted_values.empty
    ]
    fitted = pd.concat(fitted_parts, ignore_index=True) if fitted_parts else None
    return PredictionResult(forecast=forecasts, fitted_values=fitted)


@dataclass
class BackendResult:
    """Result returned by BackendEngine.execute()."""

    ledger: Ledger
    order_ledger: OrderLedger | None = None


@dataclass(frozen=True)
class ExecutionOptions:
    freq: str = "W"
    backend: Literal["local", "ray", "auto"] = "auto"
    ray_address: str | None = None
    staging_uri: str | None = None
    ray_threshold: int = 10
    max_concurrency: int | None = None
    cpu_per_task: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"local", "ray", "auto"}:
            raise ValueError("backend must be 'local', 'ray', or 'auto'")
        if self.ray_address is not None and self.staging_uri is None:
            raise ValueError("staging_uri is required when ray_address is set")
        if self.ray_threshold < 1:
            raise ValueError("ray_threshold must be at least 1")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.cpu_per_task is not None and self.cpu_per_task <= 0:
            raise ValueError("cpu_per_task must be positive")


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


@dataclass(frozen=True)
class ReconciliationOptions:
    """Reconciler + hierarchy threaded into the engine (mirrors ConformalOptions).

    A ``None`` reconciler or ``None`` hierarchy makes the Reconcile phase a no-op
    by construction — the path VN2 (``hierarchy=None``) takes, keeping the
    baseline byte-identical (R11).
    """

    reconciler: Reconciler | None = None
    hierarchy: pd.DataFrame | None = None


_DEFAULT_EXECUTION = ExecutionOptions()
_DEFAULT_OUTPUT = LedgerOutputOptions()
_DEFAULT_CONFORMAL = ConformalOptions()
_DEFAULT_RECONCILIATION = ReconciliationOptions()


class BackendEngine:
    def __init__(
        self,
        *,
        execution: ExecutionOptions = _DEFAULT_EXECUTION,
        output: LedgerOutputOptions = _DEFAULT_OUTPUT,
        conformal: ConformalOptions = _DEFAULT_CONFORMAL,
        reconciliation: ReconciliationOptions = _DEFAULT_RECONCILIATION,
        order: OrderPolicy | None = None,
    ) -> None:
        self.execution = execution
        self.output = output
        self.conformal = conformal
        self.reconciler = reconciliation.reconciler
        self.hierarchy = reconciliation.hierarchy
        self._requires_fitted_values = bool(
            reconciliation.hierarchy is not None
            and reconciliation.reconciler is not None
            and reconciliation.reconciler.requires_fitted_values
        )
        self._order_bottom_ids = (
            frozenset(build_summing_matrix(self.hierarchy).bottom_ids)
            if self.hierarchy is not None
            else None
        )
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
        self._ray_runtime: RayRuntimeHandle | None = None
        # Handles stay Any: precise generics live in private ray._private.worker and
        # would force positional-only .remote() calls; not worth it for this slice.
        self._remote_process_task: Any | None = None
        self._remote_process_global_panel: Any | None = None

    def __enter__(self) -> BackendEngine:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def execute(
        self,
        tasks: TaskGroups,
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
            ledger=InMemoryLedger(),
            order_ledger=InMemoryOrderLedger() if self.order_config is not None else None,
        )

    def iter_origins(
        self,
        tasks: TaskGroups,
        actuals: pd.DataFrame,
        origins: list[pd.Timestamp],
    ) -> Iterator[BackendResult]:
        """Yield the cumulative backend result after each completed origin.

        ``tasks`` is a pre-partitioned :class:`TaskGroups`; scope was resolved
        once in ``build_tasks`` and is never re-interpreted here.
        """
        validate_actuals_frame(actuals)
        ledger: Ledger = (
            StreamingLedger(self.streaming_output)
            if self.streaming_output is not None
            else InMemoryLedger()
        )
        if self.initial_ledger is not None and not self.initial_ledger.empty:
            ledger.append(_coerce_forecast_frame_dtypes(self.initial_ledger))
        order_ledger: OrderLedger | None = None
        if self.order_config is not None:
            order_ledger = (
                StreamingOrderLedger(self.streaming_order_output)
                if self.streaming_order_output is not None
                else InMemoryOrderLedger()
            )
        self._restore_conformal_state()
        self._advance_issued_count_from_initial_ledger()
        conformal_runtime = self.conformal_runtime

        try:
            parallel_tasks = [_with_group_tag(task) for task in tasks.local]
            direct_tasks = [_with_group_tag(task) for task in tasks.global_]

            with self._task_staging_prefix() as staging_prefix:
                parallel_refs = self._materialize_task_refs(
                    parallel_tasks,
                    join_uri(staging_prefix, "local"),
                )
                direct_refs = self._materialize_task_refs(
                    direct_tasks,
                    join_uri(staging_prefix, "global"),
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
                        self.run_origin(
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
        if self._ray_runtime is not None:
            self._ray_runtime.release()
        self._ray_runtime = None
        self._remote_process_task = None
        self._remote_process_global_panel = None

    def close(self) -> None:
        self.shutdown_owned_ray()

    @contextmanager
    def _task_staging_prefix(self) -> Iterator[str]:
        if self.execution.ray_address is not None:
            prefix = join_uri(
                self.execution.staging_uri or "",
                f"calibre-backend-{uuid4().hex}",
            )
            try:
                yield prefix
            finally:
                with suppress(Exception):
                    rm(prefix, recursive=True)
            return

        with tempfile.TemporaryDirectory(prefix="calibre-backend-") as temp_dir:
            yield temp_dir

    def run_origin(
        self,
        *,
        ledger: Ledger,
        order_ledger: OrderLedger | None,
        actuals: pd.DataFrame,
        origin: pd.Timestamp,
        conformal_runtime: ConformalRuntime | None,
        parallel_refs: list[ForecastTaskRef],
        direct_refs: list[ForecastTaskRef],
    ) -> None:
        """Run one origin through the ordered per-origin phases.

        The phases run in a fixed order and thread explicit state between them:
        ``ResolveOpen`` (carry-forward prior origins) → ``Predict`` →
        ``Reconcile`` (point → coherent point) → ``Calibrate`` → ``Order`` →
        ``Commit`` (append + final resolve + persist). Each phase is a small
        method that takes/returns the next
        phase's input so it can be driven independently in a test. This is a
        plain method-call sequence, not a dispatch framework (KTD6). A failure
        in any phase is re-raised naming the phase and origin so the fragile
        sequencing the seam exposes is debuggable.
        """
        with self._phase("ResolveOpen", origin):
            self._resolve_open(ledger, actuals, origin, conformal_runtime)
        with self._phase("Predict", origin):
            prediction = self._predict(parallel_refs, direct_refs, origin)
            origin_preds = prediction.forecast
        with self._phase("Reconcile", origin):
            origin_preds = self._reconcile(
                origin_preds,
                ReconciliationContext(fitted_values=prediction.fitted_values),
            )
        with self._phase("Calibrate", origin):
            origin_preds = self._calibrate(origin_preds, conformal_runtime)
        with self._phase("Order", origin):
            self._order(origin_preds, order_ledger)
        with self._phase("Commit", origin):
            self._commit(ledger, origin_preds, actuals, origin, conformal_runtime)

    @contextmanager
    def _phase(self, name: str, origin: pd.Timestamp) -> Iterator[None]:
        """Re-raise any phase failure naming the phase and origin.

        The original exception type is preserved so existing handlers (e.g. the
        API's ``except ValueError``) still match. If the type cannot be rebuilt
        from a single message, fall back to ``RuntimeError`` rather than masking
        the failure with a reconstruction error.
        """
        try:
            yield
        except Exception as exc:
            message = f"{name} phase failed at origin {origin}: {exc}"
            try:
                rewrapped: Exception = type(exc)(message)
            except Exception:
                rewrapped = RuntimeError(message)
            raise rewrapped from exc

    def _resolve_open(
        self,
        ledger: Ledger,
        actuals: pd.DataFrame,
        origin: pd.Timestamp,
        conformal_runtime: ConformalRuntime | None,
    ) -> None:
        """ResolveOpen phase — carry-forward resolution of prior origins.

        Resolve pending ledger rows from PRIOR origins that are now due as of
        this origin (observe + score the conformal runtime, update the ledger).
        This runs BEFORE Predict so the calibrator is updated before this
        origin's intervals are produced. No-op when there is no conformal
        runtime (nothing carries forward to feed back into calibration).
        """
        if conformal_runtime is None:
            return
        self._resolve_due(ledger, actuals, origin, conformal_runtime)

    def _predict(
        self,
        parallel_refs: list[ForecastTaskRef],
        direct_refs: list[ForecastTaskRef],
        origin: pd.Timestamp,
    ) -> PredictionResult:
        """Predict phase — local + global scopes concatenated for this origin.

        Returns an empty forecast frame when there are no tasks.
        """
        local_preds = self._run_local_scope(parallel_refs, origin)
        global_preds = self._run_global_scope(direct_refs, origin)
        return _concat_prediction_results([local_preds, global_preds])

    def _reconcile(
        self,
        origin_preds: pd.DataFrame,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        """Reconcile phase — make point forecasts coherent across the hierarchy.

        Runs BETWEEN Predict and Calibrate (R8): the reconciler rewrites point
        ``y_hat`` in place, then conformal calibrates per node so intervals stay
        per-node marginal around the reconciled point (R9). No-op (returns the
        input unchanged) when no reconciler is configured, the hierarchy is None,
        or the predictions are empty (R3, R11) — the VN2 path, byte-identical by
        construction. This phase is blind to conformal/order state.
        """
        if self.reconciler is None or self.hierarchy is None or origin_preds.empty:
            return origin_preds
        return self.reconciler(
            origin_preds,
            self.hierarchy,
            context,
        )

    def _calibrate(
        self,
        origin_preds: pd.DataFrame,
        conformal_runtime: ConformalRuntime | None,
    ) -> pd.DataFrame:
        """Calibrate phase — apply conformal intervals to this origin's preds.

        No-op (returns the input unchanged) when the runtime is None or the
        predictions are empty.
        """
        if conformal_runtime is None or origin_preds.empty:
            return origin_preds
        return conformal_runtime.apply(origin_preds)

    def _order(
        self,
        origin_preds: pd.DataFrame,
        order_ledger: OrderLedger | None,
    ) -> None:
        """Order phase — derive order decisions and append to the order ledger.

        Skipped when no order policy is configured or the predictions are empty.
        """
        if self.order_config is None or origin_preds.empty:
            return
        assert order_ledger is not None  # guaranteed when order_config is set
        order_frame = self._order_frame(origin_preds)
        if order_frame.empty:
            return
        order_result = apply_order_policy(order_frame, self.order_config)
        order_ledger.append(order_result)

    def _order_frame(self, origin_preds: pd.DataFrame) -> pd.DataFrame:
        if self._order_bottom_ids is None:
            return origin_preds
        return origin_preds[origin_preds[UNIQUE_ID].astype(str).isin(self._order_bottom_ids)].copy()

    def _commit(
        self,
        ledger: Ledger,
        origin_preds: pd.DataFrame,
        actuals: pd.DataFrame,
        origin: pd.Timestamp,
        conformal_runtime: ConformalRuntime | None,
    ) -> None:
        """Commit phase — validate + append, final resolve, persist once.

        Validate this origin's predictions and append them to the ledger, then
        resolve anything now due (observe + score the conformal runtime) and
        persist the conformal state. The mutate-and-persist lives ONLY here, and
        persist is called exactly once per origin.
        """
        if not origin_preds.empty:
            validate_forecast_frame(origin_preds)
            ledger.append(origin_preds)

        self._resolve_due(ledger, actuals, origin, conformal_runtime)
        self._persist_conformal_state(conformal_runtime)

    def _resolve_due(
        self,
        ledger: Ledger,
        actuals: pd.DataFrame,
        origin: pd.Timestamp,
        conformal_runtime: ConformalRuntime | None,
    ) -> None:
        """Resolve ledger rows now due as of ``origin``: observe + score + update.

        Mutate-only: updates the conformal runtime and the ledger but does NOT
        persist conformal state — persistence is consolidated into Commit so it
        fires exactly once per origin.
        """
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

    def _restore_conformal_state(self) -> None:
        if (
            self.run_id is None
            or self.conformal_state_store is None
            or self.conformal_config is None
        ):
            return
        partition_states = self.conformal_state_store.list_for_run(self.run_id)
        if partition_states:
            self.conformal_runtime = SymmetricIntervalRuntime.from_partition_states(
                self.conformal_config,
                partition_states,
            )
            return
        state = self.conformal_state_store.get(self.run_id, RUNTIME_PARTITION)
        if state is None:
            return
        self.conformal_runtime = SymmetricIntervalRuntime.from_state(self.conformal_config, state)

    def _persist_conformal_state(self, conformal_runtime: ConformalRuntime | None) -> None:
        if self.run_id is None or self.conformal_state_store is None or conformal_runtime is None:
            return
        if isinstance(conformal_runtime, PartitionedConformalRuntime):
            partition_states = conformal_runtime.get_partition_states()
            if partition_states:
                for partition, state in partition_states.items():
                    self.conformal_state_store.upsert(
                        self.run_id,
                        str(partition),
                        to_json_safe_state(state),
                    )
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
        self._record_coverage_drift(comparable, conformal_runtime)

    def _record_coverage_drift(
        self,
        resolved: pd.DataFrame,
        conformal_runtime: ConformalRuntime,
    ) -> None:
        drift = conformal_runtime.adaptive_drift()
        if drift is None:
            return
        partitions = (
            sorted(set(resolved[CONFORMAL_PARTITION].dropna().astype(str)))
            if CONFORMAL_PARTITION in resolved.columns
            else ["__global__"]
        )
        if not partitions:
            partitions = ["__global__"]
        for model_name in sorted(set(resolved[MODEL_NAME].dropna().astype(str))):
            for partition in partitions:
                set_conformal_coverage_drift(model_name, partition, drift)

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
            and max_issued_count > runtime.issued_count
        ):
            runtime.restore_issued_count(max_issued_count)

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
            model_config = cap_threaded_config(model_config, self.execution.cpu_per_task)
            task_base = join_uri(base_dir, str(idx))
            ref = ForecastTask(
                history=task.history,
                horizon=task.horizon,
                model_config=model_config,
                forecast_origin=task.forecast_origin,
                future_x=task.future_x,
                task_group=task.task_group,
            ).to_uri(task_base)
            refs.append(ref)
        return refs

    def _run_local_scope(
        self,
        refs: list[ForecastTaskRef],
        origin: pd.Timestamp,
    ) -> PredictionResult:
        if not refs:
            return _empty_prediction_result()
        if self._should_use_ray(len(refs)):
            return self._run_refs_on_ray(refs, origin, local_scope=True)
        return _concat_prediction_results(
            [
                _process_task_ref(
                    ref,
                    origin,
                    local_scope=True,
                    collect_fitted_values=self._requires_fitted_values,
                )
                for ref in refs
            ]
        )

    def _run_global_scope(
        self,
        refs: list[ForecastTaskRef],
        origin: pd.Timestamp,
    ) -> PredictionResult:
        """Run one global multi-series adapter per distinct model config."""
        if not refs:
            return _empty_prediction_result()
        groups = _group_global_refs_by_config(refs)
        if self._should_use_ray(len(groups)):
            return self._run_global_groups_on_ray(groups, origin)
        return _concat_prediction_results(
            [
                _process_global_panel(
                    group_refs,
                    config,
                    origin,
                    collect_fitted_values=self._requires_fitted_values,
                )
                for config, group_refs in groups
            ]
        )

    def _should_use_ray(self, task_count: int) -> bool:
        if self.execution.backend == "local":
            return False
        if self.execution.backend == "ray":
            return True
        return task_count >= self.execution.ray_threshold

    def _ensure_ray(self) -> None:
        import ray

        if self._ray_runtime is not None and ray.is_initialized():
            return
        # Acquiring a fresh runtime invalidates any cached remote-function handles;
        # drop them so the callers rebuild them against the new runtime.
        self._remote_process_task = None
        self._remote_process_global_panel = None
        self._ray_runtime = acquire_ray_runtime(address=self.execution.ray_address)

    def _run_global_groups_on_ray(
        self,
        groups: list[tuple[dict, list[ForecastTaskRef]]],
        origin: pd.Timestamp,
    ) -> PredictionResult:
        self._ensure_ray()
        import ray

        if self._remote_process_global_panel is None:
            remote_fn = ray.remote(_process_global_panel)
            if self.execution.cpu_per_task is not None:
                remote_fn = remote_fn.options(num_cpus=float(self.execution.cpu_per_task))
            self._remote_process_global_panel = remote_fn
        remote_process = self._remote_process_global_panel
        concurrency = self.execution.max_concurrency or len(groups)
        results: list[PredictionResult] = []

        for start in range(0, len(groups), concurrency):
            chunk = groups[start : start + concurrency]
            object_refs = [
                remote_process.remote(
                    group_refs,
                    config,
                    origin,
                    self._requires_fitted_values,
                )
                for config, group_refs in chunk
            ]
            results.extend(ray.get(object_refs))

        return _concat_prediction_results(results)

    def _run_refs_on_ray(
        self,
        refs: list[ForecastTaskRef],
        origin: pd.Timestamp,
        *,
        local_scope: bool,
    ) -> PredictionResult:
        self._ensure_ray()
        import ray

        if self._remote_process_task is None:
            remote_fn = ray.remote(_process_task_ref)
            if self.execution.cpu_per_task is not None:
                remote_fn = remote_fn.options(num_cpus=float(self.execution.cpu_per_task))
            self._remote_process_task = remote_fn
        remote_process = self._remote_process_task
        concurrency = self.execution.max_concurrency or len(refs)
        results: list[PredictionResult] = []

        for start in range(0, len(refs), concurrency):
            chunk = refs[start : start + concurrency]
            object_refs = [
                remote_process.remote(
                    ref,
                    origin,
                    local_scope,
                    self._requires_fitted_values,
                )
                for ref in chunk
            ]
            results.extend(ray.get(object_refs))

        return _concat_prediction_results(results)


def _with_group_tag(task: ForecastTask) -> ForecastTask:
    """Default a task's ``task_group`` to its unique_id for staging/scheduling."""
    return replace(task, task_group=task.task_group or task.unique_id)


def _model_config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def _group_global_refs_by_config(
    refs: list[ForecastTaskRef],
) -> list[tuple[dict, list[ForecastTaskRef]]]:
    grouped: dict[str, tuple[dict, list[ForecastTaskRef]]] = {}
    for ref in refs:
        key = _model_config_key(ref.model_config)
        if key not in grouped:
            grouped[key] = (dict(ref.model_config), [])
        grouped[key][1].append(ref)
    return list(grouped.values())
