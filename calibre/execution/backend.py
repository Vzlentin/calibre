"""Backtesting execution engine: fit, reconcile, calibrate, order, ledger.

:class:`BackendEngine` orchestrates the per-origin pipeline (model fit →
reconciliation → conformal calibration → ordering → ledger), dispatching work
locally or onto Ray, and is the central entry point for backtest runs.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from typing import Any, Literal, cast
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

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
    CONFORMAL_PARTITION,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    UNIQUE_ID,
    H,
    Y,
    validate_forecast_frame,
)
from calibre.core.forecast_task import (
    ChunkTaskRef,
    ForecastTask,
    ForecastTaskRef,
    TaskGroups,
    stage_local_chunk,
)
from calibre.core.io import join_uri, rm
from calibre.core.metrics import (
    observe_forecast_duration,
    observe_phase_duration,
    set_conformal_coverage,
    set_conformal_coverage_drift,
)
from calibre.core.seeding import Seed, seed_model_config, set_seed
from calibre.core.tracing import span
from calibre.evaluation.forecast_metrics import compute_row_errors
from calibre.execution.actuals import ActualsSource, as_actuals_source
from calibre.execution.ledger import (
    InMemoryLedger,
    InMemoryOrderLedger,
    Ledger,
    OrderLedger,
    StreamingLedger,
    StreamingOrderLedger,
)
from calibre.execution.origin_compute import compute_origin_intervals
from calibre.execution.prediction import (
    _coerce_forecast_frame_dtypes,
    _concat_prediction_results,
    _empty_prediction_result,
    _process_global_panel,
    _process_local_chunk,
)
from calibre.execution.ray_runtime import RayRuntimeHandle, acquire_ray_runtime
from calibre.execution.threading import cap_threaded_config, thread_budget
from calibre.forecasting.adapter_base import PredictionResult
from calibre.ordering.policy_config import OrderPolicy, apply_order_policy
from calibre.reconciliation.hierarchical_intervals import (
    HierarchicalIntervalContext,
    HierarchicalIntervalOptions,
    HierarchicalIntervalPhase,
    NixtlaHierarchicalIntervalPhase,
)
from calibre.reconciliation.protocols import Reconciler, ReconciliationContext
from calibre.reconciliation.summing import HierarchyIndex
from calibre.storage.state import RUNTIME_PARTITION, ConformalStateStore

logger = logging.getLogger(__name__)


@dataclass
class BackendResult:
    """Result returned by BackendEngine.execute()."""

    ledger: Ledger
    order_ledger: OrderLedger | None = None


@dataclass(frozen=True)
class ExecutionOptions:
    """Backend dispatch settings: frequency, local/Ray backend, and chunking."""

    freq: str = "W"
    backend: Literal["local", "ray", "auto"] = "auto"
    ray_address: str | None = None
    staging_uri: str | None = None
    # ``ray_threshold`` counts dispatchable units, not series: chunks for local
    # scope (one chunk == one staged artifact == one Ray task) and config groups
    # for global scope. ``backend="auto"`` uses Ray once that count reaches it.
    ray_threshold: int = 10
    max_concurrency: int | None = None
    cpu_per_task: float | None = None
    # ``chunk_size`` bounds how many same-config local series are staged and
    # fitted per chunk; it caps one worker's materialized histories to one
    # chunk. Local-scope-only — global dispatch is untouched (one panel per
    # config group). The chunk worker still fits each series independently, so
    # chunking never changes per-series math for any backend.
    chunk_size: int = 256
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
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")


@dataclass(frozen=True)
class LedgerOutputOptions:
    """Output sinks for the forecast and order ledgers (optional, streamed or not)."""

    forecast_path: str | None = None
    order_path: str | None = None
    streaming: bool = False


@dataclass(frozen=True)
class ConformalOptions:
    """Conformal calibration inputs: a runtime or a config, plus persistence state."""

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
    """Reconciler + hierarchy index threaded into the engine (mirrors ConformalOptions).

    A ``None`` reconciler or ``None`` hierarchy index makes the Reconcile phase a
    no-op by construction — the path VN2 (``hierarchy_index=None``) takes, keeping
    the baseline byte-identical (R11). The engine consumes the single index run
    preparation built; it never sees the raw hierarchy frame and never builds S.
    """

    reconciler: Reconciler | None = None
    hierarchy_index: HierarchyIndex | None = None


@dataclass(frozen=True)
class HierarchicalIntervalEngineOptions:
    """Fused hierarchy + interval phase threaded independently of Reconciler."""

    phase: HierarchicalIntervalPhase | None = None


_DEFAULT_EXECUTION = ExecutionOptions()
_DEFAULT_OUTPUT = LedgerOutputOptions()
_DEFAULT_CONFORMAL = ConformalOptions()
_DEFAULT_RECONCILIATION = ReconciliationOptions()
_DEFAULT_HIERARCHICAL_INTERVALS = HierarchicalIntervalEngineOptions()


class BackendEngine:
    """Orchestrate the per-origin backtest pipeline across local or Ray backends.

    Threads the model, conformal, reconciliation, and ordering options through
    each rolling origin: fit forecasts, reconcile hierarchies, calibrate
    intervals, place orders, and write the ledger.
    """

    def __init__(
        self,
        *,
        execution: ExecutionOptions = _DEFAULT_EXECUTION,
        output: LedgerOutputOptions = _DEFAULT_OUTPUT,
        conformal: ConformalOptions = _DEFAULT_CONFORMAL,
        reconciliation: ReconciliationOptions = _DEFAULT_RECONCILIATION,
        hierarchical_intervals: HierarchicalIntervalEngineOptions = (
            _DEFAULT_HIERARCHICAL_INTERVALS
        ),
        order: OrderPolicy | None = None,
    ) -> None:
        self.execution = execution
        self.output = output
        self.conformal = conformal
        self.reconciler = reconciliation.reconciler
        self.hierarchy_index = reconciliation.hierarchy_index
        self.hierarchical_interval_phase = hierarchical_intervals.phase
        if self.hierarchical_interval_phase is not None:
            if self.hierarchy_index is None:
                raise ValueError("hierarchical intervals require a hierarchy")
            if conformal.runtime is not None or conformal.config is not None:
                raise ValueError("hierarchical intervals cannot be combined with conformal runtime")
            if reconciliation.reconciler is not None:
                raise ValueError(
                    "hierarchical intervals cannot be combined with point reconciliation"
                )
        self._requires_fitted_values = bool(
            (
                reconciliation.hierarchy_index is not None
                and reconciliation.reconciler is not None
                and reconciliation.reconciler.requires_fitted_values
            )
            or (
                self.hierarchical_interval_phase is not None
                and self.hierarchical_interval_phase.requires_fitted_values
            )
        )
        self._order_bottom_ids = (
            frozenset(self.hierarchy_index.bottom_ids)
            if self.hierarchy_index is not None and order is not None
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
        self._remote_process_local_chunk: Any | None = None
        self._remote_process_global_panel: Any | None = None
        # Fused-interval Ray dispatch state. The remote handle is rebuilt and the
        # hierarchy index re-``ray.put``'d only on a fresh runtime; both persist
        # across origins on a stable runtime so the index is plasma-stored once.
        self._remote_compute_origin_intervals: Any | None = None
        self._hierarchy_index_ref: Any | None = None

    def __enter__(self) -> BackendEngine:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def execute(
        self,
        tasks: TaskGroups,
        actuals: pd.DataFrame | ActualsSource,
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
        actuals: pd.DataFrame | ActualsSource,
        origins: list[pd.Timestamp],
    ) -> Iterator[BackendResult]:
        """Yield the cumulative backend result after each completed origin.

        ``tasks`` is a pre-partitioned :class:`TaskGroups`; scope was resolved
        once in ``build_tasks`` and is never re-interpreted here. ``actuals``
        is either a pre-materialized actuals frame (wrapped in the frame-backed
        source, preserving eager behavior) or an :class:`ActualsSource` that
        resolves actuals lazily; delayed-feedback timing stays in
        ``_resolve_due`` either way.
        """
        actuals_source = as_actuals_source(actuals)
        # Preflight guard only — _restore_conformal_state may REPLACE
        # self.conformal_runtime below, so the loop's runtime is re-captured
        # after restore; mode/protection_period are construction-time facts
        # identical across that swap.
        preflight_runtime = self.conformal_runtime
        if (
            isinstance(preflight_runtime, SymmetricIntervalRuntime)
            and preflight_runtime.mode == "cumulative"
            and preflight_runtime.config.protection_period is not None
        ):
            horizons = [task.horizon for task in tasks]
            protection_period = preflight_runtime.config.protection_period
            if horizons and protection_period > min(horizons):
                # A window can never accumulate protection_period horizons, so
                # the deferral would keep every in-window row pending forever —
                # fail loudly before any ledger or sink is allocated.
                raise ValueError(
                    f"Protection period {protection_period} exceeds available "
                    f"horizon {min(horizons)}: cumulative windows could never "
                    "complete and their rows would stay pending forever"
                )
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
        with span("conformal_state_restore"):
            self._restore_conformal_state()
        self._advance_issued_count_from_initial_ledger()
        conformal_runtime = self.conformal_runtime

        try:
            local_tasks = [_with_group_tag(task) for task in tasks.local]
            direct_tasks = [_with_group_tag(task) for task in tasks.global_]

            with self._task_staging_prefix() as staging_prefix:
                with span("staging_materialize"):
                    chunk_refs = self._materialize_local_chunks(
                        local_tasks,
                        join_uri(staging_prefix, "local"),
                    )
                    direct_refs = self._materialize_task_refs(
                        direct_tasks,
                        join_uri(staging_prefix, "global"),
                    )

                completed_origins = self._completed_initial_origins()
                window = self._fused_parallel_window()
                if window is not None:
                    yield from self._iter_origins_parallel(
                        ledger=ledger,
                        order_ledger=order_ledger,
                        actuals_source=actuals_source,
                        origins=origins,
                        completed_origins=completed_origins,
                        conformal_runtime=conformal_runtime,
                        chunk_refs=chunk_refs,
                        direct_refs=direct_refs,
                        window=window,
                    )
                else:
                    yield from self._iter_origins_serial(
                        ledger=ledger,
                        order_ledger=order_ledger,
                        actuals_source=actuals_source,
                        origins=origins,
                        completed_origins=completed_origins,
                        conformal_runtime=conformal_runtime,
                        chunk_refs=chunk_refs,
                        direct_refs=direct_refs,
                    )
        finally:
            with span("ledger_close"):
                ledger.close()
            if order_ledger is not None:
                with span("order_ledger_close"):
                    order_ledger.close()

    def _fused_parallel_window(self) -> int | None:
        """Effective sliding-window depth N for fused-parallel dispatch, or None.

        The parallel branch is hard-gated: it runs only on the fused path
        (``hierarchical_interval_phase is not None``) with Ray selected and an
        effective window ``N > 1``. ``max_concurrency`` defaults to ``None`` (the
        M5 helper sets none), so coerce explicitly to the D3 default of 2 — never
        Predict's ``or len(...)`` coercion, which would unbound the window and
        break the peak-N resident-frame bound. Returns ``None`` (collapse to the
        serial path) when fused intervals are off, Ray is not selected, or N == 1.
        """
        if self.hierarchical_interval_phase is None:
            return None
        if not self._should_use_ray(1):
            return None
        window = self.execution.max_concurrency or 2
        assert window >= 1
        return window if window > 1 else None

    def _iter_origins_serial(
        self,
        *,
        ledger: Ledger,
        order_ledger: OrderLedger | None,
        actuals_source: ActualsSource,
        origins: list[pd.Timestamp],
        completed_origins: set[pd.Timestamp],
        conformal_runtime: ConformalRuntime | None,
        chunk_refs: list[ChunkTaskRef],
        direct_refs: list[ForecastTaskRef],
    ) -> Iterator[BackendResult]:
        """Drive origins one at a time — the exact inline path (serial default)."""
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
            self.run_origin(
                ledger=ledger,
                order_ledger=order_ledger,
                actuals=actuals_source,
                origin=origin,
                conformal_runtime=conformal_runtime,
                chunk_refs=chunk_refs,
                direct_refs=direct_refs,
            )
            yield self._emit_completed_origin(ledger, order_ledger, origin, origin_started)

    def _iter_origins_parallel(
        self,
        *,
        ledger: Ledger,
        order_ledger: OrderLedger | None,
        actuals_source: ActualsSource,
        origins: list[pd.Timestamp],
        completed_origins: set[pd.Timestamp],
        conformal_runtime: ConformalRuntime | None,
        chunk_refs: list[ChunkTaskRef],
        direct_refs: list[ForecastTaskRef],
        window: int,
    ) -> Iterator[BackendResult]:
        """Drive the fused path as a bounded producer→consumer sliding window.

        Predict runs on the driver per origin (fanning out to Ray as today); each
        origin's interval task is then submitted and at most ``window`` tasks are
        kept in flight. The consumer drains the in-flight deque strictly head-first
        (origins-list order) — never ``as-completed`` — so the serial
        ``ledger.append`` order, the per-origin ``yield``, the ``_phase`` spans,
        the timing log, and crash-resume are all byte-identical to the serial
        path. Peak resident interval frames is ``window``.
        """
        # Ordered in-flight tasks: (origin, ObjectRef, origin_started). The head is
        # always the lowest origins-list index still in flight, so popleft drains in
        # origins-list order regardless of worker completion order.
        in_flight: deque[tuple[pd.Timestamp, Any, float]] = deque()
        producible = iter(origins)
        try:
            while True:
                while len(in_flight) < window:
                    origin = next(producible, None)
                    if origin is None:
                        break
                    origin = pd.Timestamp(origin)
                    if origin in completed_origins:
                        # Resumed origins are NEVER dispatched — yield the resume
                        # result exactly as the serial path does.
                        logger.info(
                            "skipping resumed origin",
                            extra={"origin": origin, "phase": "resume"},
                        )
                        yield BackendResult(ledger=ledger, order_ledger=order_ledger)
                        continue
                    origin_started = time.perf_counter()
                    origin_preds, fitted_context = self._run_origin_predict(
                        ledger,
                        actuals_source,
                        origin,
                        conformal_runtime,
                        chunk_refs,
                        direct_refs,
                    )
                    ref = self._dispatch_origin_intervals(origin_preds, fitted_context)
                    in_flight.append((origin, ref, origin_started))
                if not in_flight:
                    break
                origin, ref, origin_started = in_flight.popleft()
                # Drain-in-order up to the first failure: origins before this one are
                # already committed/streamed. On failure, drop the remaining in-flight
                # refs (dereference abandoned worker frames promptly) before the named
                # re-raise propagates through the _phase context.
                try:
                    with self._phase("HierarchicalIntervals", origin):
                        origin_preds = self._get_origin_intervals(ref)
                        self._finish_origin(
                            ledger, order_ledger, actuals_source, origin, origin_preds, None
                        )
                except Exception:
                    self._drain_in_flight(in_flight)
                    raise
                yield self._emit_completed_origin(ledger, order_ledger, origin, origin_started)
        finally:
            self._drain_in_flight(in_flight)

    @staticmethod
    def _get_origin_intervals(ref: Any) -> pd.DataFrame:
        """Block on a worker interval ref, unwrapping a Ray failure to its cause.

        ``ray.get`` wraps a worker exception in a ``RayTaskError`` whose ``str``
        reproduces the worker traceback and ignores any rewrap message, so it
        would defeat the ``_phase`` origin-naming. Re-raise the underlying cause
        (a plain exception) so ``_phase`` can attribute it to the origin.
        """
        import ray
        from ray.exceptions import RayTaskError

        try:
            return ray.get(ref)
        except RayTaskError as exc:
            cause = exc.cause
            if cause is None:
                raise
            raise cause from exc

    @staticmethod
    def _drain_in_flight(in_flight: deque[tuple[pd.Timestamp, Any, float]]) -> None:
        """Cancel and drop any still-in-flight interval refs (best-effort)."""
        if not in_flight:
            return
        import ray

        while in_flight:
            _origin, ref, _started = in_flight.popleft()
            with suppress(Exception):
                ray.cancel(ref, force=True)

    def _emit_completed_origin(
        self,
        ledger: Ledger,
        order_ledger: OrderLedger | None,
        origin: pd.Timestamp,
        origin_started: float,
    ) -> BackendResult:
        """Record the per-origin timing, log completion, and build the yield result."""
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
        return BackendResult(ledger=ledger, order_ledger=order_ledger)

    def shutdown_owned_ray(self) -> None:
        """Shutdown a local Ray runtime this engine started."""
        if self._ray_runtime is not None:
            self._ray_runtime.release()
        self._ray_runtime = None
        self._remote_process_local_chunk = None
        self._remote_process_global_panel = None
        # The cached index ref is DATA (a ``ray.put`` ObjectRef), not just a
        # handle: a stale ref from a released runtime points at an object the new
        # runtime never plasma-stored, so null it alongside the remote handle.
        self._remote_compute_origin_intervals = None
        self._hierarchy_index_ref = None

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
        actuals: ActualsSource,
        origin: pd.Timestamp,
        conformal_runtime: ConformalRuntime | None,
        chunk_refs: list[ChunkTaskRef],
        direct_refs: list[ForecastTaskRef],
    ) -> None:
        """Run one origin through the ordered per-origin phases.

        The phases run in a fixed order and thread explicit state between them:
        ``ResolveOpen`` (carry-forward prior origins) → ``Predict`` → either
        ``Reconcile`` (point → coherent point) → ``Calibrate`` or the fused
        ``HierarchicalIntervals`` phase → ``Order`` → ``Commit`` (append +
        final resolve + persist). Each phase is a small method that takes/returns the next
        phase's input so it can be driven independently in a test. This is a
        plain method-call sequence, not a dispatch framework (KTD6). A failure
        in any phase is re-raised naming the phase and origin so the fragile
        sequencing the seam exposes is debuggable.
        """
        fused_phase_active = self.hierarchical_interval_phase is not None
        active_conformal_runtime = None if fused_phase_active else conformal_runtime
        origin_preds, fitted_context = self._run_origin_predict(
            ledger, actuals, origin, conformal_runtime, chunk_refs, direct_refs
        )
        if fused_phase_active:
            with self._phase("HierarchicalIntervals", origin):
                origin_preds = self._hierarchical_intervals(origin_preds, fitted_context)
        else:
            with self._phase("Reconcile", origin):
                origin_preds = self._reconcile(
                    origin_preds,
                    ReconciliationContext(fitted_values=fitted_context.fitted_values),
                )
            with self._phase("Calibrate", origin):
                origin_preds = self._calibrate(origin_preds, active_conformal_runtime)
        self._finish_origin(
            ledger, order_ledger, actuals, origin, origin_preds, active_conformal_runtime
        )

    def _run_origin_predict(
        self,
        ledger: Ledger,
        actuals: ActualsSource,
        origin: pd.Timestamp,
        conformal_runtime: ConformalRuntime | None,
        chunk_refs: list[ChunkTaskRef],
        direct_refs: list[ForecastTaskRef],
    ) -> tuple[pd.DataFrame, HierarchicalIntervalContext]:
        """Run ResolveOpen + Predict on the driver, returning preds + fitted sidecar.

        The shared front half of every origin: it carries forward prior origins'
        resolutions (no-op on the fused path, where the conformal runtime
        collapses to ``None``) and produces this origin's point forecasts. The
        returned :class:`HierarchicalIntervalContext` carries the fitted-value
        sidecar both the fused and non-fused tails consume.
        """
        fused_phase_active = self.hierarchical_interval_phase is not None
        active_conformal_runtime = None if fused_phase_active else conformal_runtime
        with self._phase("ResolveOpen", origin):
            self._resolve_open(ledger, actuals, origin, active_conformal_runtime)
        with self._phase("Predict", origin):
            prediction = self._predict(chunk_refs, direct_refs, origin)
        return prediction.forecast, HierarchicalIntervalContext(
            fitted_values=prediction.fitted_values
        )

    def _finish_origin(
        self,
        ledger: Ledger,
        order_ledger: OrderLedger | None,
        actuals: ActualsSource,
        origin: pd.Timestamp,
        origin_preds: pd.DataFrame,
        active_conformal_runtime: ConformalRuntime | None,
    ) -> None:
        """Run the shared serial tail Order + Commit for one origin.

        This is the tail both the fused and non-fused branches reach (it sits
        outside the branch in ``run_origin``); extracting it keeps the
        append-then-resolve order identical across serial and parallel dispatch.
        """
        with self._phase("Order", origin):
            self._order(origin_preds, order_ledger)
        with self._phase("Commit", origin):
            self._commit(ledger, origin_preds, actuals, origin, active_conformal_runtime)

    @contextmanager
    def _phase(self, name: str, origin: pd.Timestamp) -> Iterator[None]:
        """Re-raise any phase failure naming the phase and origin.

        The original exception type is preserved so existing handlers (e.g. the
        API's ``except ValueError``) still match. If the type cannot be rebuilt
        from a single message, fall back to ``RuntimeError`` rather than masking
        the failure with a reconstruction error.

        Timing lives in ``finally`` so a failing phase is still attributed — the
        failure path is exactly where attribution matters most. The
        rewrap-and-raise path is otherwise byte-identical to before.
        """
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            message = f"{name} phase failed at origin {origin}: {exc}"
            try:
                rewrapped: Exception = type(exc)(message)
            except Exception:
                rewrapped = RuntimeError(message)
            raise rewrapped from exc
        finally:
            duration = time.perf_counter() - started
            observe_phase_duration(name, duration)
            logger.info(
                "completed phase",
                extra={
                    "phase": name,
                    "origin": origin,
                    "duration_ms": round(duration * 1000.0, 3),
                },
            )

    def _resolve_open(
        self,
        ledger: Ledger,
        actuals: ActualsSource,
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
        chunk_refs: list[ChunkTaskRef],
        direct_refs: list[ForecastTaskRef],
        origin: pd.Timestamp,
    ) -> PredictionResult:
        """Predict phase — local + global scopes concatenated for this origin.

        Returns an empty forecast frame when there are no tasks.
        """
        local_preds = self._run_local_scope(chunk_refs, origin)
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
        if self.reconciler is None or self.hierarchy_index is None or origin_preds.empty:
            return origin_preds
        return self.reconciler(
            origin_preds,
            self.hierarchy_index,
            context,
        )

    def _hierarchical_intervals(
        self,
        origin_preds: pd.DataFrame,
        context: HierarchicalIntervalContext,
    ) -> pd.DataFrame:
        """Fused hierarchical conformal phase — replace Reconcile + Calibrate.

        The ``apply`` runs under the same ``threadpool_limits`` budget the Ray
        worker uses (:func:`~calibre.execution.origin_compute.compute_origin_intervals`),
        so the BLAS thread count is held constant serial-vs-parallel — a
        thread-count asymmetry would break byte-identity on dense-BLAS strategies.
        """
        if self.hierarchical_interval_phase is None or origin_preds.empty:
            return origin_preds
        assert self.hierarchy_index is not None  # checked at construction
        with threadpool_limits(limits=thread_budget(self.execution.cpu_per_task)):
            return self.hierarchical_interval_phase.apply(
                origin_preds, self.hierarchy_index, context
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
        actuals: ActualsSource,
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
            with span("ledger_append", origin=origin):
                ledger.append(origin_preds)

        self._resolve_due(ledger, actuals, origin, conformal_runtime)
        self._persist_conformal_state(conformal_runtime)

    def _resolve_due(
        self,
        ledger: Ledger,
        actuals: ActualsSource,
        origin: pd.Timestamp,
        conformal_runtime: ConformalRuntime | None,
    ) -> None:
        """Resolve ledger rows now due as of ``origin``: observe + score + update.

        Mutate-only: updates the conformal runtime and the ledger but does NOT
        persist conformal state — persistence is consolidated into Commit so it
        fires exactly once per origin.
        """
        # Span name is intentionally "ledger_resolution_frame" even though the
        # method is now due_frame(): the name is load-bearing for M5 log analysis
        # (it keys the ResolveOpen/Commit timing breakdown) and is held constant
        # across the keyed-ledger refactor.
        with span("ledger_resolution_frame", origin=origin):
            current = ledger.due_frame(origin)

        # The actuals_lookup span fires on every Commit-reached origin (even when
        # the due frame is empty): resolve on an empty frame is cheap and the
        # span's emission under this exact name is a load-bearing observability
        # contract. The meaningful early return is on empty newly_resolved.
        with span("actuals_lookup", origin=origin):
            updated, newly_resolved = actuals.resolve(current, origin)

        # Cumulative engine-internal conformal scores a window only at its
        # terminal horizon and only when the whole window is present. When a
        # window's horizons resolve across multiple origins, resolving its early
        # rows per-origin would strand them: the runtime skips the still-partial
        # window (no side effect) and apply_resolutions then removes them from
        # the open set, so the window can never be presented complete and never
        # scores. Defer incomplete cumulative windows here — their rows stay
        # unresolved (y NaN) in the open set until the terminal horizon resolves,
        # at which point the whole window is due together. Strictly gated to the
        # SymmetricIntervalRuntime cumulative path; perhorizon, runtime-None, and
        # CumulativeRiskRuntime paths are untouched.
        if (
            isinstance(conformal_runtime, SymmetricIntervalRuntime)
            and conformal_runtime.mode == "cumulative"
            and conformal_runtime.config.protection_period is not None
        ):
            updated, newly_resolved = self._defer_incomplete_cumulative_windows(
                updated,
                newly_resolved,
                int(conformal_runtime.config.protection_period),
            )

        if newly_resolved.empty:
            return

        with span("resolve_scoring", origin=origin):
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

        # Span name is intentionally "ledger_update_resolved" even though the
        # method is now apply_resolutions(): kept constant for M5 log analysis
        # (it is not test-asserted, so a rename here would pass CI silently —
        # the constant name is the contract).
        with span("ledger_update_resolved", origin=origin):
            ledger.apply_resolutions(updated)

    @staticmethod
    def _defer_incomplete_cumulative_windows(
        updated: pd.DataFrame,
        newly_resolved: pd.DataFrame,
        protection_period: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Hold back incomplete cumulative windows from this resolution batch.

        Mirrors the runtime's completeness rule (``calibre/conformal/runtime.py``
        ``_observe_cumulative``): within each ``(unique_id, model_name,
        forecast_origin)`` window, only horizons ``h <= protection_period`` are
        scored, and the window is observable only once all of them are present
        (count ``>= protection_period``) with no NaN ``y``. Windows that fail
        that test have their ``h <= protection_period`` rows reverted to
        unresolved: dropped from ``newly_resolved`` and reset to ``y`` NaN in
        ``updated`` so the ledger keeps them open (y NaN) for a later origin.

        Rows with ``h > protection_period`` sit outside every scoring window and
        are never deferred — deferring them would strand them in the open set
        forever (no window completes them), so they resolve per-origin exactly as
        today. The mirror assumes contiguous ``h = 1..horizon``, which the task
        builder guarantees.
        """
        if newly_resolved.empty:
            return updated, newly_resolved

        window_rows = newly_resolved[newly_resolved[H] <= protection_period]
        if window_rows.empty:
            return updated, newly_resolved

        group_cols = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]
        # Completeness is judged against the window as seen in `updated`, which
        # carries every present horizon for this window (including any already
        # filled at a prior call), not only this batch's new rows. One grouped
        # pass over the in-window slice keeps this O(batch); a per-group rescan
        # of `updated` would be quadratic in the resolution batch.
        in_window = updated[updated[H] <= protection_period]
        stats = in_window.groupby(group_cols, sort=False)[Y].agg(["size", "count"])
        complete = stats[(stats["size"] >= protection_period) & (stats["count"] == stats["size"])]

        window_keys = pd.MultiIndex.from_frame(window_rows[group_cols])
        defer_index = window_rows.index[~window_keys.isin(complete.index)]
        if defer_index.empty:
            return updated, newly_resolved

        updated.loc[defer_index, Y] = np.nan
        newly_resolved = newly_resolved.drop(index=defer_index)
        return updated, newly_resolved

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

    def _resolved_model_config(self, task: ForecastTask) -> dict:
        model_config = {
            **seed_model_config(task.model_config, self.seed),
            "freq": self.freq,
        }
        return cap_threaded_config(model_config, self.execution.cpu_per_task)

    def _materialize_task_refs(
        self,
        tasks: list[ForecastTask],
        base_dir: str,
    ) -> list[ForecastTaskRef]:
        refs: list[ForecastTaskRef] = []
        for idx, task in enumerate(tasks):
            task_base = join_uri(base_dir, str(idx))
            ref = ForecastTask(
                history=task.history,
                horizon=task.horizon,
                model_config=self._resolved_model_config(task),
                forecast_origin=task.forecast_origin,
                future_x=task.future_x,
                task_group=task.task_group,
            ).to_uri(task_base)
            refs.append(ref)
        return refs

    def _materialize_local_chunks(
        self,
        tasks: list[ForecastTask],
        base_dir: str,
    ) -> list[ChunkTaskRef]:
        """Group local tasks by resolved config and stage O(chunks) artifacts.

        Tasks are grouped by their resolved (model config, horizon) — so
        distinct configs or horizons never share a chunk and duplicate
        (uid, config, horizon) tasks collapse to one — then each group is
        split into ``chunk_size`` slices.
        Each slice is staged as one panel-history parquet plus, when any series
        carries exogenous features, one uid-filtered future_x parquet — so total
        staging is O(chunks), not O(series).
        """
        resolved = [
            ForecastTask(
                history=task.history,
                horizon=task.horizon,
                model_config=self._resolved_model_config(task),
                forecast_origin=task.forecast_origin,
                future_x=task.future_x,
                task_group=task.task_group,
            )
            for task in tasks
        ]
        groups = _group_local_tasks_by_config(resolved)
        chunk_size = self.execution.chunk_size
        chunk_refs: list[ChunkTaskRef] = []
        for group_idx, (model_config, group_tasks) in enumerate(groups):
            for chunk_idx, start in enumerate(range(0, len(group_tasks), chunk_size)):
                chunk_tasks = group_tasks[start : start + chunk_size]
                chunk_base = join_uri(base_dir, str(group_idx), str(chunk_idx))
                chunk_refs.append(
                    stage_local_chunk(
                        chunk_tasks,
                        chunk_base,
                        horizon=chunk_tasks[0].horizon,
                        model_config=model_config,
                        task_group=chunk_tasks[0].task_group,
                    )
                )
        return chunk_refs

    def _run_local_scope(
        self,
        chunk_refs: list[ChunkTaskRef],
        origin: pd.Timestamp,
    ) -> PredictionResult:
        if not chunk_refs:
            return _empty_prediction_result()
        if self._should_use_ray(len(chunk_refs)):
            return self._run_chunks_on_ray(chunk_refs, origin)
        return _concat_prediction_results(
            [
                _process_local_chunk(
                    chunk_ref,
                    origin,
                    collect_fitted_values=self._requires_fitted_values,
                )
                for chunk_ref in chunk_refs
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
        # Acquiring a fresh runtime invalidates any cached remote-function handles
        # and the ``ray.put`` index ref; drop them so the callers rebuild them
        # against the new runtime. These resets stay on the re-acquisition path
        # (after the warm early-return) so the index ref and handle persist across
        # origins on a stable runtime — re-putting the index every origin would be
        # needless overhead.
        self._remote_process_local_chunk = None
        self._remote_process_global_panel = None
        self._remote_compute_origin_intervals = None
        self._hierarchy_index_ref = None
        self._ray_runtime = acquire_ray_runtime(address=self.execution.ray_address)

    def _build_remote(self, fn: Callable[..., Any]) -> Any:
        """Build a ``ray.remote`` handle, applying ``num_cpus`` admission when set.

        ``cpu_per_task`` maps to ``num_cpus`` only when configured; otherwise
        Ray's auto-detected CPU count governs admission. Shared by the Predict
        runners and the fused-interval dispatch so the admission rule lives once.
        """
        import ray

        remote_fn = ray.remote(fn)
        if self.execution.cpu_per_task is not None:
            remote_fn = remote_fn.options(num_cpus=float(self.execution.cpu_per_task))
        return remote_fn

    def _run_global_groups_on_ray(
        self,
        groups: list[tuple[dict, list[ForecastTaskRef]]],
        origin: pd.Timestamp,
    ) -> PredictionResult:
        self._ensure_ray()
        import ray

        if self._remote_process_global_panel is None:
            self._remote_process_global_panel = self._build_remote(_process_global_panel)
        remote_process = self._remote_process_global_panel
        concurrency = self.execution.max_concurrency or len(groups)
        results: list[PredictionResult] = []

        for start in range(0, len(groups), concurrency):
            batch = groups[start : start + concurrency]
            object_refs = [
                remote_process.remote(
                    group_refs,
                    config,
                    origin,
                    self._requires_fitted_values,
                )
                for config, group_refs in batch
            ]
            results.extend(ray.get(object_refs))

        return _concat_prediction_results(results)

    def _run_chunks_on_ray(
        self,
        chunk_refs: list[ChunkTaskRef],
        origin: pd.Timestamp,
    ) -> PredictionResult:
        self._ensure_ray()
        import ray

        if self._remote_process_local_chunk is None:
            self._remote_process_local_chunk = self._build_remote(_process_local_chunk)
        remote_process = self._remote_process_local_chunk
        concurrency = self.execution.max_concurrency or len(chunk_refs)
        results: list[PredictionResult] = []

        for start in range(0, len(chunk_refs), concurrency):
            batch = chunk_refs[start : start + concurrency]
            object_refs = [
                remote_process.remote(
                    chunk_ref,
                    origin,
                    self._requires_fitted_values,
                )
                for chunk_ref in batch
            ]
            results.extend(ray.get(object_refs))

        return _concat_prediction_results(results)

    def _dispatch_origin_intervals(
        self,
        origin_preds: pd.DataFrame,
        context: HierarchicalIntervalContext,
    ) -> Any:
        """Submit one origin's fused interval task to Ray and return its ObjectRef.

        Lazily builds and caches the ``ray.remote`` handle and a single
        ``ray.put`` of the run-constant hierarchy index (reused by every task on
        a stable runtime); applies ``num_cpus`` admission only when
        ``cpu_per_task`` is set, mirroring the Predict runners. Dispatch is one
        task per origin — it never traverses the ``max_concurrency`` batching
        loop; ``max_concurrency`` is reused only as the window depth in
        :meth:`iter_origins`.
        """
        assert self.hierarchical_interval_phase is not None  # gated by the caller
        assert self.hierarchy_index is not None  # checked at construction
        self._ensure_ray()
        import ray

        if self._remote_compute_origin_intervals is None:
            self._remote_compute_origin_intervals = self._build_remote(compute_origin_intervals)
        if self._hierarchy_index_ref is None:
            self._hierarchy_index_ref = ray.put(self.hierarchy_index)
        options: HierarchicalIntervalOptions = cast(
            NixtlaHierarchicalIntervalPhase, self.hierarchical_interval_phase
        ).options
        return self._remote_compute_origin_intervals.remote(
            origin_preds,
            self._hierarchy_index_ref,
            context,
            options,
            self.execution.cpu_per_task,
        )


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


def _group_local_tasks_by_config(
    tasks: list[ForecastTask],
) -> list[tuple[dict, list[ForecastTask]]]:
    """Group local tasks by resolved (config, horizon) so they never co-chunk.

    Mirrors :func:`_group_global_refs_by_config` but over un-staged tasks (the
    grouping decides chunk membership *before* staging). Insertion order is
    preserved within and across groups, keeping chunk assignment deterministic.

    Duplicate (unique_id, config, horizon) tasks collapse to their first
    occurrence — the local mirror of ``build_tasks``' global dedup. The chunk
    worker slices the staged panel per uid, so a duplicate's concatenated
    history would otherwise read back as one series with every row doubled.
    """
    grouped: dict[tuple[str, int], tuple[dict, list[ForecastTask]]] = {}
    seen: set[tuple[str, str, int]] = set()
    for task in tasks:
        config_key = _model_config_key(task.model_config)
        task_key = (task.unique_id, config_key, task.horizon)
        if task_key in seen:
            continue
        seen.add(task_key)
        group_key = (config_key, task.horizon)
        if group_key not in grouped:
            grouped[group_key] = (dict(task.model_config), [])
        grouped[group_key][1].append(task)
    return list(grouped.values())
