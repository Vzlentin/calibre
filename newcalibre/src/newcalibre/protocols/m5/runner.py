"""Compose strict M5 inputs through the generic successor engine."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    ActualsSemantics,
    HierarchyIndex,
    Panel,
    Scope,
    SessionIdentity,
)
from newcalibre.engine import (
    DispatchBackend,
    Engine,
    InMemoryIndexedRunStore,
    InMemoryLedgerReader,
    InMemoryPanelSource,
    InProcessDispatch,
    OriginIntent,
    OriginRequest,
    PhaseEvent,
    RayDispatch,
    TimeLoop,
    TimeLoopRequest,
)
from newcalibre.forecasting import resolve_adapter
from newcalibre.protocols.m5.compiler import _CompiledM5Protocol, compile_m5_protocol
from newcalibre.protocols.m5.config import M5ProtocolConfig, load_m5_config
from newcalibre.protocols.m5.loader import M5Dataset, load_m5_dataset
from newcalibre.protocols.m5.scorer import M5Diagnostics, score_m5

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class M5RunResult:
    """Return compact completion facts and closed-ledger M5 diagnostics."""

    session: SessionIdentity
    input_inventory_sha256: str
    forecast_origin_count: int
    commit_count: int
    node_count: int
    expected_row_count: int
    resolved_row_count: int
    eligible_row_count: int
    scored_row_count: int
    pending_row_count: int
    diagnostics: M5Diagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("M5 run session must be a SessionIdentity")
        if (
            not isinstance(self.input_inventory_sha256, str)
            or _SHA256.fullmatch(self.input_inventory_sha256) is None
        ):
            raise TypeError("M5 run input inventory identity must be a SHA-256 digest")
        count_names = (
            "forecast_origin_count",
            "commit_count",
            "node_count",
            "expected_row_count",
            "resolved_row_count",
            "eligible_row_count",
            "scored_row_count",
            "pending_row_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TypeError(f"M5 run {name} must be a non-negative integer")
        if not isinstance(self.diagnostics, M5Diagnostics):
            raise TypeError("M5 run diagnostics must be M5Diagnostics")
        context = self.diagnostics.context
        counts = self.diagnostics.population.counts
        expected_rows = self.forecast_origin_count * self.node_count * context.horizon
        if self.commit_count != self.forecast_origin_count + 1:
            raise ValueError("M5 run must contain one close commit after its forecast origins")
        if self.session.value != context.session_id:
            raise ValueError("M5 run session does not match its diagnostics")
        if context.origin_count != self.forecast_origin_count:
            raise ValueError("M5 run origin count does not match its diagnostics")
        if context.node_count != self.node_count:
            raise ValueError("M5 run node count does not match its diagnostics")
        if context.expected_row_count != self.expected_row_count:
            raise ValueError("M5 run expected row count does not match its diagnostics")
        if self.expected_row_count != expected_rows or counts.total != expected_rows:
            raise ValueError("M5 run row universe does not match its configured dimensions")
        if self.resolved_row_count != counts.resolved:
            raise ValueError("M5 run resolved count does not match its diagnostics")
        if self.eligible_row_count != counts.eligible:
            raise ValueError("M5 run eligible count does not match its diagnostics")
        if self.scored_row_count != counts.scored:
            raise ValueError("M5 run scored count does not match its diagnostics")
        if self.resolved_row_count + self.pending_row_count != self.expected_row_count:
            raise ValueError("M5 run resolved and pending counts do not cover its row universe")


@dataclass(frozen=True, slots=True)
class M5FitPredictResult:
    """Return one bounded-profile Fit/Predict fan-out measurement."""

    concurrency: int
    wall_seconds: float
    dispatch_count: int

    def __post_init__(self) -> None:
        if self.concurrency not in (1, 16):
            raise ValueError("M5 Fit/Predict profile concurrency must equal one or 16")
        if not isinstance(self.wall_seconds, float) or self.wall_seconds <= 0.0:
            raise ValueError("M5 Fit/Predict profile wall duration must be positive")
        if self.dispatch_count != 16:
            raise ValueError("M5 Fit/Predict profile must dispatch all 16 logical shards")


@dataclass(frozen=True, slots=True)
class _M5Composition:
    """Retain the shared prepared M5 runtime graph."""

    config: M5ProtocolConfig
    dataset: M5Dataset
    compiled: _CompiledM5Protocol
    forecast_panel: Panel
    session: SessionIdentity
    store: InMemoryIndexedRunStore


def run_m5(
    config_path: Path,
    *,
    reporter: Callable[[PhaseEvent], None] | None = None,
) -> M5RunResult:
    """Run one strict M5 configuration through the generic time-loop engine."""
    runtime = _prepare_m5(config_path)
    config = runtime.config
    compiled = runtime.compiled
    store = runtime.store
    dispatch = RayDispatch(
        logical_shards=compiled.execution.logical_shards,
        workers=compiled.execution.workers,
        numeric_threads_per_worker=compiled.execution.numeric_threads_per_worker,
        retries=compiled.execution.retries,
    )
    try:
        engine = _engine(runtime, dispatch=dispatch)
        time_loop = TimeLoop(
            engine=engine,
            run_store=store,
            request=TimeLoopRequest(
                session=runtime.session,
                origins=compiled.origins,
                settlement_end=compiled.origins[-1],
                scope=Scope(config.model_scope),
                initial_inventory_positions={},
                actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
            ),
            reporter=reporter,
        ).run()
    finally:
        dispatch.shutdown()
    reader = InMemoryLedgerReader(store)
    diagnostics = score_m5(
        config,
        reader,
        output_dir=_PROJECT_ROOT / compiled.output_dir,
    )
    pending_count = store.pending_observation_count
    return M5RunResult(
        session=runtime.session,
        input_inventory_sha256=runtime.dataset.input_inventory_sha256,
        forecast_origin_count=store.forecast_origin_count,
        commit_count=len(time_loop.receipts),
        node_count=len(compiled.hierarchy.node_labels),
        expected_row_count=diagnostics.context.expected_row_count,
        resolved_row_count=diagnostics.population.counts.resolved,
        eligible_row_count=diagnostics.population.counts.eligible,
        scored_row_count=diagnostics.population.counts.scored,
        pending_row_count=pending_count,
        diagnostics=diagnostics,
    )


def run_m5_fit_predict(config_path: Path, *, concurrency: int) -> M5FitPredictResult:
    """Measure one 1,000-series origin through Fit/Predict only."""
    if concurrency not in (1, 16):
        raise ValueError("M5 Fit/Predict concurrency must equal one or 16")
    runtime = _prepare_m5(config_path)
    config = runtime.config
    if config.population.kind != "digest_rank" or config.population.bottom_count != 1000:
        raise ValueError("M5 Fit/Predict profiling requires the 1,000-series population")
    compiled = runtime.compiled
    dispatch = (
        InProcessDispatch(logical_shards=compiled.execution.logical_shards)
        if concurrency == 1
        else RayDispatch(
            logical_shards=compiled.execution.logical_shards,
            workers=compiled.execution.workers,
            numeric_threads_per_worker=compiled.execution.numeric_threads_per_worker,
            retries=compiled.execution.retries,
        )
    )
    try:
        engine = _engine(runtime, dispatch=dispatch)
        origin = compiled.origins[0]
        snapshot = runtime.store.open(OriginIntent(runtime.session, origin))
        engine.observe(origin, session=runtime.session, snapshot=snapshot)
        request = OriginRequest(
            session=runtime.session,
            origin=origin,
            scope=Scope(config.model_scope),
        )
        started = time.perf_counter()
        fitted = engine.fit(request)
        engine.predict(fitted)
        wall_seconds = time.perf_counter() - started
    finally:
        if isinstance(dispatch, RayDispatch):
            dispatch.shutdown()
    return M5FitPredictResult(
        concurrency=concurrency,
        wall_seconds=wall_seconds,
        dispatch_count=compiled.execution.logical_shards,
    )


def _prepare_m5(config_path: Path) -> _M5Composition:
    """Prepare the canonical shared M5 runtime graph."""
    config = load_m5_config(config_path)
    dataset = load_m5_dataset(_PROJECT_ROOT, config)
    compiled = compile_m5_protocol(dataset, config)
    forecast_panel = _all_node_panel(compiled.panel, hierarchy=compiled.hierarchy)
    session = SessionIdentity.derive(
        tenant=config.dataset,
        series_keys=compiled.hierarchy.node_labels,
        calendar=forecast_panel.calendar,
        horizon=config.horizon,
        model_config=compiled.model_config,
        conformal_config=compiled.conformal_config,
    )
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=forecast_panel.calendar,
        actuals=compiled.panel,
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
        hierarchy=compiled.hierarchy,
    )
    return _M5Composition(config, dataset, compiled, forecast_panel, session, store)


def _engine(runtime: _M5Composition, *, dispatch: DispatchBackend) -> Engine:
    """Compose one engine over the shared prepared M5 runtime graph."""
    return Engine(
        session=runtime.session,
        panel_source=InMemoryPanelSource(runtime.forecast_panel),
        run_store=runtime.store,
        dispatch_backend=dispatch,
        hierarchy=runtime.compiled.hierarchy,
        adapter_resolver=resolve_adapter,
        reconciliation_strategy=runtime.compiled.reconciliation_strategy,
        orderer=None,
    )


def _all_node_panel(panel: Panel, *, hierarchy: HierarchyIndex) -> Panel:
    """Aggregate complete bottom histories into one canonical all-node panel."""
    frame = panel.frame
    node_labels = hierarchy.node_labels
    node_values: dict[str, list[int | float | None]] = {label: [] for label in node_labels}
    timestamps: list[pd.Timestamp] = []
    for timestamp, section in frame.groupby(TIMESTAMP, sort=True, observed=True):
        values = dict(zip(section[SERIES_KEY], section[OBSERVED_VALUE], strict=True))
        aggregated = hierarchy.aggregate(values)
        timestamps.append(cast(pd.Timestamp, timestamp))
        for label in node_labels:
            node_values[label].append(aggregated[label])
    row_count = len(timestamps)
    all_nodes = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(
                [label for label in node_labels for _ in range(row_count)],
                dtype="string",
            ),
            TIMESTAMP: pd.to_datetime(timestamps * len(node_labels)),
            OBSERVED_VALUE: [value for label in node_labels for value in node_values[label]],
        }
    )
    return Panel.from_frame(
        all_nodes,
        calendar=panel.calendar,
        target_support=panel.target_support,
    )


__all__ = ["M5FitPredictResult", "M5RunResult", "run_m5", "run_m5_fit_predict"]
