"""Prove serial and Ray forecast placement preserve committed semantics."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    ActualsSemantics,
    Calendar,
    HierarchyIndex,
    Panel,
    Scope,
    SessionIdentity,
    TargetSupport,
)
from newcalibre.engine import (
    Engine,
    InMemoryIndexedRunStore,
    InMemoryPanelSource,
    InProcessDispatch,
    OriginIntent,
    OriginRequest,
    RayDispatch,
    Spine,
)
from newcalibre.forecasting import resolve_adapter

pytestmark = pytest.mark.tier2

_CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
_SERIES = tuple(f"series-{ordinal:02d}" for ordinal in range(18))
_ORIGINS = (pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06"))


def _panel() -> Panel:
    timestamps = pd.date_range("2026-01-01", periods=10, freq="D")
    frame = pd.DataFrame.from_records(
        [
            {
                SERIES_KEY: key,
                TIMESTAMP: timestamp,
                OBSERVED_VALUE: float(ordinal + day % 3),
            }
            for ordinal, key in enumerate(reversed(_SERIES), start=1)
            for day, timestamp in enumerate(timestamps)
        ]
    ).astype({SERIES_KEY: "string", OBSERVED_VALUE: "float64"})
    return Panel.from_frame(frame, calendar=_CALENDAR, target_support=TargetSupport.REAL)


def _session() -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="dispatch-invariance",
        series_keys=_SERIES,
        calendar=_CALENDAR,
        horizon=3,
        model_config={"backend": "seasonal-naive", "m": 2},
    )


def run_dispatch_world(
    dispatch_factory: Callable[[], InProcessDispatch | RayDispatch],
    *,
    reconstruct: bool,
) -> tuple[object, ...]:
    """Return committed state under one dispatch and reconstruction schedule."""
    panel = _panel()
    session = _session()
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=_CALENDAR,
        actuals=panel,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    dispatch = dispatch_factory()
    engine: Engine | None = None
    receipts = []
    try:
        for ordinal, origin in enumerate(_ORIGINS):
            if reconstruct and ordinal and isinstance(dispatch, RayDispatch):
                dispatch.shutdown()
                dispatch = dispatch_factory()
            if engine is None or reconstruct:
                engine = Engine(
                    session=session,
                    panel_source=InMemoryPanelSource(panel),
                    run_store=store,
                    dispatch_backend=dispatch,
                    hierarchy=HierarchyIndex.flat(panel.series_keys),
                    adapter_resolver=resolve_adapter,
                    orderer=None,
                )
            snapshot = store.open(OriginIntent(session, origin))
            result = Spine(engine).run_origin(
                OriginRequest(session=session, origin=origin, scope=Scope.GLOBAL),
                snapshot=snapshot,
            )
            receipts.append(result.receipt)
    finally:
        if isinstance(dispatch, RayDispatch):
            dispatch.shutdown()
    return (
        store.forecasts,
        store.orders,
        store.settlements,
        store.observed_history,
        tuple(sorted(store.checkpoints.items())),
        tuple(sorted(store.checkpoint_indexes.items())),
        tuple(receipts),
    )


def test_serial_and_ray_are_byte_identical_across_restart() -> None:
    """Preserve frames, checkpoint bytes, receipts, and ledger projections."""
    serial = run_dispatch_world(lambda: InProcessDispatch(logical_shards=16), reconstruct=False)
    serial_restarted = run_dispatch_world(
        lambda: InProcessDispatch(logical_shards=16), reconstruct=True
    )
    ray = run_dispatch_world(RayDispatch, reconstruct=False)
    ray_restarted = run_dispatch_world(RayDispatch, reconstruct=True)

    assert serial_restarted == serial
    assert ray == serial
    assert ray_restarted == serial
