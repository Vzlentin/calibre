"""Exercise the revisioned transactional run store."""

from __future__ import annotations

import pandas as pd
import pytest

from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    ActualsSemantics,
    Calendar,
    Panel,
    SessionIdentity,
    TargetSupport,
)
from newcalibre.engine import OriginCommit, OriginIntent
from newcalibre.engine.ports.memory import InMemoryIndexedRunStore
from newcalibre.ledger import LedgerError
from newcalibre.observe import ObserveCycle, ObservedActual

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))


def _session() -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=("a",),
        calendar=CALENDAR,
        horizon=1,
        model_config={"backend": "fixture", "name": "fixture"},
    )


def _panel() -> Panel:
    return Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(["a", "a"], dtype="string"),
                TIMESTAMP: pd.to_datetime(["2026-01-01", "2026-01-02"]),
                OBSERVED_VALUE: pd.Series([1.0, 2.0], dtype="float64"),
            }
        ),
        calendar=CALENDAR,
        target_support=TargetSupport.REAL,
    )


def test_store_opens_revision_bound_delta_and_commits_atomically() -> None:
    session = _session()
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals=_panel(),
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    origin = pd.Timestamp("2026-01-03")

    snapshot = store.open(OriginIntent(session, origin))
    assert snapshot.revision == 0
    assert tuple(record.key for record in snapshot.actuals.records) == (
        ("a", pd.Timestamp("2026-01-01")),
        ("a", pd.Timestamp("2026-01-02")),
    )

    write = OriginCommit(
        session=session,
        origin=origin,
        expected_revision=snapshot.revision,
        observe_cycle=ObserveCycle(
            history_appends=(
                ObservedActual.from_record(record) for record in snapshot.actuals.records
            )
        ),
    )
    receipt = store.commit(write)
    assert receipt.expected_revision == 0
    assert receipt.revision == 1
    assert store.commit(write) == receipt

    reopened = store.open(OriginIntent(session, origin))
    assert reopened.revision == 1
    assert reopened.receipt == receipt
    assert reopened.actuals.records == ()

    with pytest.raises(LedgerError, match="revision"):
        store.commit(
            OriginCommit(
                session=session,
                origin=pd.Timestamp("2026-01-04"),
                expected_revision=0,
            )
        )
