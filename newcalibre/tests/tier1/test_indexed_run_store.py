"""Exercise the revisioned transactional run store."""

from __future__ import annotations

import pandas as pd
import pytest

from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    OBSERVED_VALUE,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    TIMESTAMP,
    TOTAL_NODE_LABEL,
    ActualsSemantics,
    Calendar,
    HierarchyIndex,
    Panel,
    SessionIdentity,
    TargetSupport,
)
from newcalibre.engine import (
    ActualsCommit,
    ActualsCommitKey,
    ActualsIntent,
    ForecastWrite,
    LedgerColumn,
    LedgerSelection,
    OriginCommit,
    OriginIntent,
)
from newcalibre.engine.ports.memory import InMemoryIndexedRunStore, InMemoryLedgerReader
from newcalibre.ledger import LedgerError
from newcalibre.observe import (
    ActualRecord,
    ActualsSubmission,
    ObserveCycle,
    ObservedActual,
    ObserveLoop,
)

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))


def _session(series_keys: tuple[str, ...] = ("a",)) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=series_keys,
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


def _forecast_write(
    origin: pd.Timestamp,
    *,
    point: float = 7.0,
    series_keys: tuple[str, ...] = ("a",),
) -> ForecastWrite:
    keys = tuple((series_key, origin, 1, "fixture") for series_key in series_keys)
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(series_keys, dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime([origin] * len(series_keys)),
            ACTUAL_VALUE: pd.Series([None] * len(series_keys), dtype="float64"),
            POINT_FORECAST: pd.Series([point] * len(series_keys), dtype="float64"),
            HORIZON_STEP: pd.Series([1] * len(series_keys), dtype="int64"),
            ORIGIN: pd.to_datetime([origin] * len(series_keys)),
            MODEL_NAME: pd.Series(["fixture"] * len(series_keys), dtype="string"),
        }
    )
    return ForecastWrite(frame, {key: {} for key in keys})


def _observe(snapshot, *, hierarchy: HierarchyIndex | None = None) -> ObserveCycle:
    loop = ObserveLoop(
        hierarchy=hierarchy or HierarchyIndex.flat(snapshot.session.series_keys),
        observed_history=snapshot.observed_history,
        pending_observations=snapshot.pending_observations,
    )
    loop.accept(snapshot.actuals)
    return loop.cycle(snapshot.origin)


def _checkpoint_index(checkpoint_key: str) -> bytes:
    identity = "0" * 64
    return (
        f'{{"checkpoint_key":"{checkpoint_key}","cursor":'
        f'{{"panel_identity":"{identity}","series_start":0,'
        f'"series_stop":1,"time_bound":1}},'
        f'"schema":"newcalibre.forecast-checkpoint-index/v1"}}'
    ).encode()


def _durable_families(store: InMemoryIndexedRunStore) -> tuple[object, ...]:
    """Snapshot every externally visible durable store family."""
    return (
        store.revision,
        store.observed_history,
        store.pending_observations,
        store.forecasts,
        store.orders,
        store.settlements,
        store.observation_resolutions,
        store.observe_annotations,
        dict(store.states),
        dict(store.checkpoints),
        dict(store.checkpoint_indexes),
        store.resume_marker,
        dict(store.receipts),
        dict(store.settlement_receipts),
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
    assert snapshot.revision == 1
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
    assert receipt.expected_revision == 1
    assert receipt.revision == 2
    assert store.commit(write) == receipt

    reopened = store.open(OriginIntent(session, origin))
    assert reopened.revision == 2
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


def test_state_footprint_tracks_committed_state_rows_and_bytes() -> None:
    """Report the conformal state level without materializing a second mapping."""
    session = _session()
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals_semantics=ActualsSemantics.DEMAND,
    )

    assert store.state_footprint == (0, 0)

    store.commit(
        OriginCommit(
            session=session,
            origin=pd.Timestamp("2026-01-03"),
            expected_revision=store.revision,
            state_updates={"one": b"abc", "two": b"de"},
        )
    )

    assert store.state_footprint == (2, 5)

    store.commit(
        OriginCommit(
            session=session,
            origin=pd.Timestamp("2026-01-04"),
            expected_revision=store.revision,
            state_updates={"two": b"defg", "three": b"h"},
        )
    )

    assert store.state_footprint == (3, 8)
    assert store.state_footprint == (
        len(store.states),
        sum(len(value) for value in store.states.values()),
    )


def test_natural_keys_replay_stored_receipts_and_reject_stale_revisions() -> None:
    """Replay the stored receipt at either key kind and still reject stale writes."""
    session = _session()
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    origin = pd.Timestamp("2026-01-03")
    origin_write = OriginCommit(
        session=session,
        origin=origin,
        expected_revision=store.revision,
        state_updates={"partition": b"one"},
    )
    origin_receipt = store.commit(origin_write)

    assert store.commit(origin_write) is origin_receipt
    assert store.receipt(origin) is origin_receipt

    submission = ActualsSubmission((ActualRecord("a", pd.Timestamp("2026-01-02"), 2.0),))
    snapshot = store.open(ActualsIntent(session, submission))
    actuals_write = ActualsCommit(
        session=session,
        origin=snapshot.origin,
        expected_revision=snapshot.revision,
        actual_keys=tuple(record.key for record in submission.records),
        observe_cycle=_observe(snapshot),
    )
    actuals_receipt = store.commit(actuals_write)

    assert store.commit(actuals_write) is actuals_receipt
    assert store.receipt(ActualsCommitKey(actuals_write.actual_keys)) is actuals_receipt

    with pytest.raises(LedgerError, match="revision is stale"):
        store.commit(
            OriginCommit(
                session=session,
                origin=pd.Timestamp("2026-01-04"),
                expected_revision=origin_write.expected_revision,
                state_updates={"partition": b"changed"},
            )
        )


def test_actuals_open_reads_only_affected_hierarchy_memberships() -> None:
    """Address submitted bottoms and their aggregates without target-wide scans."""
    hierarchy = HierarchyIndex.flat(("a", "b"))
    session = _session(hierarchy.node_labels)
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals_semantics=ActualsSemantics.DEMAND,
        hierarchy=hierarchy,
    )
    target = pd.Timestamp("2026-01-02")
    store.commit(
        OriginCommit(
            session=session,
            origin=target,
            expected_revision=store.revision,
            forecasts=(
                _forecast_write(
                    target,
                    series_keys=("a", "b", TOTAL_NODE_LABEL),
                ),
            ),
        )
    )

    first_submission = ActualsSubmission((ActualRecord("a", target, 2.0),))
    first = store.open(ActualsIntent(session, first_submission))
    assert {row.forecast_key.series_key for row in first.pending_observations} == {
        "a",
        TOTAL_NODE_LABEL,
    }
    first_cycle = _observe(first, hierarchy=hierarchy)
    assert {row.forecast_key.series_key for row in first_cycle.resolutions} == {"a"}
    store.commit(
        ActualsCommit(
            session=session,
            origin=first.origin,
            expected_revision=first.revision,
            actual_keys=tuple(record.key for record in first_submission.records),
            observe_cycle=first_cycle,
        )
    )

    second_submission = ActualsSubmission((ActualRecord("b", target, 3.0),))
    second = store.open(ActualsIntent(session, second_submission))
    assert {row.forecast_key.series_key for row in second.pending_observations} == {
        "b",
        TOTAL_NODE_LABEL,
    }
    second_cycle = _observe(second, hierarchy=hierarchy)
    assert {row.forecast_key.series_key for row in second_cycle.resolutions} == {
        "b",
        TOTAL_NODE_LABEL,
    }
    assert store.audit().pending_rows_examined == 4


def test_failed_commit_exposes_no_durable_family_or_cursor_advance() -> None:
    """Keep every durable family invisible after late index validation fails."""
    session = _session()
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals=_panel(),
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    store.commit(
        OriginCommit(
            session=session,
            origin=pd.Timestamp("2026-01-03"),
            expected_revision=store.revision,
            checkpoint_updates={"fixed": b"one"},
        )
    )
    snapshot = store.open(OriginIntent(session, pd.Timestamp("2026-01-04")))
    before = (
        store.revision,
        store.observed_history,
        store.pending_observations,
        store.forecasts,
        store.orders,
        store.settlements,
        store.observation_resolutions,
        store.observe_annotations,
        dict(store.states),
        dict(store.checkpoints),
        dict(store.checkpoint_indexes),
        store.resume_marker,
        dict(store.receipts),
        dict(store.settlement_receipts),
    )

    with pytest.raises(LedgerError, match="missing checkpoint"):
        store.commit(
            OriginCommit(
                session=session,
                origin=snapshot.origin,
                expected_revision=snapshot.revision,
                observe_cycle=_observe(snapshot),
                forecasts=(_forecast_write(snapshot.origin),),
                state_updates={"partition": b"staged"},
                checkpoint_updates={"staged": b"bytes"},
                checkpoint_indexes={"late": _checkpoint_index("missing")},
                resume_marker=snapshot.origin,
            )
        )

    assert (
        store.revision,
        store.observed_history,
        store.pending_observations,
        store.forecasts,
        store.orders,
        store.settlements,
        store.observation_resolutions,
        store.observe_annotations,
        dict(store.states),
        dict(store.checkpoints),
        dict(store.checkpoint_indexes),
        store.resume_marker,
        dict(store.receipts),
        dict(store.settlement_receipts),
    ) == before
    reopened = store.open(OriginIntent(session, snapshot.origin))
    assert reopened.actuals.records == snapshot.actuals.records


def test_unknown_hierarchy_membership_fails_before_any_publication() -> None:
    """Validate staged pending memberships before publishing ledger rows."""
    session = _session()
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals=_panel(),
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    snapshot = store.open(OriginIntent(session, pd.Timestamp("2026-01-03")))
    before = _durable_families(store)

    with pytest.raises(LedgerError, match="unknown hierarchy node"):
        store.commit(
            OriginCommit(
                session=session,
                origin=snapshot.origin,
                expected_revision=snapshot.revision,
                observe_cycle=_observe(snapshot),
                forecasts=(_forecast_write(snapshot.origin, series_keys=("unknown",)),),
                state_updates={"partition": b"staged"},
                checkpoint_updates={"staged": b"bytes"},
                checkpoint_indexes={"index": _checkpoint_index("staged")},
                resume_marker=snapshot.origin,
            )
        )

    assert _durable_families(store) == before


def test_sixty_four_origin_work_is_delta_proportional_and_reader_equivalent() -> None:
    """Keep equal append/resolve deltas flat while accumulated history grows."""
    hierarchy = HierarchyIndex.flat(("a", "b"))
    session = _session(hierarchy.node_labels)
    timestamps = pd.date_range("2026-01-01", periods=64, freq="D")
    series_keys = tuple(
        series_key for _timestamp in timestamps for series_key in hierarchy.bottom_series
    )
    repeated_timestamps = tuple(
        timestamp for timestamp in timestamps for _series_key in hierarchy.bottom_series
    )
    panel = Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(series_keys, dtype="string"),
                TIMESTAMP: pd.to_datetime(repeated_timestamps),
                OBSERVED_VALUE: pd.Series(
                    range(1, len(repeated_timestamps) + 1),
                    dtype="float64",
                ),
            }
        ),
        calendar=CALENDAR,
        target_support=TargetSupport.REAL,
    )
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals=panel,
        actuals_semantics=ActualsSemantics.DEMAND,
        hierarchy=hierarchy,
    )
    origins = tuple(CALENDAR.advance(timestamps[0], step) for step in range(1, 65))
    equal_delta_work: list[tuple[int, ...]] = []

    for index, origin in enumerate(origins):
        before = store.audit()
        snapshot = store.open(OriginIntent(session, origin))
        store.commit(
            OriginCommit(
                session=session,
                origin=origin,
                expected_revision=snapshot.revision,
                observe_cycle=_observe(snapshot, hierarchy=hierarchy),
                forecasts=(
                    _forecast_write(
                        origin,
                        point=float(index + 1),
                        series_keys=("a", "b", TOTAL_NODE_LABEL),
                    ),
                ),
                checkpoint_updates=(
                    {"checkpoint-a": b"a", "checkpoint-b": b"b"} if index == 0 else {}
                ),
                checkpoint_indexes=(
                    {
                        "index-a": _checkpoint_index("checkpoint-a"),
                        "index-b": _checkpoint_index("checkpoint-b"),
                    }
                    if index == 0
                    else {}
                ),
            )
        )
        if index:
            assert set(snapshot.checkpoints) == {"checkpoint-a", "checkpoint-b"}
        after = store.audit()
        delta = tuple(
            getattr(after, name) - getattr(before, name)
            for name in (
                "origin_opens",
                "source_rows_examined",
                "target_buckets_examined",
                "pending_rows_examined",
                "history_rows_examined",
                "commits",
                "history_rows_appended",
                "forecast_rows_appended",
                "resolution_rows_applied",
                "staged_rows_validated",
                "due_targets_indexed",
                "checkpoint_indexes_decoded",
            )
        )
        if index:
            equal_delta_work.append(delta)

    assert len(equal_delta_work) == 63
    assert set(equal_delta_work) == {(1, 4, 1, 3, 0, 1, 2, 3, 3, 3, 1, 0)}
    assert len(store.forecasts) == 192
    assert len(store.observation_resolutions) == 189
    assert len(store.observed_history) == 128
    assert store.audit().checkpoint_indexes_decoded == 2

    reader = InMemoryLedgerReader(store)

    def logical_rows(batch_size: int) -> tuple[tuple[object, ...], ...]:
        rows: list[tuple[object, ...]] = []
        for batch in reader.scan(
            LedgerSelection(
                session,
                (
                    LedgerColumn.TARGET_TIMESTAMP,
                    LedgerColumn.POINT_FORECAST,
                    LedgerColumn.RESOLUTION,
                ),
                batch_size,
            )
        ):
            rows.extend(
                (
                    key,
                    batch.columns[LedgerColumn.TARGET_TIMESTAMP.value][index],
                    batch.columns[LedgerColumn.POINT_FORECAST.value][index],
                    batch.columns[LedgerColumn.RESOLUTION.value][index],
                )
                for index, key in enumerate(batch.keys)
            )
        return tuple(rows)

    one_by_one = logical_rows(1)
    assert one_by_one == logical_rows(17)
    assert len(one_by_one) == 192
    assert sum(row[-1] is not None for row in one_by_one) == 189
