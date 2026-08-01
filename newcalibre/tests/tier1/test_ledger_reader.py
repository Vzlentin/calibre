"""Exercise bounded immutable reads over one closed logical ledger."""

from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
from threading import Thread
from typing import Any, cast

import pandas as pd
import pytest
from tests.conformal_fixtures import delivery_batch

import newcalibre.engine as engine_exports
from newcalibre.conformal import (
    EmissionForm,
    IssuedBoundFacts,
    ObserveAnnotation,
    ResolvedObservation,
    derive_partition_label,
)
from newcalibre.conformal import (
    ForecastKey as ConformalForecastKey,
)
from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    ActualsSemantics,
    Calendar,
    CensoringAssertion,
    DecisionScope,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeDescriptor,
    GuaranteeType,
    ScoredSeries,
    SessionIdentity,
    interval_columns,
)
from newcalibre.engine import (
    ForecastWrite,
    InMemoryLedgerReader,
    LedgerBatch,
    LedgerBoundIssuance,
    LedgerBoundScore,
    LedgerColumn,
    LedgerForecastKey,
    LedgerReader,
    LedgerResolution,
    LedgerSelection,
    LedgerSessionMetadata,
    OriginCommit,
    reporting,
)
from newcalibre.engine.ports.memory import InMemoryIndexedRunStore
from newcalibre.ledger import (
    BoundKey,
    ForecastIssuance,
    ForecastKey,
    GuaranteedSide,
    Ledger,
)
from newcalibre.observe import ObservationResolution, ObserveCycle

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
ORIGIN_DATE = pd.Timestamp("2026-01-02")


def _session(*, tenant: str = "tenant-a") -> SessionIdentity:
    return SessionIdentity.derive(
        tenant=tenant,
        series_keys=("a", "b"),
        calendar=CALENDAR,
        horizon=2,
        model_config={"name": "reader-fixture"},
    )


def _descriptor(
    *,
    window: EmissionScope = EmissionScope.PER_STEP,
) -> GuaranteeDescriptor:
    return GuaranteeDescriptor(
        type=GuaranteeType(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
        level=0.9,
        scored_series=ScoredSeries.DEMAND_HONEST,
        window=window,
        scope=DecisionScope(
            kind=DecisionScopeKind.PER_DECISION_NODE,
            class_system_name=None,
        ),
    )


def _row_write(
    *,
    series_key: str,
    origin: pd.Timestamp,
    horizon_step: int,
    model_name: str,
    window: EmissionScope = EmissionScope.PER_STEP,
) -> ForecastWrite:
    lower, upper = interval_columns(0.9)
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series([series_key], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime([CALENDAR.advance(origin, horizon_step - 1)]),
            ACTUAL_VALUE: pd.Series([None], dtype="float64"),
            POINT_FORECAST: pd.Series([7.0], dtype="float64"),
            HORIZON_STEP: pd.Series([horizon_step], dtype="int64"),
            ORIGIN: pd.to_datetime([origin]),
            MODEL_NAME: pd.Series([model_name], dtype="string"),
            lower: pd.Series([5.0], dtype="float64"),
            upper: pd.Series([10.0], dtype="float64"),
        }
    )
    key: ForecastKey = (series_key, origin, horizon_step, model_name)
    descriptor = _descriptor(window=window)
    observation_issuances = (
        {
            key: IssuedBoundFacts(
                method_name="split-per-step",
                emission_form=EmissionForm.ONE_SIDED_UPPER,
                emission_scope=EmissionScope.PER_STEP,
                partition_label=derive_partition_label(
                    model_name,
                    "global",
                    EmissionScope.PER_STEP,
                ),
                working_level=0.9,
                state_reference="split-per-step:reader-fixture",
                lower_bound=5.0,
                upper_bound=10.0,
                calibration_ready=True,
                bounds_null_reason=None,
                effective_descriptor=descriptor,
            )
        }
        if model_name == "z-model"
        else {}
    )
    return ForecastWrite(
        frame,
        {
            key: {
                (lower,): ForecastIssuance(
                    descriptor=descriptor,
                    guaranteed_side=GuaranteedSide.LOWER,
                    calibration_ready=True,
                    bounds_finite=True,
                    bounds_null_reason=None,
                ),
                (upper,): ForecastIssuance(
                    descriptor=descriptor,
                    guaranteed_side=GuaranteedSide.UPPER,
                    calibration_ready=True,
                    bounds_finite=True,
                    bounds_null_reason=None,
                ),
            }
        },
        observation_issuances,
    )


def _closed_sink(*, reverse_chunks: bool = False) -> InMemoryIndexedRunStore:
    session = _session()
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    logical_rows = (
        ("b", pd.Timestamp("2026-01-02"), 2, "z-model"),
        ("a", pd.Timestamp("2026-01-02"), 1, "y-model"),
        ("a", pd.Timestamp("2026-01-02"), 2, "x-model"),
        ("b", pd.Timestamp("2026-01-03"), 1, "pending"),
        ("a", pd.Timestamp("2026-01-03"), 1, "x-model"),
    )
    for origin in (pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")):
        writes = [
            _row_write(
                series_key=series_key,
                origin=row_origin,
                horizon_step=step,
                model_name=model,
            )
            for series_key, row_origin, step, model in logical_rows
            if row_origin == origin
        ]
        if reverse_chunks:
            writes.reverse()
        store.commit(
            OriginCommit(
                session=session,
                origin=origin,
                expected_revision=store.revision,
                observe_cycle=ObserveCycle(
                    pending_retentions=store.pending_observations,
                ),
                forecasts=tuple(writes),
            )
        )

    pending_by_key = {
        (
            value.forecast_key.series_key,
            value.forecast_key.origin,
            value.forecast_key.horizon_step,
            value.forecast_key.model_name,
        ): value
        for value in store.pending_observations
    }
    actuals = {
        key: 12.0 if key[3] == "z-model" else 7.0 for key in pending_by_key if key[3] != "pending"
    }
    resolution_rows = tuple(
        ObservationResolution(
            pending_by_key[key].forecast_key,
            pending_by_key[key].target_timestamp,
            actual,
            CensoringAssertion.CENSORED if key[3] == "z-model" else None,
            11.0 if key[3] == "z-model" else None,
        )
        for key, actual in actuals.items()
    )
    delivered = next(pending for key, pending in pending_by_key.items() if key[3] == "z-model")
    assert delivered.issued is not None
    delivery = delivery_batch(
        delivered.issued.partition_label,
        (
            ResolvedObservation(
                delivered.forecast_key,
                delivered.target_timestamp,
                actuals[("b", pd.Timestamp("2026-01-02"), 2, "z-model")],
                delivered.point_forecast,
                CensoringAssertion.CENSORED,
                11.0,
                delivered.issued,
            ),
        ),
    )
    store.commit(
        OriginCommit(
            session=session,
            origin=pd.Timestamp("2026-01-05"),
            expected_revision=store.revision,
            observe_cycle=ObserveCycle(
                resolutions=resolution_rows,
                pending_removals=tuple(
                    ConformalForecastKey(key[0], key[1], key[2], key[3]) for key in actuals
                ),
                pending_retentions=tuple(
                    value for key, value in pending_by_key.items() if key not in actuals
                ),
                deliveries=delivery,
                annotations=(
                    ObserveAnnotation(
                        delivered.forecast_key,
                        None,
                        "declared-censored",
                        False,
                    ),
                ),
            ),
        )
    )
    return store


def _window_sum_sink() -> InMemoryIndexedRunStore:
    session = _session()
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    models = ("complete-window", "partial-window")
    store.commit(
        OriginCommit(
            session=session,
            origin=ORIGIN_DATE,
            expected_revision=store.revision,
            forecasts=tuple(
                _row_write(
                    series_key="a",
                    origin=ORIGIN_DATE,
                    horizon_step=step,
                    model_name=model,
                    window=EmissionScope.WINDOW_SUM,
                )
                for model in models
                for step in (1, 2)
            ),
        )
    )

    pending_by_key = {
        (
            value.forecast_key.series_key,
            value.forecast_key.origin,
            value.forecast_key.horizon_step,
            value.forecast_key.model_name,
        ): value
        for value in store.pending_observations
    }
    actuals = {
        ("a", ORIGIN_DATE, 1, "complete-window"): 4.0,
        ("a", ORIGIN_DATE, 2, "complete-window"): 7.0,
        ("a", ORIGIN_DATE, 1, "partial-window"): 6.0,
    }
    store.commit(
        OriginCommit(
            session=session,
            origin=pd.Timestamp("2026-01-04"),
            expected_revision=store.revision,
            observe_cycle=ObserveCycle(
                resolutions=tuple(
                    ObservationResolution(
                        pending_by_key[key].forecast_key,
                        pending_by_key[key].target_timestamp,
                        actual,
                        None,
                        None,
                    )
                    for key, actual in actuals.items()
                ),
                pending_removals=tuple(pending_by_key[key].forecast_key for key in actuals),
                pending_retentions=tuple(
                    pending for key, pending in pending_by_key.items() if key not in actuals
                ),
            ),
        )
    )
    return store


def _scan_rows(
    reader: LedgerReader,
    *,
    batch_size: int,
    origin_start: pd.Timestamp | None = None,
    origin_end: pd.Timestamp | None = None,
) -> list[tuple[LedgerForecastKey, dict[str, object]]]:
    columns = tuple(column.value for column in LedgerColumn)
    rows: list[tuple[LedgerForecastKey, dict[str, object]]] = []
    for batch in reader.scan(
        LedgerSelection(
            _session(),
            columns,
            batch_size,
            origin_start=origin_start,
            origin_end=origin_end,
        )
    ):
        assert batch.row_count <= batch_size
        assert tuple(batch.columns) == columns
        for index, key in enumerate(batch.keys):
            rows.append(
                (
                    key,
                    {name: values[index] for name, values in batch.columns.items()},
                )
            )
    return rows


def test_selection_normalizes_and_validates_the_closed_column_vocabulary() -> None:
    session = _session()
    selection = LedgerSelection(
        session=session,
        columns=[LedgerColumn.SERIES_KEY, "resolution"],
        batch_size=2,
        origin_start=ORIGIN_DATE,
        origin_end=ORIGIN_DATE,
    )

    assert selection.session == session
    assert selection.columns == ("series_key", "resolution")
    assert selection.batch_size == 2
    assert selection.origin_start == ORIGIN_DATE
    assert selection.origin_end == ORIGIN_DATE

    with pytest.raises(TypeError, match="SessionIdentity"):
        LedgerSelection(cast(Any, "session"), ("series_key",), 1)
    with pytest.raises(TypeError, match="columns must be an iterable"):
        LedgerSelection(session, "series_key", 1)
    with pytest.raises(ValueError, match="must not be empty"):
        LedgerSelection(session, (), 1)
    with pytest.raises(ValueError, match="duplicate"):
        LedgerSelection(session, ("series_key", "series_key"), 1)
    with pytest.raises(ValueError, match="unsupported"):
        LedgerSelection(session, ("store_row_id",), 1)


@pytest.mark.parametrize("batch_size", [True, 0, -1, 1.5])
def test_selection_rejects_invalid_batch_sizes(batch_size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="batch size"):
        LedgerSelection(_session(), ("series_key",), cast(Any, batch_size))


@pytest.mark.parametrize(
    ("start", "end", "error"),
    [
        ("2026-01-01", None, TypeError),
        (None, pd.Timestamp("2026-01-01", tz="UTC"), ValueError),
        (pd.NaT, None, TypeError),
        (pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-02"), ValueError),
    ],
)
def test_selection_rejects_malformed_inclusive_origin_ranges(
    start: object,
    end: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="origin"):
        LedgerSelection(
            _session(),
            ("origin",),
            1,
            origin_start=cast(Any, start),
            origin_end=cast(Any, end),
        )


def test_batch_snapshots_columns_and_enforces_alignment_and_bound() -> None:
    session = _session()
    keys = [LedgerForecastKey("a", ORIGIN_DATE, 1, "fixture")]
    values = ["a"]
    batch = LedgerBatch(
        session=session,
        keys=keys,
        columns={"series_key": values},
        batch_size=1,
    )
    keys.clear()
    values[0] = "mutated"

    assert batch.keys == (LedgerForecastKey("a", ORIGIN_DATE, 1, "fixture"),)
    assert batch.columns == {"series_key": ("a",)}
    assert batch.row_count == 1
    assert len(batch) == 1
    with pytest.raises(TypeError):
        cast(Any, batch.columns)["series_key"] = ("mutated",)
    with pytest.raises(FrozenInstanceError):
        cast(Any, batch).batch_size = 2

    with pytest.raises(ValueError, match="equal lengths"):
        LedgerBatch(
            session=session,
            keys=batch.keys,
            columns={"series_key": ("a",), "origin": ()},
            batch_size=1,
        )
    with pytest.raises(ValueError, match="batch size"):
        LedgerBatch(
            session=session,
            keys=(batch.keys[0], LedgerForecastKey("b", ORIGIN_DATE, 1, "fixture")),
            columns={"series_key": ("a", "b")},
            batch_size=1,
        )


def test_reader_protocol_is_runtime_checkable() -> None:
    class Reader:
        @property
        def metadata(self) -> LedgerSessionMetadata:
            return LedgerSessionMetadata(_session(), _session().series_keys)

        def scan(self, selection: LedgerSelection) -> Iterator[LedgerBatch]:
            del selection
            return iter(())

    assert isinstance(Reader(), LedgerReader)


def test_reader_metadata_is_immutable_and_session_consistent() -> None:
    reader = InMemoryLedgerReader(_closed_sink())

    assert reader.metadata == LedgerSessionMetadata(_session(), ("a", "b"))
    assert reader.metadata.series_keys == ("a", "b")
    with pytest.raises(FrozenInstanceError):
        cast(Any, reader.metadata).series_keys = ("mutated",)
    with pytest.raises(ValueError, match="session"):
        LedgerSessionMetadata(_session(), ("a",))


def test_engine_exports_only_the_reporting_contract_and_adapter() -> None:
    assert set(reporting.__all__) <= set(engine_exports.__all__)
    assert "InMemoryLedgerReader" in engine_exports.__all__
    assert "Ledger" not in engine_exports.__all__
    assert "ForecastRow" not in engine_exports.__all__


def test_reporting_values_are_deeply_immutable() -> None:
    key = LedgerForecastKey("a", ORIGIN_DATE, 1, "fixture")
    resolution = LedgerResolution(
        target_timestamp=ORIGIN_DATE,
        actual_value=3.0,
        censoring_assertion=None,
        availability_bound=None,
        annotation=None,
    )
    assert resolution.actual_value == 3.0
    with pytest.raises(FrozenInstanceError):
        cast(Any, key).series_key = "mutated"
    with pytest.raises(FrozenInstanceError):
        cast(Any, resolution).actual_value = 4.0


def test_reader_rejects_unknown_sessions_before_returning_an_iterator() -> None:
    reader = InMemoryLedgerReader(_closed_sink())
    selection = LedgerSelection(
        _session(tenant="other-tenant"),
        ("series_key",),
        2,
    )

    with pytest.raises(ValueError, match="session"):
        reader.scan(selection)


def test_closed_scan_is_canonical_and_independent_of_chunk_append_order() -> None:
    forward = _scan_rows(InMemoryLedgerReader(_closed_sink()), batch_size=2)
    reversed_chunks = _scan_rows(
        InMemoryLedgerReader(_closed_sink(reverse_chunks=True)),
        batch_size=2,
    )

    assert forward == reversed_chunks
    assert [
        (key.origin, key.series_key, key.model_name, key.horizon_step) for key, _columns in forward
    ] == sorted(
        (key.origin, key.series_key, key.model_name, key.horizon_step) for key, _columns in forward
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4, 20])
def test_closed_scan_is_batch_invariant_and_origin_bounds_are_inclusive(
    batch_size: int,
) -> None:
    reader = InMemoryLedgerReader(_closed_sink())
    expected = _scan_rows(reader, batch_size=20)

    assert _scan_rows(reader, batch_size=batch_size) == expected
    selected = _scan_rows(
        reader,
        batch_size=batch_size,
        origin_start=pd.Timestamp("2026-01-03"),
        origin_end=pd.Timestamp("2026-01-03"),
    )
    assert selected
    assert {key.origin for key, _columns in selected} == {pd.Timestamp("2026-01-03")}


def test_suspended_scan_releases_the_sink_lock_and_keeps_its_row_cutoff() -> None:
    store = _closed_sink()
    reader = InMemoryLedgerReader(store)
    iterator = reader.scan(LedgerSelection(_session(), ("series_key",), 1))
    first = next(iterator)
    write = OriginCommit(
        session=_session(),
        origin=pd.Timestamp("2026-01-04"),
        expected_revision=store.revision,
        observe_cycle=ObserveCycle(pending_retentions=store.pending_observations),
        forecasts=(
            _row_write(
                series_key="a",
                origin=pd.Timestamp("2026-01-04"),
                horizon_step=1,
                model_name="later-model",
            ),
        ),
    )
    errors: list[BaseException] = []

    def commit_later_forecast() -> None:
        try:
            store.commit(write)
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    thread = Thread(target=commit_later_forecast, daemon=True)
    thread.start()
    thread.join(timeout=2)
    blocked = thread.is_alive()
    if blocked:
        iterator.close()
        thread.join(timeout=2)

    assert blocked is False
    assert errors == []
    rows = [*first.keys, *(key for batch in iterator for key in batch.keys)]
    assert len(rows) == 5
    assert all(key.model_name != "later-model" for key in rows)
    assert len(_scan_rows(reader, batch_size=20)) == 6


def test_scan_matches_owned_issuance_resolution_and_registered_scores() -> None:
    store = _closed_sink()
    rows = dict(_scan_rows(InMemoryLedgerReader(store), batch_size=2))
    stored_rows = {row.key: row for row in store.forecasts}
    resolutions = {
        (
            value.forecast_key.series_key,
            value.forecast_key.origin,
            value.forecast_key.horizon_step,
            value.forecast_key.model_name,
        ): value
        for value in store.observation_resolutions
    }
    annotations = {
        (
            value.forecast_key.series_key,
            value.forecast_key.origin,
            value.forecast_key.horizon_step,
            value.forecast_key.model_name,
        ): value
        for value in store.observe_annotations
    }
    outcomes = {
        (outcome.forecast_key, outcome.bound_key): outcome
        for outcome in store.coverage_report().outcomes
    }

    for reported_key, columns in rows.items():
        key: ForecastKey = (
            reported_key.series_key,
            reported_key.origin,
            reported_key.horizon_step,
            reported_key.model_name,
        )
        stored = stored_rows[key]
        assert columns["series_key"] == stored.series_key
        assert columns["origin"] == stored.origin
        assert columns["horizon_step"] == stored.horizon_step
        assert columns["model_name"] == stored.model_name
        assert columns["target_timestamp"] == stored.target_timestamp
        assert columns["point_forecast"] == stored.point_forecast

        reported_issuances = cast(tuple[LedgerBoundIssuance, ...], columns["issuances"])
        assert len(reported_issuances) == len(stored.issuances)
        for issuance in reported_issuances:
            expected = stored.issuances[issuance.bound_key]
            assert issuance.bound_values == tuple(
                stored.values[column] for column in issuance.bound_key
            )
            assert issuance.descriptor == expected.descriptor
            assert issuance.guaranteed_side == expected.guaranteed_side
            assert issuance.calibration_ready is expected.calibration_ready
            assert issuance.bounds_finite is expected.bounds_finite
            assert issuance.bounds_null_reason == expected.bounds_null_reason

        reported_resolution = cast(LedgerResolution | None, columns["resolution"])
        expected_resolution = resolutions.get(key)
        if expected_resolution is None:
            assert reported_resolution is None
        else:
            assert reported_resolution is not None
            assert reported_resolution.target_timestamp == expected_resolution.target_timestamp
            assert reported_resolution.actual_value == expected_resolution.actual
            assert (
                reported_resolution.censoring_assertion is expected_resolution.censoring_assertion
            )
            assert reported_resolution.availability_bound == expected_resolution.availability_bound
            expected_annotation = annotations.get(key)
            if expected_annotation is None:
                assert reported_resolution.annotation is None
            else:
                assert reported_resolution.annotation is not None
                assert reported_resolution.annotation.score == expected_annotation.score
                assert (
                    reported_resolution.annotation.exclusion_cause
                    == expected_annotation.exclusion_cause
                )
                assert (
                    reported_resolution.annotation.advanced_delivered_score
                    is expected_annotation.advanced_delivered_score
                )

        reported_scores = cast(tuple[Any, ...], columns["scores"])
        assert len(reported_scores) == len(stored.issuances)
        for score in reported_scores:
            expected_outcome = outcomes[(key, score.bound_key)]
            assert score.descriptor == expected_outcome.target.descriptor
            assert score.guaranteed_side == expected_outcome.target.guaranteed_side
            assert score.resolved is expected_outcome.resolved
            assert score.scored is expected_outcome.scored
            assert score.value == expected_outcome.value
            assert score.covered is expected_outcome.covered
            assert score.unscored_reason == expected_outcome.unscored_reason


def test_window_sum_scan_matches_complete_and_partial_coverage_outcomes() -> None:
    store = _window_sum_sink()
    rows = dict(_scan_rows(InMemoryLedgerReader(store), batch_size=1))
    outcomes = {
        (outcome.forecast_key, outcome.bound_key): outcome
        for outcome in store.coverage_report().outcomes
    }

    for reported_key, columns in rows.items():
        key: ForecastKey = (
            reported_key.series_key,
            reported_key.origin,
            reported_key.horizon_step,
            reported_key.model_name,
        )
        for score in cast(tuple[LedgerBoundScore, ...], columns["scores"]):
            expected = outcomes[(key, score.bound_key)]
            assert score.descriptor == expected.target.descriptor
            assert score.guaranteed_side == expected.target.guaranteed_side
            assert score.resolved is expected.resolved
            assert score.scored is expected.scored
            assert score.value == expected.value
            assert score.covered is expected.covered
            assert score.unscored_reason == expected.unscored_reason

    complete = cast(
        tuple[LedgerBoundScore, ...],
        rows[LedgerForecastKey("a", ORIGIN_DATE, 2, "complete-window")]["scores"],
    )
    partial = cast(
        tuple[LedgerBoundScore, ...],
        rows[LedgerForecastKey("a", ORIGIN_DATE, 2, "partial-window")]["scores"],
    )
    assert all(score.resolved and score.scored for score in complete)
    assert all(not score.resolved and not score.scored for score in partial)


def test_consumer_mutation_cannot_change_stored_or_fresh_scan_facts() -> None:
    store = _closed_sink()
    reader = InMemoryLedgerReader(store)
    selection = LedgerSelection(_session(), ("issuances",), 2)
    batch = next(reader.scan(selection))
    issuances = cast(tuple[LedgerBoundIssuance, ...], batch.columns["issuances"][0])
    first = _scan_rows(reader, batch_size=2)

    with pytest.raises(TypeError):
        cast(Any, batch.columns)["new"] = ()
    with pytest.raises(TypeError):
        cast(Any, batch.columns["issuances"])[0] = ()
    with pytest.raises(TypeError):
        cast(Any, issuances)[0] = issuances[0]
    with pytest.raises(FrozenInstanceError):
        cast(Any, issuances[0]).bound_values = (99.0,)

    assert _scan_rows(reader, batch_size=2) == first


def test_reader_never_calls_full_ledger_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _closed_sink()

    def forbidden_projection(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("reader called a forbidden full-ledger projection")

    monkeypatch.setattr(Ledger, "forecasts", property(forbidden_projection))
    monkeypatch.setattr(Ledger, "coverage_report", forbidden_projection)

    assert len(_scan_rows(InMemoryLedgerReader(store), batch_size=2)) == 5


def _many_row_sink(row_count: int) -> InMemoryIndexedRunStore:
    session = _session()
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=CALENDAR,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    origin = pd.Timestamp("2026-01-02")
    lower, upper = interval_columns(0.9)
    models = [f"model-{index:04d}" for index in range(row_count)]
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["a"] * row_count, dtype="string"),
            TARGET_TIMESTAMP: pd.Series([origin] * row_count, dtype="datetime64[ns]"),
            ACTUAL_VALUE: pd.Series([None] * row_count, dtype="float64"),
            POINT_FORECAST: pd.Series([7.0] * row_count, dtype="float64"),
            HORIZON_STEP: pd.Series([1] * row_count, dtype="int64"),
            ORIGIN: pd.Series([origin] * row_count, dtype="datetime64[ns]"),
            MODEL_NAME: pd.Series(models, dtype="string"),
            lower: pd.Series([5.0] * row_count, dtype="float64"),
            upper: pd.Series([10.0] * row_count, dtype="float64"),
        }
    )
    descriptor = _descriptor()
    issuance: Mapping[BoundKey, ForecastIssuance] = {
        (lower,): ForecastIssuance(
            descriptor,
            GuaranteedSide.LOWER,
            True,
            True,
            None,
        ),
        (upper,): ForecastIssuance(
            descriptor,
            GuaranteedSide.UPPER,
            True,
            True,
            None,
        ),
    }
    by_key = {("a", origin, 1, model): issuance for model in models}
    store.commit(
        OriginCommit(
            session=session,
            origin=origin,
            expected_revision=store.revision,
            forecasts=(ForecastWrite(frame, by_key),),
        )
    )
    return store


def test_scan_reads_each_stored_row_once_across_many_batches() -> None:
    store = _many_row_sink(21)
    stored = store._ledger._forecasts
    reads = 0

    class CountingForecasts(dict[ForecastKey, object]):
        def __getitem__(self, key: ForecastKey) -> object:
            nonlocal reads
            reads += 1
            return super().__getitem__(key)

    store._ledger._forecasts = cast(Any, CountingForecasts(stored))

    assert len(_scan_rows(InMemoryLedgerReader(store), batch_size=3)) == 21
    assert reads == 21


def _scan_peak_bytes(store: InMemoryIndexedRunStore) -> int:
    reader = InMemoryLedgerReader(store)
    selection = LedgerSelection(_session(), ("series_key",), 3)
    gc.collect()
    tracemalloc.start()
    try:
        for batch in reader.scan(selection):
            assert 0 < batch.row_count <= 3
            del batch
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_scan_allocation_does_not_scale_with_closed_ledger_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_projection(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("reader called a forbidden full-ledger projection")

    small = _many_row_sink(120)
    large = _many_row_sink(480)
    monkeypatch.setattr(Ledger, "forecasts", property(forbidden_projection))
    monkeypatch.setattr(Ledger, "coverage_report", forbidden_projection)
    small_peak = _scan_peak_bytes(small)
    large_peak = _scan_peak_bytes(large)

    assert large_peak <= small_peak + 100_000
