"""Exercise all six engine ports through their in-memory adapters."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest
from tests.conformal_fixtures import delivery_batch

from newcalibre.conformal import (
    ConformalStateBatch,
    ObserveAnnotation,
    ResolvedObservation,
    resolve_method,
)
from newcalibre.conformal import (
    ForecastKey as ConformalForecastKey,
)
from newcalibre.domain import (
    ACTUAL_VALUE,
    AVAILABILITY_BOUND,
    CENSOR_STATUS,
    HORIZON_STEP,
    MODEL_NAME,
    OBSERVED_VALUE,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    TIMESTAMP,
    ActualsSemantics,
    Calendar,
    CostStructure,
    DecisionTiming,
    Panel,
    SessionIdentity,
    StockoutRule,
    TargetSupport,
)
from newcalibre.engine import (
    CommitReceipt,
    Engine,
    EventDriver,
    ForecastWrite,
    OriginCommit,
    Spine,
    TimeLoop,
)
from newcalibre.engine import ports as engine_ports
from newcalibre.engine.ports import SettlementSnapshot
from newcalibre.engine.ports.memory import (
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerReader,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    InProcessDispatch,
)
from newcalibre.ledger import LedgerError, OrderRow
from newcalibre.observe import (
    ObservationResolution,
    ObserveCycle,
    ObservedActual,
    PendingObservation,
)

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
ORIGIN_DATE = pd.Timestamp("2026-01-05")


def _panel() -> Panel:
    return Panel.from_frame(
        pd.DataFrame(
            {
                SERIES_KEY: pd.Series(["b", "a", "a"], dtype="string"),
                TIMESTAMP: pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
                OBSERVED_VALUE: pd.Series([3.0, 1.0, 2.0], dtype="float64"),
            }
        ),
        calendar=CALENDAR,
        target_support=TargetSupport.REAL,
    )


def _session() -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=("a", "b"),
        calendar=CALENDAR,
        horizon=2,
        model_config={"backend": "fixture", "name": "fixture"},
        ordering_policy={"name": "newsvendor"},
        decision_series_keys=("a", "b"),
        cost_structure=CostStructure(1.0, 1.0, 1.0, 1.0),
        decision_timing=DecisionTiming(lead_time=1, review_period=1),
        stockout_rule=StockoutRule.LOST_SALES,
    )


def _forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["a"], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime([ORIGIN_DATE]),
            ACTUAL_VALUE: pd.Series([float("nan")], dtype="float64"),
            POINT_FORECAST: pd.Series([2.0], dtype="float64"),
            HORIZON_STEP: pd.Series([1], dtype="int64"),
            ORIGIN: pd.to_datetime([ORIGIN_DATE]),
            MODEL_NAME: pd.Series(["fixture"], dtype="string"),
        }
    )


def test_in_memory_adapters_preserve_snapshots_and_deterministic_order() -> None:
    panel = _panel()
    panel_source = InMemoryPanelSource(panel)
    loaded = panel_source.load()
    mutated = loaded.frame
    mutated.loc[:, OBSERVED_VALUE] = 99.0
    assert panel_source.load().frame[OBSERVED_VALUE].tolist() == [1.0, 2.0, 3.0]

    actuals = InMemoryActualsSource(
        panel,
        actuals_semantics=ActualsSemantics.DEMAND,
    ).reveal(before=pd.Timestamp("2026-01-02"))
    assert tuple((record.key, record.recorded_value) for record in actuals.records) == (
        (("a", pd.Timestamp("2026-01-01")), 1.0),
        (("b", pd.Timestamp("2026-01-01")), 3.0),
    )

    artifacts = InMemoryArtifactStore()
    artifacts.save("model:a", b"one")
    artifacts.save("model:a", b"one")
    assert artifacts.load("model:a") == b"one"
    with pytest.raises(ValueError, match="different bytes"):
        artifacts.save("model:a", b"two")
    artifacts.save_index("model-index:a", b"first")
    artifacts.save_index("model-index:a", b"second")
    assert artifacts.load_index("model-index:a") == b"second"
    assert artifacts.artifact_indexes == {"model-index:a": b"second"}
    artifacts.publish(
        {"model:b": b"three"},
        {"model-index:b": b"third"},
    )
    with pytest.raises(ValueError, match="different bytes"):
        artifacts.publish(
            {"model:a": b"conflict"},
            {"model-index:a": b"must-not-publish"},
        )
    assert artifacts.load("model:b") == b"three"
    assert artifacts.load_index("model-index:a") == b"second"

    states = InMemoryCalibrationStateStore()
    session = _session()
    states.save(session, "series:a", b"state", sequence=1)
    assert states.snapshot(session) == {"series:a": b"state"}
    states.save(
        session,
        "series:a",
        b"stale",
        sequence=0,
    )
    assert states.snapshot(session) == {"series:a": b"state"}
    with pytest.raises(ValueError, match="already holds different bytes"):
        states.save(session, "series:a", b"conflict", sequence=1)

    dispatch = InProcessDispatch()
    assert dispatch.map(lambda value: value * 2, (3, 1, 2)) == (6, 2, 4)


def test_actuals_source_requires_and_enforces_observation_semantics() -> None:
    panel = _panel()
    with pytest.raises(TypeError, match="actuals_semantics"):
        InMemoryActualsSource(panel)  # type: ignore[call-arg]

    censored_frame = panel.frame
    censored_frame[CENSOR_STATUS] = pd.Series(
        ["censored", "uncensored", "undeclared"],
        dtype="string",
    )
    censored_frame[AVAILABILITY_BOUND] = pd.Series(
        [1.5, None, 3.5],
        dtype="float64",
    )
    censored_panel = Panel.from_frame(
        censored_frame, calendar=CALENDAR, target_support=TargetSupport.REAL
    )

    with pytest.raises(ValueError, match="cannot supply demand-honest actuals"):
        InMemoryActualsSource(
            censored_panel,
            actuals_semantics=ActualsSemantics.DEMAND,
        )

    surrogate = InMemoryActualsSource(
        censored_panel,
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )
    assert surrogate.actuals_semantics is ActualsSemantics.CENSORED_SALES_SURROGATE
    revealed = surrogate.reveal(before=pd.Timestamp("2026-01-03"))
    assert [
        record.censoring_assertion.value if record.censoring_assertion is not None else None
        for record in revealed.records
    ] == ["censored", "uncensored", None]
    assert [record.availability_bound for record in revealed.records] == [1.5, None, 3.5]


def test_engine_declares_exactly_the_six_chapter_03_ports() -> None:
    protocols = {
        name
        for name, value in vars(engine_ports).items()
        if inspect.isclass(value)
        and value.__module__ == engine_ports.__name__
        and getattr(value, "_is_protocol", False)
    }
    assert protocols == {
        "PanelSource",
        "ActualsSource",
        "ArtifactStore",
        "CalibrationStateStore",
        "LedgerSink",
        "DispatchBackend",
    }
    assert not hasattr(engine_ports, "LedgerReader")


def test_reporting_adapter_is_structurally_absent_from_every_write_path() -> None:
    write_path_modules = {
        inspect.getmodule(value)
        for value in (
            Engine,
            Spine,
            TimeLoop,
            EventDriver,
            OriginCommit,
            CommitReceipt,
        )
    }
    assert None not in write_path_modules
    for module in write_path_modules:
        source = inspect.getsource(module)
        assert "InMemoryLedgerReader" not in source
        assert "engine.reporting" not in source

    assert "InMemoryLedgerReader" not in inspect.getsource(InMemoryLedgerSink)
    assert isinstance(InMemoryLedgerReader, type)


def test_ledger_sink_exposes_only_a_period_bound_compact_settlement_snapshot() -> None:
    session = _session()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)

    snapshot = sink.settlement_snapshot((ORIGIN_DATE,))

    assert sink.pending_observation_count == 0
    assert isinstance(snapshot, SettlementSnapshot)
    assert snapshot.periods == (ORIGIN_DATE,)
    assert snapshot.frontier is None
    assert snapshot.latest_positions == {}
    assert snapshot.open_order_quantities == {"a": 0.0, "b": 0.0}
    assert snapshot.due_arrivals == {}
    assert snapshot.actuals_semantics is None
    assert not hasattr(snapshot, "forecasts")
    assert not hasattr(snapshot, "orders")
    assert not hasattr(snapshot, "settlements")


def test_ledger_sink_refuses_initial_arrivals_without_a_decision_session() -> None:
    session = SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=("a",),
        calendar=CALENDAR,
        horizon=2,
        model_config={"backend": "fixture", "name": "fixture"},
    )

    with pytest.raises(LedgerError, match="require a session decision configuration"):
        InMemoryLedgerSink(
            session=session,
            calendar=CALENDAR,
            initial_arrivals={("a", ORIGIN_DATE): 1.0},
        )


def test_ledger_sink_rejects_a_misattributed_origin_without_a_partial_commit() -> None:
    session = _session()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    key = ("a", ORIGIN_DATE, 1, "fixture")
    forecast = ForecastWrite(_forecast_frame(), {key: {}})
    sink.commit(OriginCommit(session=session, origin=ORIGIN_DATE, forecasts=(forecast,)))

    order = OrderRow(
        session=session,
        series_key="a",
        origin=ORIGIN_DATE,
        model_name="fixture",
        quantity=1.0,
        arrival_period=pd.Timestamp("2026-01-06"),
    )
    with pytest.raises(LedgerError, match="forecast row origin must match"):
        sink.commit(
            OriginCommit(
                session=session,
                origin=pd.Timestamp("2026-01-06"),
                forecasts=(forecast,),
                orders=(order,),
            )
        )

    assert len(sink.forecasts) == 1
    assert sink.orders == ()


def test_commit_digest_is_sensitive_to_every_observe_materialization_family() -> None:
    session = _session()
    origin = pd.Timestamp("2026-01-06")
    key = ConformalForecastKey("a", ORIGIN_DATE, 1, "fixture")
    resolution = ObservationResolution(key, ORIGIN_DATE, 2.0, None, None)
    pending = PendingObservation(key, ORIGIN_DATE, 1.0, resolution=resolution)
    annotation = ObserveAnnotation(key, 1.0, None, True)
    runtime = resolve_method(
        {"method": "split-per-step", "coverage": 0.5, "partition_by": "global"}
    )
    issued = runtime.apply(_forecast_frame(), ConformalStateBatch()).issuances[key]
    delivery = delivery_batch(
        issued.partition_label,
        (
            ResolvedObservation(
                key,
                ORIGIN_DATE,
                2.0,
                1.0,
                None,
                None,
                issued,
            ),
        ),
    )
    without_delivery = ObserveCycle(
        resolutions=(resolution,),
        pending_removals=(key,),
        annotations=(annotation,),
    )
    with_delivery = ObserveCycle(
        resolutions=(resolution,),
        pending_removals=(key,),
        deliveries=delivery,
        annotations=(annotation,),
    )
    variants = (
        ObserveCycle(history_appends=(ObservedActual("a", ORIGIN_DATE, 2),)),
        ObserveCycle(history_appends=(ObservedActual("a", ORIGIN_DATE, 2.0),)),
        ObserveCycle(resolutions=(resolution,)),
        ObserveCycle(pending_removals=(key,)),
        ObserveCycle(pending_retentions=(pending,)),
        ObserveCycle(annotations=(annotation,)),
        without_delivery,
        with_delivery,
        ObserveCycle(state_updates={"state": b"value"}),
    )

    digests = {
        OriginCommit(session=session, origin=origin, observe_cycle=cycle).digest
        for cycle in variants
    }

    assert (
        OriginCommit(
            session=session,
            origin=origin,
            observe_cycle=with_delivery,
        ).digest
        != OriginCommit(
            session=session,
            origin=origin,
            observe_cycle=without_delivery,
        ).digest
    )
    assert len(digests) == len(variants)

    committed = OriginCommit(
        session=session,
        origin=origin,
        observe_cycle=variants[-1],
        state_updates={"state": b"value"},
    )
    receipt = CommitReceipt.from_commit(committed, sequence=1)
    assert receipt.observe_cycle == committed.observe_cycle
    assert receipt.state_updates == committed.state_updates
    assert receipt.digest == committed.digest


def test_full_observe_fact_retry_is_idempotent_and_conflict_preserves_history() -> None:
    session = _session()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    observed = ObservedActual(
        "a",
        pd.Timestamp("2026-01-01"),
        1.0,
        availability_bound=2.0,
    )
    commit = OriginCommit(
        session=session,
        origin=ORIGIN_DATE,
        observe_cycle=ObserveCycle(history_appends=(observed,)),
    )

    first = sink.commit(commit)
    assert sink.commit(commit) == first
    assert sink.observed_history == (observed,)

    conflict = OriginCommit(
        session=session,
        origin=ORIGIN_DATE,
        observe_cycle=ObserveCycle(
            history_appends=(
                ObservedActual(
                    "a",
                    pd.Timestamp("2026-01-01"),
                    1.0,
                    availability_bound=3.0,
                ),
            )
        ),
    )
    with pytest.raises(LedgerError, match="different committed write"):
        sink.commit(conflict)
    assert sink.observed_history == (observed,)


def test_commit_digest_frames_state_key_and_value_boundaries() -> None:
    session = _session()
    origin = pd.Timestamp("2026-01-06")
    first = OriginCommit(
        session=session,
        origin=origin,
        state_updates={"a": b"bc"},
    )
    shifted = OriginCommit(
        session=session,
        origin=origin,
        state_updates={"ab": b"c"},
    )
    assert first.digest != shifted.digest

    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)
    sink.commit(first)
    with pytest.raises(LedgerError, match="different committed write"):
        sink.commit(shifted)


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64])
def test_commit_receipt_rejects_noncanonical_digests(digest: str) -> None:
    with pytest.raises(ValueError, match="SHA-256 hex string"):
        CommitReceipt(
            session=_session(),
            origin=ORIGIN_DATE,
            digest=digest,
            state_updates={},
            sequence=0,
        )


def test_commit_receipt_validates_and_freezes_state_updates() -> None:
    state_updates = {"series:a": b"state"}
    receipt = CommitReceipt(
        session=_session(),
        origin=ORIGIN_DATE,
        digest="a" * 64,
        state_updates=state_updates,
        sequence=0,
    )
    state_updates["series:a"] = b"mutated"
    assert receipt.state_updates == {"series:a": b"state"}

    with pytest.raises(ValueError, match="non-empty trimmed string"):
        CommitReceipt(
            session=_session(),
            origin=ORIGIN_DATE,
            digest="a" * 64,
            state_updates={" untrimmed": b"state"},
            sequence=0,
        )
    with pytest.raises(TypeError, match="must contain bytes"):
        CommitReceipt(
            session=_session(),
            origin=ORIGIN_DATE,
            digest="a" * 64,
            state_updates={"series:a": "state"},  # type: ignore[dict-item]
            sequence=0,
        )
