"""Exercise all six engine ports through their in-memory adapters."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from newcalibre.domain import (
    ACTUAL_VALUE,
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
)
from newcalibre.engine import CommitReceipt, ForecastWrite, OriginCommit
from newcalibre.engine import ports as engine_ports
from newcalibre.engine.ports import SettlementSnapshot
from newcalibre.engine.ports.memory import (
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    InProcessDispatch,
)
from newcalibre.ledger import LedgerError, OrderRow

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
    ).for_keys(
        (
            ("a", pd.Timestamp("2026-01-01")),
            ("b", pd.Timestamp("2026-01-01")),
            ("a", pd.Timestamp("2026-01-02")),
        ),
        before=pd.Timestamp("2026-01-02"),
    )
    assert actuals == {
        ("a", pd.Timestamp("2026-01-01")): 1.0,
        ("b", pd.Timestamp("2026-01-01")): 3.0,
    }

    artifacts = InMemoryArtifactStore()
    artifacts.save("model:a", b"one")
    artifacts.save("model:a", b"one")
    assert artifacts.load("model:a") == b"one"
    with pytest.raises(ValueError, match="different bytes"):
        artifacts.save("model:a", b"two")

    states = InMemoryCalibrationStateStore()
    session = _session()
    states.save(session, "series:a", b"state", origin=ORIGIN_DATE)
    assert states.load(session, "series:a") == b"state"
    states.save(
        session,
        "series:a",
        b"stale",
        origin=pd.Timestamp("2026-01-04"),
    )
    assert states.load(session, "series:a") == b"state"
    with pytest.raises(ValueError, match="already holds different bytes"):
        states.save(session, "series:a", b"conflict", origin=ORIGIN_DATE)

    dispatch = InProcessDispatch()
    assert dispatch.map(lambda value: value * 2, (3, 1, 2)) == (6, 2, 4)


def test_actuals_source_requires_and_enforces_observation_semantics() -> None:
    panel = _panel()
    with pytest.raises(TypeError, match="actuals_semantics"):
        InMemoryActualsSource(panel)  # type: ignore[call-arg]

    censored_frame = panel.frame
    censored_frame[CENSOR_STATUS] = pd.Series(
        ["censored", "uncensored", "uncensored"],
        dtype="string",
    )
    censored_panel = Panel.from_frame(censored_frame, calendar=CALENDAR)

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


def test_ledger_sink_exposes_only_a_period_bound_compact_settlement_snapshot() -> None:
    session = _session()
    sink = InMemoryLedgerSink(session=session, calendar=CALENDAR)

    snapshot = sink.settlement_snapshot((ORIGIN_DATE,))

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
        )


def test_commit_receipt_validates_and_freezes_state_updates() -> None:
    state_updates = {"series:a": b"state"}
    receipt = CommitReceipt(
        session=_session(),
        origin=ORIGIN_DATE,
        digest="a" * 64,
        state_updates=state_updates,
    )
    state_updates["series:a"] = b"mutated"
    assert receipt.state_updates == {"series:a": b"state"}

    with pytest.raises(ValueError, match="non-empty trimmed string"):
        CommitReceipt(
            session=_session(),
            origin=ORIGIN_DATE,
            digest="a" * 64,
            state_updates={" untrimmed": b"state"},
        )
    with pytest.raises(TypeError, match="must contain bytes"):
        CommitReceipt(
            session=_session(),
            origin=ORIGIN_DATE,
            digest="a" * 64,
            state_updates={"series:a": "state"},  # type: ignore[dict-item]
        )
