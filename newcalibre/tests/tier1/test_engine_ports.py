"""Exercise the remaining engine ports and their in-memory adapters."""

from __future__ import annotations

import inspect

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
    InMemoryIndexedRunStore,
    InMemoryLedgerReader,
    InMemoryPanelSource,
    InProcessDispatch,
    OriginCommit,
    OriginIntent,
    Spine,
    TimeLoop,
)
from newcalibre.engine import ports as engine_ports
from newcalibre.engine.run_store import IndexedRunStore, OriginSnapshot, SettlementSnapshot
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
        target_support=TargetSupport.REAL,
    )


def _session(*, decisions: bool = True) -> SessionIdentity:
    keywords = (
        {
            "ordering_policy": {"name": "newsvendor"},
            "decision_series_keys": ("a", "b"),
            "cost_structure": CostStructure(1.0, 1.0, 1.0, 1.0),
            "decision_timing": DecisionTiming(lead_time=1, review_period=1),
            "stockout_rule": StockoutRule.LOST_SALES,
        }
        if decisions
        else {}
    )
    return SessionIdentity.derive(
        tenant="tenant-a",
        series_keys=("a", "b"),
        calendar=CALENDAR,
        horizon=2,
        model_config={"backend": "fixture", "name": "fixture"},
        **keywords,
    )


def _store(*, decisions: bool = True, actuals: Panel | None = None) -> InMemoryIndexedRunStore:
    return InMemoryIndexedRunStore(
        session=_session(decisions=decisions),
        calendar=CALENDAR,
        actuals=actuals,
        actuals_semantics=ActualsSemantics.DEMAND,
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


def test_in_memory_adapters_preserve_snapshots_and_order() -> None:
    """Keep panel loads defensive, actuals canonical, and dispatch deterministic."""
    panel = _panel()
    panel_source = InMemoryPanelSource(panel)
    mutated = panel_source.load().frame
    mutated.loc[:, OBSERVED_VALUE] = 99.0
    assert panel_source.load().frame[OBSERVED_VALUE].tolist() == [1.0, 2.0, 3.0]

    store = _store(actuals=panel)
    snapshot = store.open(OriginIntent(store.session, pd.Timestamp("2026-01-02")))
    assert isinstance(snapshot, OriginSnapshot)
    assert tuple((record.key, record.recorded_value) for record in snapshot.actuals.records) == (
        (("a", pd.Timestamp("2026-01-01")), 1.0),
        (("b", pd.Timestamp("2026-01-01")), 3.0),
    )
    assert InProcessDispatch().map(lambda value: value * 2, (3, 1, 2)) == (6, 2, 4)


def test_engine_port_namespace_contains_only_io_and_dispatch_seams() -> None:
    """Keep persistence represented solely by the separate two-method protocol."""
    protocols = {
        name
        for name, value in vars(engine_ports).items()
        if inspect.isclass(value)
        and value.__module__ == engine_ports.__name__
        and getattr(value, "_is_protocol", False)
    }
    assert protocols == {"PanelSource", "DispatchBackend"}
    assert set(IndexedRunStore.__dict__) & {"open", "commit"} == {"open", "commit"}


def test_reporting_adapter_is_absent_from_every_write_path() -> None:
    """Keep reporting read-only and outside engine/driver/store composition."""
    write_path_modules = {
        inspect.getmodule(value)
        for value in (Engine, Spine, TimeLoop, EventDriver, OriginCommit, CommitReceipt)
    }
    assert None not in write_path_modules
    for module in write_path_modules:
        source = inspect.getsource(module)
        assert "InMemoryLedgerReader" not in source
        assert "engine.reporting" not in source
    assert isinstance(InMemoryLedgerReader, type)


def test_store_exposes_only_a_period_bound_settlement_projection() -> None:
    """Return compact indexed settlement state without ledger row families."""
    store = _store()
    snapshot = store.settlement_snapshot((ORIGIN_DATE,))

    assert store.pending_observation_count == 0
    assert isinstance(snapshot, SettlementSnapshot)
    assert snapshot.periods == (ORIGIN_DATE,)
    assert snapshot.frontier is None
    assert snapshot.latest_positions == {}
    assert snapshot.open_order_quantities == {"a": 0.0, "b": 0.0}
    assert snapshot.due_arrivals == {}
    assert snapshot.actuals_semantics is None
    assert not any(hasattr(snapshot, name) for name in ("forecasts", "orders", "settlements"))


def test_store_refuses_initial_arrivals_without_decision_configuration() -> None:
    """Reject inventory facts when the session has no ordering domain."""
    session = _session(decisions=False)
    with pytest.raises(LedgerError, match="require a session decision configuration"):
        InMemoryIndexedRunStore(
            session=session,
            calendar=CALENDAR,
            actuals_semantics=ActualsSemantics.DEMAND,
            initial_arrivals={("a", ORIGIN_DATE): 1.0},
        )


def test_failed_origin_validation_publishes_no_partial_rows() -> None:
    """Validate all families before exposing any part of a transaction."""
    store = _store()
    key = ("a", ORIGIN_DATE, 1, "fixture")
    forecast = ForecastWrite(_forecast_frame(), {key: {}})
    order = OrderRow(
        session=store.session,
        series_key="a",
        origin=ORIGIN_DATE,
        model_name="fixture",
        quantity=1.0,
        arrival_period=pd.Timestamp("2026-01-06"),
    )

    with pytest.raises(LedgerError, match="forecast row origin must match"):
        store.commit(
            OriginCommit(
                session=store.session,
                origin=pd.Timestamp("2026-01-06"),
                expected_revision=store.revision,
                forecasts=(forecast,),
                orders=(order,),
            )
        )

    assert store.forecasts == ()
    assert store.orders == ()
    assert store.revision == 1


def test_forecast_digest_encodes_cells_and_dtypes_without_lossy_prehash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind SHA-256 directly to canonical schema and cell values."""

    def forbidden_prehash(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("forecast digest used pandas row prehashing")

    monkeypatch.setattr(pd.util, "hash_pandas_object", forbidden_prehash)
    integer = ForecastWrite(pd.DataFrame({"value": pd.Series([1], dtype="int64")}), {})
    floating = ForecastWrite(pd.DataFrame({"value": pd.Series([1.0], dtype="float64")}), {})
    changed = ForecastWrite(pd.DataFrame({"value": pd.Series([2], dtype="int64")}), {})

    assert len({integer.digest, floating.digest, changed.digest}) == 3


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64])
def test_commit_receipt_rejects_noncanonical_digests(digest: str) -> None:
    """Require canonical content identities on every durable receipt."""
    with pytest.raises(ValueError, match="SHA-256 hex string"):
        CommitReceipt(
            session=_session(),
            origin=ORIGIN_DATE,
            digest=digest,
            expected_revision=1,
            revision=2,
            state_updates={},
        )
