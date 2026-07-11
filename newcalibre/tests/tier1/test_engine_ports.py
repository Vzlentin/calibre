"""Exercise all six engine ports through their in-memory adapters."""

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
    Calendar,
    Panel,
    SessionIdentity,
)
from newcalibre.engine import ForecastWrite, OriginCommit
from newcalibre.engine import ports as engine_ports
from newcalibre.engine.ports.memory import (
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    InProcessDispatch,
)
from newcalibre.ledger import Ledger, LedgerError, OrderRow

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
        horizon=1,
        model_config={"backend": "fixture", "name": "fixture"},
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

    actuals = InMemoryActualsSource(panel).for_keys(
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
    states.save(session, "series:a", b"state")
    assert states.load(session, "series:a") == b"state"

    dispatch = InProcessDispatch()
    assert dispatch.map(lambda value: value * 2, (3, 1, 2)) == (6, 2, 4)


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


def test_ledger_sink_rejects_a_partial_cross_family_commit() -> None:
    session = _session()
    sink = InMemoryLedgerSink(Ledger(session=session, calendar=CALENDAR))
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
    with pytest.raises(LedgerError, match="duplicate forecast key"):
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
