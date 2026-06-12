"""SqlLifecycleStore round-trip + restart survival (roadmap P0.2).

The in-memory store loses everything on restart and is invisible across
workers. These tests assert the SQL store persists fit/tune records, frames
(as parquet by reference), and session-owned conformal state across a simulated
restart — a brand-new store/engine on the same database + artifact base.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from calibre.api.lifecycle import FitRecord, LifecycleStore, OrderKey, TuneRecord, order_pk
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, Y
from calibre.core.run_status import RunStatus
from calibre.storage.lifecycle_repo import SqlLifecycleStore
from calibre.storage.models import Base, LifecycleFitRecord
from calibre.storage.postgres import make_engine, make_session_factory, session_scope


def _stores(tmp_path):
    """A factory returning fresh stores over the same db + artifact base."""
    db_url = f"sqlite+pysqlite:///{(tmp_path / 'lc.db').as_posix()}"
    artifact_base = str(tmp_path / "artifacts")
    Base.metadata.create_all(make_engine(db_url))

    def make() -> SqlLifecycleStore:
        return SqlLifecycleStore(make_session_factory(make_engine(db_url)), artifact_base)

    return make


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "A"],
            DS: pd.to_datetime(["2024-01-07", "2024-01-14", "2024-01-21"]),
            Y: [1.0, 2.0, 3.0],
        }
    )


def _fit_record() -> FitRecord:
    return FitRecord(
        fit_id="fit-1",
        session_id="sess-1",
        tenant="acme",
        sku_set=["A"],
        forecaster_config={"backend": "statsforecast", "model": "Naive"},
        horizon=2,
        freq="W-SUN",
        history=_history(),
        future_x=None,
        conformal_config={"method": "aci", "coverage": 0.9},
        status=RunStatus.QUEUED,
    )


def test_fit_survives_restart_with_frames(tmp_path):
    make = _stores(tmp_path)
    make().put_fit(_fit_record())

    # Restart: brand-new store/engine on the same db + artifact base.
    loaded = make().get_fit("fit-1")
    assert loaded is not None
    assert loaded.tenant == "acme"
    assert loaded.sku_set == ["A"]
    assert loaded.conformal_config == {"method": "aci", "coverage": 0.9}
    assert loaded.status == RunStatus.QUEUED
    assert loaded.future_x is None  # absent frame stays absent
    pd.testing.assert_frame_equal(loaded.history.reset_index(drop=True), _history())


def test_update_fit_persists_status_and_new_frame(tmp_path):
    make = _stores(tmp_path)
    make().put_fit(_fit_record())

    forecast = pd.DataFrame({UNIQUE_ID: ["A"], DS: pd.to_datetime(["2024-01-28"]), Y_HAT: [4.0]})
    make().update_fit("fit-1", status=RunStatus.SUCCEEDED, last_forecast=forecast)

    loaded = make().get_fit("fit-1")
    assert loaded is not None
    assert loaded.status == RunStatus.SUCCEEDED
    assert loaded.last_forecast is not None
    pd.testing.assert_frame_equal(loaded.last_forecast.reset_index(drop=True), forecast)


def test_conformal_state_is_session_owned_and_survives_restart(tmp_path):
    make = _stores(tmp_path)
    make().upsert_conformal_state("sess-1", {"p1": {"score": 1.0}, "p2": {"score": 2.0}})

    assert make().get_conformal_state("sess-1") == {
        "p1": {"score": 1.0},
        "p2": {"score": 2.0},
    }

    # Upsert overwrites an existing partition, leaves others.
    make().upsert_conformal_state("sess-1", {"p1": {"score": 9.0}})
    assert make().get_conformal_state("sess-1") == {
        "p1": {"score": 9.0},
        "p2": {"score": 2.0},
    }


def test_lookups_by_session_and_tenant(tmp_path):
    make = _stores(tmp_path)
    store = make()
    store.put_fit(_fit_record())

    assert {r.fit_id for r in make().fits_for_session("sess-1")} == {"fit-1"}
    assert make().last_fit_for_session("sess-1").fit_id == "fit-1"
    assert make().last_fit_for_session("missing") is None
    assert {r.fit_id for r in make().fits_for_tenant_uid("acme", "A")} == {"fit-1"}
    assert make().fits_for_tenant_uid("acme", "ZZZ") == []


def test_study_round_trip_and_update(tmp_path):
    make = _stores(tmp_path)
    make().put_study(
        TuneRecord(
            study_id="study-1",
            session_id="sess-1",
            tenant="acme",
            sku_set=["A"],
            status=RunStatus.QUEUED,
        )
    )
    make().update_study(
        "study-1",
        status=RunStatus.SUCCEEDED,
        best_candidates={"A": {"model_config": {"season_length": 4}}},
    )

    loaded = make().get_study("study-1")
    assert loaded is not None
    assert loaded.status == RunStatus.SUCCEEDED
    assert loaded.best_candidates == {"A": {"model_config": {"season_length": 4}}}


def test_last_fit_uses_creation_order_not_fit_id(tmp_path):
    """A session can hold several fits; "last" must be the most recently created,
    not the lexicographically largest (random) fit_id."""
    db_url = f"sqlite+pysqlite:///{(tmp_path / 'lc.db').as_posix()}"
    Base.metadata.create_all(make_engine(db_url))
    factory = make_session_factory(make_engine(db_url))

    earlier = datetime(2024, 1, 1, tzinfo=UTC)
    # The later-created fit deliberately has the SMALLER fit_id, so ordering by
    # fit_id alone would wrongly pick the earlier one.
    with session_scope(factory) as session:
        for fit_id, created in [("zzz", earlier), ("aaa", earlier + timedelta(hours=1))]:
            session.add(
                LifecycleFitRecord(
                    fit_id=fit_id,
                    session_id="s",
                    tenant="t",
                    sku_set=["A"],
                    forecaster_config={},
                    horizon=1,
                    freq="W",
                    conformal_config=None,
                    status="queued",
                    artifact_urls={},
                    frame_uris={},
                    created_at=created,
                )
            )

    store = SqlLifecycleStore(make_session_factory(make_engine(db_url)), str(tmp_path / "art"))
    last = store.last_fit_for_session("s")
    assert last is not None
    assert last.fit_id == "aaa", "last_fit should be the most recently created, not max fit_id"


def test_last_fit_sql_breaks_created_at_ties_by_fit_id(tmp_path):
    """On equal created_at, the SQL store breaks the tie by fit_id desc, so the
    selection is deterministic rather than insertion-arbitrary."""
    db_url = f"sqlite+pysqlite:///{(tmp_path / 'lc.db').as_posix()}"
    Base.metadata.create_all(make_engine(db_url))
    factory = make_session_factory(make_engine(db_url))

    same = datetime(2024, 1, 1, tzinfo=UTC)
    with session_scope(factory) as session:
        for fit_id in ("aaa", "zzz"):
            session.add(
                LifecycleFitRecord(
                    fit_id=fit_id,
                    session_id="s",
                    tenant="t",
                    sku_set=["A"],
                    forecaster_config={},
                    horizon=1,
                    freq="W",
                    conformal_config=None,
                    status="queued",
                    artifact_urls={},
                    frame_uris={},
                    created_at=same,
                )
            )

    store = SqlLifecycleStore(make_session_factory(make_engine(db_url)), str(tmp_path / "art"))
    last = store.last_fit_for_session("s")
    assert last is not None
    assert last.fit_id == "zzz", "created_at ties break by fit_id desc"


def test_last_fit_in_memory_is_last_by_insertion_order():
    """The in-memory store selects the most recently put fit for a session,
    parity with the SQL store's most-recent selection."""
    store = LifecycleStore()
    for fit_id in ("first", "second", "third"):
        record = _fit_record()
        record.fit_id = fit_id
        record.session_id = "sess-1"
        store.put_fit(record)

    last = store.last_fit_for_session("sess-1")
    assert last is not None
    assert last.fit_id == "third"
    assert store.last_fit_for_session("missing") is None


def _orders_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            FORECAST_ORIGIN: pd.to_datetime(["2024-02-04", "2024-02-11"]),
            MODEL_NAME: ["Naive", "Naive"],
            "order_qty": [5.0, 7.0],
            "target_stock_level": [10.0, 12.0],
        }
    )


def _assert_orders_round_trip(store) -> None:
    store.put_orders("acme", "sess-1", _orders_frame())

    loaded = store.orders_for_tenant_uid("acme", "A")
    assert len(loaded) == 2
    assert sorted(loaded["order_qty"].tolist()) == [5.0, 7.0]
    assert loaded["target_stock_level"].tolist() == [10.0, 12.0]
    assert pd.api.types.is_datetime64_any_dtype(loaded[FORECAST_ORIGIN])

    # Tenant- and uid-scoped: a different tenant or unknown uid sees nothing.
    assert store.orders_for_tenant_uid("acme", "ZZZ").empty
    assert store.orders_for_tenant_uid("other", "A").empty

    # Re-placing the same decision tuple upserts (no duplicate row).
    revised = _orders_frame()
    revised.loc[0, "order_qty"] = 9.0
    store.put_orders("acme", "sess-1", revised)
    reloaded = store.orders_for_tenant_uid("acme", "A")
    assert len(reloaded) == 2
    assert sorted(reloaded["order_qty"].tolist()) == [7.0, 9.0]


def test_orders_round_trip_in_memory():
    _assert_orders_round_trip(LifecycleStore())


def test_orders_round_trip_sql(tmp_path):
    make = _stores(tmp_path)
    _assert_orders_round_trip(make())


def test_order_pk_builds_typed_key():
    key = order_pk(
        "sess-1",
        {UNIQUE_ID: "A", FORECAST_ORIGIN: "2024-02-04", MODEL_NAME: "Naive"},
    )
    assert key == OrderKey("sess-1", "A", "2024-02-04", "Naive")
    # model_name stays optional, defaulting to "".
    assert order_pk("sess-1", {UNIQUE_ID: "A", FORECAST_ORIGIN: "2024-02-04"}).model_name == ""


@pytest.mark.parametrize(
    "record",
    [
        {UNIQUE_ID: "A"},  # forecast_origin missing
        {FORECAST_ORIGIN: "2024-02-04"},  # unique_id missing
        {UNIQUE_ID: "A", FORECAST_ORIGIN: ""},  # forecast_origin empty
    ],
)
def test_order_pk_rejects_missing_key_fields(record):
    # Both stores must reject the same malformed input rather than silently
    # keying a row on "" (m4 fail-fast).
    with pytest.raises(ValueError, match="unique_id and forecast_origin"):
        order_pk("sess-1", record)
