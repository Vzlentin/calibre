from __future__ import annotations

import pandas as pd
import pytest

from calibre.api.lifecycle import (
    FitRecord,
    MemoryLifecycleStore,
    SqlLifecycleStore,
    TuneRecord,
)
from calibre.core.run_status import RunStatus
from calibre.storage.models import Base
from calibre.storage.postgres import make_engine, make_session_factory


def _make_sql_store() -> SqlLifecycleStore:
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    return SqlLifecycleStore(factory)


def _sample_fit_record(fit_id: str = "f1", session_id: str = "s1") -> FitRecord:
    return FitRecord(
        fit_id=fit_id,
        session_id=session_id,
        tenant="t",
        sku_set=["sku-a"],
        forecaster_config={"backend": "stub"},
        horizon=2,
        freq="D",
        history=pd.DataFrame(
            {
                "unique_id": ["sku-a", "sku-a"],
                "ds": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "y": [1.0, 2.0],
            }
        ),
        future_x=None,
        conformal_config=None,
        status=RunStatus.QUEUED,
    )


@pytest.mark.parametrize("make_store", [MemoryLifecycleStore, _make_sql_store])
def test_fit_round_trip(make_store):
    store = make_store()
    record = _sample_fit_record()
    store.put_fit(record)

    fetched = store.get_fit("f1")
    assert fetched is not None
    assert fetched.fit_id == "f1"
    assert fetched.session_id == "s1"
    assert fetched.sku_set == ["sku-a"]
    assert fetched.horizon == 2


@pytest.mark.parametrize("make_store", [MemoryLifecycleStore, _make_sql_store])
def test_update_fit_status(make_store):
    store = make_store()
    store.put_fit(_sample_fit_record())
    updated = store.update_fit("f1", status=RunStatus.SUCCEEDED, error=None)
    assert updated.status == RunStatus.SUCCEEDED

    fetched = store.get_fit("f1")
    assert fetched is not None
    assert fetched.status == RunStatus.SUCCEEDED


@pytest.mark.parametrize("make_store", [MemoryLifecycleStore, _make_sql_store])
def test_first_fit_for_session(make_store):
    store = make_store()
    store.put_fit(_sample_fit_record())
    result = store.first_fit_for_session("s1")
    assert result is not None
    assert result.fit_id == "f1"
    assert store.first_fit_for_session("unknown") is None


@pytest.mark.parametrize("make_store", [MemoryLifecycleStore, _make_sql_store])
def test_conformal_state_round_trip(make_store):
    store = make_store()
    store.put_fit(_sample_fit_record())

    states = {"partition-a": {"some": "data"}}
    store.upsert_conformal_state("s1", states)
    fetched = store.get_conformal_state("s1")
    assert "partition-a" in fetched
    assert fetched["partition-a"]["some"] == "data"


@pytest.mark.parametrize("make_store", [MemoryLifecycleStore, _make_sql_store])
def test_tune_record_round_trip(make_store):
    store = make_store()
    record = TuneRecord(
        study_id="study-1",
        session_id="s1",
        tenant="t",
        sku_set=["sku-a"],
    )
    store.put_study(record)
    fetched = store.get_study("study-1")
    assert fetched is not None
    assert fetched.study_id == "study-1"


def test_sql_lifecycle_store_survives_reconnect():
    """State persists across SqlLifecycleStore instances sharing the same engine."""
    engine = make_engine("sqlite:///file::memory:?cache=shared&uri=true")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)

    store1 = SqlLifecycleStore(factory)
    store1.put_fit(_sample_fit_record())
    store1.update_fit("f1", status=RunStatus.SUCCEEDED)

    store2 = SqlLifecycleStore(factory)
    fetched = store2.get_fit("f1")
    assert fetched is not None
    assert fetched.status == RunStatus.SUCCEEDED
