from __future__ import annotations

import pandas as pd

from calibre.api.lifecycle import FitRecord, LifecycleStore, SqlLifecycleStore
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.run_status import RunStatus
from calibre.storage.models import Base
from calibre.storage.postgres import make_engine, make_session_factory


def _store(db_url: str) -> SqlLifecycleStore:
    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    return SqlLifecycleStore(make_session_factory(engine))


def _fit_record() -> FitRecord:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            DS: pd.to_datetime(["2024-01-01", "2024-01-08"]),
            Y: [1.0, 2.0],
        }
    )
    future_x = pd.DataFrame({UNIQUE_ID: ["A"], DS: [pd.Timestamp("2024-01-15")], "promo": [1]})
    return FitRecord(
        fit_id=LifecycleStore.new_fit_id(),
        session_id="session-a",
        tenant="tenant",
        sku_set=["A"],
        forecaster_config={"backend": "stub", "model": "stub_model"},
        horizon=2,
        freq="W",
        history=history,
        future_x=future_x,
        conformal_config={"method": "aci", "coverage": 0.9},
    )


def test_sql_lifecycle_store_survives_reconnect(tmp_path) -> None:
    db_url = f"sqlite+pysqlite:///{(tmp_path / 'lifecycle.db').as_posix()}"
    first = _store(db_url)
    record = _fit_record()
    first.put_fit(record)
    first.upsert_conformal_state(record.session_id, {"partition": {"score_history": [1.0]}})

    second = _store(db_url)
    loaded = second.first_fit_for_session(record.session_id)

    assert loaded is not None
    assert loaded.fit_id == record.fit_id
    assert loaded.history[DS].tolist() == record.history[DS].tolist()
    assert second.get_conformal_state(record.session_id) == {"partition": {"score_history": [1.0]}}


def test_sql_lifecycle_store_fit_round_trip(tmp_path) -> None:
    db_url = f"sqlite+pysqlite:///{(tmp_path / 'lifecycle.db').as_posix()}"
    store = _store(db_url)
    record = _fit_record()
    store.put_fit(record)
    forecast = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            DS: pd.to_datetime(["2024-01-15", "2024-01-22"]),
            Y_HAT: [3.0, 4.0],
            H: [1, 2],
        }
    )

    store.update_fit(
        record.fit_id,
        status=RunStatus.SUCCEEDED,
        artifact_urls={"model": "file:///tmp/model.bin"},
        last_forecast=forecast,
    )
    loaded = store.get_fit(record.fit_id)

    assert loaded is not None
    assert loaded.status == RunStatus.SUCCEEDED
    assert loaded.artifact_urls == {"model": "file:///tmp/model.bin"}
    pd.testing.assert_frame_equal(loaded.last_forecast, forecast)
