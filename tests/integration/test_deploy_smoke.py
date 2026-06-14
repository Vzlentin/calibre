"""Deploy smoke gate.

Proves the service is *deployable*: the schema applies via ``alembic upgrade
head`` (not ``create_all()``), the app boots wired to that migrated database,
and the core lifecycle — ``/fit`` → ``/predict`` → ``/calibrate`` → ``/order``
→ ``/observe`` — serves end to end with a real forecasting model.

PR #38 was certified "all green" by a suite that never ran the migration path
and never booted the app against a real database. This gate closes that hole;
as the SQL-backed lifecycle store lands, the same flow automatically exercises
the migrated lifecycle tables.
"""

from __future__ import annotations

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from calibre.api.main import create_app
from calibre.api.run_store import SqlRunStore
from calibre.core.forecast_frame import UNIQUE_ID, interval_column_names


@pytest.fixture(params=["memory", "sql"])
def migrated_app(request, fresh_db_url, tmp_path, monkeypatch):
    """App constructed against a freshly migrated database, with clean stores.

    Parametrized over the lifecycle backend so the full roundtrip runs against
    both the in-memory store and the SQL store (frames -> parquet on a migrated
    DB) — the SQL path is the one PR #38 broke. The env is set first so the
    factory resolves the run store, db factory, and (env-selected) lifecycle
    store at construction time — no module globals to reset.
    """
    monkeypatch.setenv("CALIBRE_DATABASE_URL", fresh_db_url)
    if request.param == "sql":
        monkeypatch.setenv("LIFECYCLE_STORE", "sql")
        monkeypatch.setenv("CALIBRE_ARTIFACT_URI", str(tmp_path / "artifacts"))
    command.upgrade(Config("alembic.ini"), "head")
    return create_app()


@pytest.fixture
def migrated_client(migrated_app):
    return TestClient(migrated_app)


def _history_records(uid: str = "A") -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=8, freq="W-SUN")
    return [
        {UNIQUE_ID: uid, "ds": ds.strftime("%Y-%m-%d"), "y": float(idx + 1)}
        for idx, ds in enumerate(dates)
    ]


@pytest.fixture
def sales_uri(tmp_path):
    path = tmp_path / "sales.parquet"
    pd.DataFrame(_history_records("A")).to_parquet(path)
    return str(path)


def _fit_payload(sales_uri: str) -> dict:
    return {
        "tenant": "acme",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "sales_uri": sales_uri,
        "forecaster_config": {"backend": "statsforecast", "model": "Naive"},
        "conformal_config": {
            "method": "aci",
            "coverage": 0.9,
            "calibration_window": 4,
            "gamma": 0.05,
        },
    }


def test_app_wired_to_migrated_db(migrated_app) -> None:
    """With CALIBRE_DATABASE_URL set, the app selects the SQL-backed store."""
    client = TestClient(migrated_app)
    assert client.get("/healthz").json() == {"status": "ok"}
    assert isinstance(migrated_app.state.stores.run_store, SqlRunStore)


def test_create_app_fails_fast_on_sql_store_without_db_url(monkeypatch) -> None:
    """LIFECYCLE_STORE=sql with no DB URL fails fast at construction.

    Such a service can never serve, so construction raises immediately
    (deploy-time fail-fast) instead of booting a service that 500s on every
    lifecycle request.
    """
    monkeypatch.delenv("CALIBRE_DATABASE_URL", raising=False)
    monkeypatch.setenv("LIFECYCLE_STORE", "sql")

    with pytest.raises(RuntimeError, match="CALIBRE_DATABASE_URL"):
        create_app()


def test_create_app_with_well_formed_db_url_constructs_lazily(tmp_path, monkeypatch) -> None:
    """A well-formed URL constructs without touching the database.

    Engines never connect at build time, so reachability failures keep landing on
    first store use; the pre-refactor failure timing is preserved. Proven
    driver-free: the sqlite file must NOT exist after construction.
    """
    db_path = tmp_path / "never-touched.db"
    monkeypatch.setenv("CALIBRE_DATABASE_URL", f"sqlite+pysqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("LIFECYCLE_STORE", "sql")

    app = create_app()

    assert TestClient(app).get("/healthz").json() == {"status": "ok"}
    assert not db_path.exists(), "construction must not open a database connection"


def test_lifecycle_roundtrip_on_migrated_db(migrated_app, sales_uri) -> None:
    client = TestClient(migrated_app)

    fit = client.post("/fit", json=_fit_payload(sales_uri))
    assert fit.status_code == 202, fit.text
    fit_id = fit.json()["fit_id"]
    session_id = fit.json()["session_id"]
    assert client.get(f"/fits/{fit_id}").json()["status"] == "succeeded"

    predict = client.post("/predict", json={"fit_id": fit_id, "origin": "2024-02-04"})
    assert predict.status_code == 200, predict.text
    forecast = predict.json()["forecast"]
    assert len(forecast) == 2

    calibrate = client.post("/calibrate", json={"session_id": session_id, "forecast": forecast})
    assert calibrate.status_code == 200, calibrate.text
    calibrated = calibrate.json()["calibrated"]
    lower_col, upper_col = interval_column_names(0.9)
    assert all(lower_col in row and upper_col in row for row in calibrated)

    order = client.post(
        "/order",
        json={
            "calibrated": calibrated,
            "session_id": session_id,
            "ordering": {
                "policy": "rs",
                "coverage": 0.9,
                "params": [
                    {
                        "unique_id": "A",
                        "inventory_position": 5.0,
                        "lead_time": 1,
                        "review_period": 1,
                    }
                ],
            },
        },
    )
    assert order.status_code == 200, order.text

    # Orders are now durable rows; reading them back via /sessions exercises the
    # migrated ``orders`` table write+read path (the SQL store for the sql param).
    session_state = client.get("/sessions/acme/A")
    assert session_state.status_code == 200, session_state.text
    open_orders = session_state.json()["open_orders"]
    assert open_orders is not None and len(open_orders) == 1
    assert open_orders[0][UNIQUE_ID] == "A"

    observe = client.post(
        "/observe",
        json={
            "session_id": session_id,
            "actuals": [
                {UNIQUE_ID: "A", "ds": "2024-02-11", "y": 9.0},
                {UNIQUE_ID: "A", "ds": "2024-02-18", "y": 11.0},
            ],
        },
    )
    assert observe.status_code == 202, observe.text
    assert migrated_app.state.stores.lifecycle_store.get_conformal_state(session_id), (
        "observe should persist conformal state"
    )
