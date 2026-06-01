from __future__ import annotations

import optuna
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api import tuning_service
from calibre.api.lifecycle import LifecycleStore
from calibre.api.main import app
from calibre.api.tuning_service import (
    register_tuning_objective,
    register_tuning_search_space,
)
from calibre.core.forecast_frame import UNIQUE_ID
from calibre.evaluation.point_metrics import smape
from calibre.tuning import Accuracy, LocalTuningTask, TuningCandidate


def _seasonal_search_space(trial: optuna.Trial) -> TuningCandidate:
    return TuningCandidate(
        model_config={
            "season_length": trial.suggest_categorical("season_length", [4, 13, 26, 52]),
        },
        conformal_config={"gamma": trial.suggest_float("gamma", 0.01, 0.1)},
    )


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    fresh = LifecycleStore()
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", fresh)
    monkeypatch.setattr(tuning_service, "_SEARCH_SPACES", {})
    monkeypatch.setattr(tuning_service, "_OBJECTIVES", {})
    return fresh


@pytest.fixture
def client():
    return TestClient(app)


def _history_records(uid: str = "A") -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=8, freq="W-SUN")
    return [
        {UNIQUE_ID: uid, "ds": ds.strftime("%Y-%m-%d"), "y": float(idx + 1)}
        for idx, ds in enumerate(dates)
    ]


@pytest.fixture
def tune_payload(tmp_path):
    """Stage sales + actuals parquet once; return a payload builder (URI ingress)."""
    sales_path = tmp_path / "sales.parquet"
    actuals_path = tmp_path / "actuals.parquet"
    pd.DataFrame(_history_records("A")).to_parquet(sales_path)
    actuals = pd.DataFrame(_history_records("A"))
    actuals["ds"] = pd.to_datetime(actuals["ds"])
    actuals.to_parquet(actuals_path)

    def _build(**overrides) -> dict:
        base = {
            "tenant": "acme",
            "sku_set": ["A"],
            "horizon": 2,
            "freq": "W-SUN",
            "sales_uri": str(sales_path),
            "actuals_uri": str(actuals_path),
            "origins": ["2024-02-04"],
            "base_model_config": {"backend": "stub", "model": "stub_model"},
            "search_space_id": "seasonal",
            "objective_id": "accuracy",
            "n_trials": 3,
            "conformal_config": {
                "method": "aci",
                "coverage": 0.9,
                "calibration_window": 4,
                "gamma": 0.05,
            },
        }
        base.update(overrides)
        return base

    return _build


def test_tune_rejects_unknown_search_space(client, tune_payload) -> None:
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    response = client.post("/tune", json=tune_payload())
    assert response.status_code == 400
    assert "search_space_id" in response.json()["detail"]


def test_tune_rejects_unknown_objective(client, tune_payload) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    response = client.post("/tune", json=tune_payload())
    assert response.status_code == 400
    assert "objective_id" in response.json()["detail"]


def test_tune_rejects_global_hpo_with_local_model(client, tune_payload) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    payload = tune_payload(hpo_scope="global")
    response = client.post("/tune", json=payload)
    assert response.status_code == 422
    assert "scope" in response.json()["detail"]


def test_tune_rejects_local_hpo_with_global_model(client, tune_payload) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    payload = tune_payload(
        base_model_config={"backend": "stub", "model": "stub_model", "scope": "global"}
    )
    response = client.post("/tune", json=payload)
    assert response.status_code == 422
    assert "scope" in response.json()["detail"]


def test_tune_endpoint_persists_best_candidate(monkeypatch, client, tune_payload) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))

    captured: dict[str, LocalTuningTask] = {}

    def _fake_optimize(task: LocalTuningTask) -> TuningCandidate:
        captured["task"] = task
        return TuningCandidate(
            model_config={"backend": "stub", "model": "stub_model", "season_length": 26},
            conformal_config={"gamma": 0.07},
            ordering_config={},
        )

    monkeypatch.setattr(tuning_service, "optimize_local_task_candidate", _fake_optimize)

    submit = client.post("/tune", json=tune_payload())
    assert submit.status_code == 202, submit.text
    handle = submit.json()
    study_id = handle["study_id"]
    assert handle["status"] == "queued"
    assert handle["session_id"]

    detail = client.get(f"/studies/{study_id}")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["status"] == "succeeded"
    assert body["best_candidates"]["A"]["model_config_values"]["season_length"] == 26
    assert body["best_candidates"]["A"]["conformal_config"]["gamma"] == 0.07
    assert body["best_candidates"]["A"]["ordering_config"] == {}
    assert body["sku_set"] == ["A"]

    assert captured["task"].unique_id == "A"
    assert captured["task"].horizon == 2
    assert captured["task"].study_config.n_trials == 3


def test_get_study_returns_404_for_unknown_id(client) -> None:
    response = client.get("/studies/does-not-exist")
    assert response.status_code == 404


def test_tune_rejects_empty_origins(client, tune_payload) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    response = client.post("/tune", json=tune_payload(origins=[]))
    assert response.status_code == 400
    assert "origins" in response.json()["detail"]


def test_tune_handle_returns_deterministic_session_id(monkeypatch, client, tune_payload) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    monkeypatch.setattr(
        tuning_service,
        "optimize_local_task_candidate",
        lambda task: TuningCandidate(model_config={}),
    )

    first = client.post("/tune", json=tune_payload()).json()
    second = client.post("/tune", json=tune_payload()).json()
    assert first["session_id"] == second["session_id"]
    assert first["study_id"] != second["study_id"]


def test_failed_study_records_error(monkeypatch, client, tune_payload) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))

    def _boom(task: LocalTuningTask) -> TuningCandidate:
        raise RuntimeError("optuna exploded")

    monkeypatch.setattr(tuning_service, "optimize_local_task_candidate", _boom)

    submit = client.post("/tune", json=tune_payload())
    study_id = submit.json()["study_id"]
    detail = client.get(f"/studies/{study_id}").json()

    assert detail["status"] == "failed"
    assert "optuna exploded" in detail["error"]
    assert detail["best_candidates"] == {}
