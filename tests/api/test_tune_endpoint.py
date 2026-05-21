from __future__ import annotations

import optuna
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api.lifecycle import LifecycleStore
from calibre.api.main import (
    app,
    register_tuning_objective,
    register_tuning_search_space,
)
from calibre.core.forecast_frame import UNIQUE_ID
from calibre.evaluation.point_metrics import smape
from calibre.tuning import Accuracy, TuningCandidate, TuningTask


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
    monkeypatch.setattr(api_main, "_lifecycle_store", lambda: fresh)
    monkeypatch.setattr(api_main, "_SEARCH_SPACES", {})
    monkeypatch.setattr(api_main, "_OBJECTIVES", {})
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


def _tune_payload() -> dict:
    return {
        "tenant": "acme",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "history": _history_records("A"),
        "actuals": _history_records("A"),
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


def test_tune_rejects_unknown_search_space(client) -> None:
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    response = client.post("/tune", json=_tune_payload())
    assert response.status_code == 400
    assert "search_space_id" in response.json()["detail"]


def test_tune_rejects_unknown_objective(client) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    response = client.post("/tune", json=_tune_payload())
    assert response.status_code == 400
    assert "objective_id" in response.json()["detail"]


def test_tune_endpoint_persists_best_candidate(monkeypatch, client) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))

    captured: dict[str, TuningTask] = {}

    def _fake_optimize(task: TuningTask) -> TuningCandidate:
        captured["task"] = task
        return TuningCandidate(
            model_config={"backend": "stub", "model": "stub_model", "season_length": 26},
            conformal_config={"gamma": 0.07},
            ordering_config={},
        )

    monkeypatch.setattr(api_main, "optimize_task_candidate", _fake_optimize)

    submit = client.post("/tune", json=_tune_payload())
    assert submit.status_code == 202, submit.text
    handle = submit.json()
    study_id = handle["study_id"]
    assert handle["status"] == "queued"
    assert handle["session_id"]

    detail = client.get(f"/studies/{study_id}")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["status"] == "succeeded"
    assert body["best_candidate"]["model_config_values"]["season_length"] == 26
    assert body["best_candidate"]["conformal_config"]["gamma"] == 0.07
    assert body["best_candidate"]["ordering_config"] == {}
    assert body["sku_set"] == ["A"]

    assert captured["task"].unique_id == "A"
    assert captured["task"].horizon == 2
    assert captured["task"].n_trials == 3


def test_get_study_returns_404_for_unknown_id(client) -> None:
    response = client.get("/studies/does-not-exist")
    assert response.status_code == 404


def test_tune_rejects_empty_origins(client) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    payload = _tune_payload()
    payload["origins"] = []
    response = client.post("/tune", json=payload)
    assert response.status_code == 400
    assert "origins" in response.json()["detail"]


def test_tune_handle_returns_deterministic_session_id(monkeypatch, client) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    monkeypatch.setattr(
        api_main,
        "optimize_task_candidate",
        lambda task: TuningCandidate(model_config={}),
    )

    first = client.post("/tune", json=_tune_payload()).json()
    second = client.post("/tune", json=_tune_payload()).json()
    assert first["session_id"] == second["session_id"]
    assert first["study_id"] != second["study_id"]


def test_failed_study_records_error(monkeypatch, client) -> None:
    register_tuning_search_space("seasonal", _seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))

    def _boom(task: TuningTask) -> TuningCandidate:
        raise RuntimeError("optuna exploded")

    monkeypatch.setattr(api_main, "optimize_task_candidate", _boom)

    submit = client.post("/tune", json=_tune_payload())
    study_id = submit.json()["study_id"]
    detail = client.get(f"/studies/{study_id}").json()

    assert detail["status"] == "failed"
    assert "optuna exploded" in detail["error"]
    assert detail["best_candidate"] is None
