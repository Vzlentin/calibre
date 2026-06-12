from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import tuning_service
from calibre.api.lifecycle import LifecycleStore
from calibre.api.main import create_app
from calibre.api.tuning_service import (
    register_tuning_objective,
    register_tuning_search_space,
)
from calibre.evaluation.point_metrics import smape
from calibre.tuning import Accuracy, LocalTuningTask, TuningCandidate


@pytest.fixture(autouse=True)
def _reset_registries(monkeypatch):
    monkeypatch.setattr(tuning_service, "_SEARCH_SPACES", {})
    monkeypatch.setattr(tuning_service, "_OBJECTIVES", {})


@pytest.fixture
def client():
    return TestClient(create_app(lifecycle_store=LifecycleStore()))


@pytest.fixture
def tune_payload(tmp_path, tuning_history_records):
    """Stage sales + actuals parquet once; return a payload builder (URI ingress)."""
    sales_path = tmp_path / "sales.parquet"
    actuals_path = tmp_path / "actuals.parquet"
    pd.DataFrame(tuning_history_records(["A"])).to_parquet(sales_path)
    actuals = pd.DataFrame(tuning_history_records(["A"]))
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


def test_tune_rejects_unknown_objective(client, tune_payload, seasonal_search_space) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
    response = client.post("/tune", json=tune_payload())
    assert response.status_code == 400
    assert "objective_id" in response.json()["detail"]


def test_tune_rejects_global_hpo_with_local_model(
    client, tune_payload, seasonal_search_space
) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    payload = tune_payload(hpo_scope="global")
    response = client.post("/tune", json=payload)
    assert response.status_code == 422
    assert "scope" in response.json()["detail"]


def test_tune_rejects_local_hpo_with_global_model(
    client, tune_payload, seasonal_search_space
) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    payload = tune_payload(
        base_model_config={"backend": "stub", "model": "stub_model", "scope": "global"}
    )
    response = client.post("/tune", json=payload)
    assert response.status_code == 422
    assert "scope" in response.json()["detail"]


def test_tune_endpoint_persists_best_candidate(
    monkeypatch, client, tune_payload, seasonal_search_space
) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))

    # Capture the task the optimizer was handed and the candidate it produced,
    # then assert the *captured* candidate is what surfaces in the study — the
    # request->persistence->response plumbing — rather than re-typing the literal
    # season_length the fake happened to echo back.
    captured: dict[str, object] = {}

    def _fake_optimize(task: LocalTuningTask) -> TuningCandidate:
        captured["task"] = task
        candidate = TuningCandidate(
            model_config={"backend": "stub", "model": "stub_model", "season_length": 26},
            conformal_config={"gamma": 0.07},
            ordering_config={},
        )
        captured["candidate"] = candidate
        return candidate

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
    assert body["sku_set"] == ["A"]

    # The optimizer ran against the right per-uid task derived from the request.
    task = captured["task"]
    assert task.unique_id == "A"
    assert task.horizon == 2
    assert task.study_config.n_trials == 3
    assert set(task.history["unique_id"].unique()) == {"A"}

    # The candidate computed for uid "A" is the one persisted under uid "A".
    produced = captured["candidate"]
    stored = body["best_candidates"]["A"]
    assert stored["model_config_values"] == produced.model_config
    assert stored["conformal_config"] == produced.conformal_config
    assert stored["ordering_config"] == produced.ordering_config


def test_get_study_returns_404_for_unknown_id(client) -> None:
    response = client.get("/studies/does-not-exist")
    assert response.status_code == 404


def test_tune_rejects_empty_origins(client, tune_payload, seasonal_search_space) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))
    response = client.post("/tune", json=tune_payload(origins=[]))
    assert response.status_code == 400
    assert "origins" in response.json()["detail"]


def test_tune_handle_returns_deterministic_session_id(
    monkeypatch, client, tune_payload, seasonal_search_space
) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
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


def test_failed_study_records_error(
    monkeypatch, client, tune_payload, seasonal_search_space
) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
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
