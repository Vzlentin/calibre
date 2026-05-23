from __future__ import annotations

import optuna
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from calibre.api import main as api_main
from calibre.api.lifecycle import MemoryLifecycleStore
from calibre.api.main import app, register_tuning_objective, register_tuning_search_space
from calibre.core.forecast_frame import UNIQUE_ID, Y_HAT
from calibre.core.order_types import CostStruct
from calibre.tuning import Regret, TuningCandidate, TuningTask


def _target_from_yhat(frame: pd.DataFrame, costs: CostStruct) -> float:
    del costs
    return float(frame[Y_HAT].sum())


def _order_from_target(target: float, ip: float, *, reorder_point=None) -> float:
    del reorder_point
    return max(float(target) - float(ip), 0.0)


def _search_space(trial: optuna.Trial) -> TuningCandidate:
    del trial
    return TuningCandidate(model_config={"backend": "stub", "model": "stub_model"})


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    fresh = MemoryLifecycleStore()
    monkeypatch.setattr(api_main, "_LIFECYCLE_STORE", fresh)
    monkeypatch.setattr(api_main, "_SEARCH_SPACES", {})
    monkeypatch.setattr(api_main, "_OBJECTIVES", {})
    return fresh


@pytest.fixture
def client():
    return TestClient(app)


def _history_records(uid: str = "A") -> list[dict]:
    dates = pd.date_range("2024-01-07", periods=6, freq="W-SUN")
    return [
        {UNIQUE_ID: uid, "ds": ds.strftime("%Y-%m-%d"), "y": float(idx + 1)}
        for idx, ds in enumerate(dates)
    ]


def _payload() -> dict:
    return {
        "tenant": "acme",
        "sku_set": ["A"],
        "horizon": 2,
        "freq": "W-SUN",
        "history": _history_records(),
        "actuals": _history_records(),
        "origins": ["2024-01-14"],
        "base_model_config": {"backend": "stub", "model": "stub_model"},
        "search_space_id": "seasonal",
        "objective_id": "regret",
        "n_trials": 1,
    }


def _register_regret() -> None:
    register_tuning_search_space("seasonal", _search_space)
    register_tuning_objective(
        "regret",
        Regret(
            _target_from_yhat,
            _order_from_target,
            CostStruct(underage_cost=2.0, overage_cost=1.0),
            oracle_cost=999.0,
        ),
    )


def test_tune_computes_oracle_before_study(monkeypatch, client) -> None:
    _register_regret()
    captured: dict[str, TuningTask] = {}

    def _fake_optimize(task: TuningTask) -> TuningCandidate:
        captured["task"] = task
        return TuningCandidate(model_config={"season_length": 4})

    monkeypatch.setattr(api_main, "optimize_task_candidate", _fake_optimize)

    response = client.post("/tune", json=_payload())

    assert response.status_code == 202, response.text
    task = captured["task"]
    assert isinstance(task.objective, Regret)
    assert task.objective.oracle_cost == 0.0


def test_studies_endpoint_exposes_oracle_cost(monkeypatch, client) -> None:
    _register_regret()
    monkeypatch.setattr(
        api_main,
        "optimize_task_candidate",
        lambda task: TuningCandidate(model_config={"season_length": 4}),
    )

    handle = client.post("/tune", json=_payload()).json()
    detail = client.get(f"/studies/{handle['study_id']}")

    assert detail.status_code == 200, detail.text
    assert detail.json()["oracle_cost"] == 0.0
