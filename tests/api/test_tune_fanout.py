"""Tests for local/global tuning fan-out behind /tune."""

from __future__ import annotations

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from calibre.api import tuning_service
from calibre.api.lifecycle import LifecycleStore
from calibre.api.main import create_app
from calibre.api.schemas import TuneRequest
from calibre.api.tuning_service import (
    register_tuning_objective,
    register_tuning_search_space,
)
from calibre.core.forecast_frame import UNIQUE_ID
from calibre.evaluation.point_metrics import smape
from calibre.storage.postgres import (
    GlobalTuningRunRepo,
    TuningRunRepo,
    make_engine,
    make_session_factory,
    session_scope,
)
from calibre.tuning import Accuracy, LocalTuningTask, TuningCandidate

SKU_SET = ["A", "B", "C", "D", "E"]


@pytest.fixture
def uris(tmp_path, tuning_history_records):
    """Stage sales + actuals parquet covering the whole SKU_SET (URI ingress)."""
    sales_path = tmp_path / "sales.parquet"
    actuals_path = tmp_path / "actuals.parquet"
    pd.DataFrame(tuning_history_records(SKU_SET)).to_parquet(sales_path)
    actuals = pd.DataFrame(tuning_history_records(SKU_SET))
    actuals["ds"] = pd.to_datetime(actuals["ds"])
    actuals.to_parquet(actuals_path)
    return str(sales_path), str(actuals_path)


def _tune_payload(sku_set: list[str], sales_uri: str, actuals_uri: str) -> dict:
    return {
        "tenant": "acme",
        "sku_set": sku_set,
        "horizon": 2,
        "freq": "W-SUN",
        "sales_uri": sales_uri,
        "actuals_uri": actuals_uri,
        "origins": ["2024-02-04"],
        "base_model_config": {"backend": "stub", "model": "stub_model"},
        "search_space_id": "seasonal",
        "objective_id": "accuracy",
        "n_trials": 1,
        "conformal_config": {
            "method": "aci",
            "coverage": 0.9,
            "calibration_window": 4,
            "gamma": 0.05,
        },
    }


def _global_tune_payload(sku_set: list[str], sales_uri: str, actuals_uri: str) -> dict:
    payload = _tune_payload(sku_set, sales_uri, actuals_uri)
    payload["hpo_scope"] = "global"
    payload["base_model_config"] = {"backend": "stub", "model": "stub_model", "scope": "global"}
    return payload


@pytest.fixture
def tuning_db(tmp_path, monkeypatch):
    db_path = tmp_path / "tune.db"
    db_url = f"sqlite+pysqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("CALIBRE_DATABASE_URL", db_url)
    config = Config("alembic.ini")
    command.upgrade(config, "head")
    return db_url


@pytest.fixture(autouse=True)
def _reset_registries(monkeypatch):
    monkeypatch.setattr(tuning_service, "_SEARCH_SPACES", {})
    monkeypatch.setattr(tuning_service, "_OBJECTIVES", {})


@pytest.fixture
def client(tuning_db):
    # Build the app after CALIBRE_DATABASE_URL is set so its db_factory (used by
    # the /tune fanout's tuning repos) resolves to this test's migrated DB at
    # construction time; lifecycle records stay in-memory.
    return TestClient(create_app(lifecycle_store=LifecycleStore()))


def _factory_for(db_url: str):
    return make_session_factory(make_engine(db_url))


def test_per_sku_best_configs_persisted(
    monkeypatch, tuning_db, client, uris, seasonal_search_space
) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))

    # Each uid must get ITS OWN candidate, routed to the right uid in both the
    # API response and the per-uid table. Capture the candidate produced for
    # each task (and the history slice it was handed) and assert the
    # stored/returned values are those captured candidates — the
    # request->per-uid-persistence plumbing — rather than re-deriving the fake's
    # formula in lockstep with it.
    produced: dict[str, TuningCandidate] = {}
    seen_history: dict[str, set] = {}

    def _fake_optimize(task: LocalTuningTask) -> TuningCandidate:
        idx = SKU_SET.index(task.unique_id)
        candidate = TuningCandidate(
            model_config={
                "backend": "stub",
                "model": "stub_model",
                "season_length": [4, 13, 26, 52][idx % 4],
            },
            conformal_config={"gamma": 0.01 + 0.01 * idx},
            ordering_config={},
        )
        produced[task.unique_id] = candidate
        seen_history[task.unique_id] = set(task.history[UNIQUE_ID].unique())
        return candidate

    monkeypatch.setattr(tuning_service, "optimize_local_task_candidate", _fake_optimize)

    submit = client.post("/tune", json=_tune_payload(SKU_SET, *uris))
    assert submit.status_code == 202, submit.text
    handle = submit.json()
    study_id = handle["study_id"]
    session_id = handle["session_id"]

    detail = client.get(f"/studies/{study_id}").json()
    assert detail["status"] == "succeeded"
    assert set(detail["best_candidates"]) == set(SKU_SET)

    # One optimizer call per uid, each handed only that uid's history slice.
    assert set(produced) == set(SKU_SET)
    for uid in SKU_SET:
        assert seen_history[uid] == {uid}

    # API response routes each uid's own computed candidate back under that uid.
    for uid in SKU_SET:
        stored = detail["best_candidates"][uid]
        assert stored["model_config_values"] == produced[uid].model_config
        assert stored["conformal_config"]["gamma"] == pytest.approx(
            produced[uid].conformal_config["gamma"]
        )

    # The per-uid table persists the same candidate under the right uid + session.
    factory = _factory_for(tuning_db)
    with session_scope(factory) as session:
        rows = TuningRunRepo(session).list_for_session(session_id)
    assert {row.unique_id for row in rows} == set(SKU_SET)
    for row in rows:
        assert row.candidate["model_config"] == produced[row.unique_id].model_config
        assert row.candidate["conformal_config"]["gamma"] == pytest.approx(
            produced[row.unique_id].conformal_config["gamma"]
        )


def test_tune_resumes_partial_completion(
    monkeypatch, tuning_db, client, uris, seasonal_search_space
) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))

    payload = _tune_payload(SKU_SET[:3], *uris)
    signature = tuning_service._tuning_signature(
        TuneRequest(**payload), [pd.Timestamp(o) for o in payload["origins"]]
    )

    factory = _factory_for(tuning_db)
    handle_session = client.post("/tune", json=payload).json()
    pre_session_id = handle_session["session_id"]

    with session_scope(factory) as session:
        TuningRunRepo(session).upsert(
            pre_session_id,
            "A",
            config_signature=signature,
            candidate={
                "model_config": {"season_length": 99},
                "conformal_config": {"gamma": 0.42},
                "ordering_config": {},
            },
            score=None,
        )
        TuningRunRepo(session).upsert(
            pre_session_id,
            "B",
            config_signature=signature,
            candidate={
                "model_config": {"season_length": 88},
                "conformal_config": {"gamma": 0.33},
                "ordering_config": {},
            },
            score=None,
        )

    tuned_uids: list[str] = []

    def _fake_optimize(task: LocalTuningTask) -> TuningCandidate:
        tuned_uids.append(task.unique_id)
        return TuningCandidate(
            model_config={"season_length": 13},
            conformal_config={"gamma": 0.05},
            ordering_config={},
        )

    monkeypatch.setattr(tuning_service, "optimize_local_task_candidate", _fake_optimize)

    submit = client.post("/tune", json=payload)
    study_id = submit.json()["study_id"]
    detail = client.get(f"/studies/{study_id}").json()

    assert detail["status"] == "succeeded"
    # Only the un-cached uid is (re)tuned; A and B resume their seeded candidates.
    assert tuned_uids == ["C"]
    candidates = detail["best_candidates"]
    assert candidates["A"]["model_config_values"]["season_length"] == 99
    assert candidates["A"]["conformal_config"]["gamma"] == pytest.approx(0.42)
    assert candidates["B"]["model_config_values"]["season_length"] == 88
    assert candidates["B"]["conformal_config"]["gamma"] == pytest.approx(0.33)
    assert candidates["C"]["model_config_values"]["season_length"] == 13


def test_local_hpo_reruns_when_config_changes(
    monkeypatch, tuning_db, client, uris, seasonal_search_space
) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))

    sku_set = SKU_SET[:2]
    run_count = {"n": 0}

    def _fake_optimize(task: LocalTuningTask) -> TuningCandidate:
        run_count["n"] += 1
        return TuningCandidate(
            model_config={"season_length": 13},
            conformal_config={"gamma": 0.05},
            ordering_config={},
        )

    monkeypatch.setattr(tuning_service, "optimize_local_task_candidate", _fake_optimize)

    payload = _tune_payload(sku_set, *uris)
    first = client.post("/tune", json=payload).json()
    assert client.get(f"/studies/{first['study_id']}").json()["status"] == "succeeded"
    assert run_count["n"] == len(sku_set)

    # Identical inputs (same session + signature) -> resume, no re-tuning.
    second = client.post("/tune", json=payload).json()
    assert client.get(f"/studies/{second['study_id']}").json()["status"] == "succeeded"
    assert run_count["n"] == len(sku_set)

    # Same session (tenant/sku_set/model/conformal unchanged) but a different
    # tuning input (n_trials) must invalidate the per-uid cache and re-tune,
    # rather than silently returning the stale candidates.
    changed = _tune_payload(sku_set, *uris)
    changed["n_trials"] = 99
    third = client.post("/tune", json=changed).json()
    assert client.get(f"/studies/{third['study_id']}").json()["status"] == "succeeded"
    assert run_count["n"] == 2 * len(sku_set)


def test_global_hpo_broadcasts_single_candidate(
    monkeypatch, tuning_db, client, uris, seasonal_search_space
) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))

    tasks: list = []
    captured: dict[str, TuningCandidate] = {}

    def _fake_panel(task) -> TuningCandidate:
        tasks.append(task)
        candidate = TuningCandidate(
            model_config={
                "backend": "stub",
                "model": "stub_model",
                "scope": "global",
                "season_length": 52,
            },
            conformal_config={"gamma": 0.09},
            ordering_config={},
        )
        captured["candidate"] = candidate
        return candidate

    monkeypatch.setattr(tuning_service, "optimize_global_task_candidate", _fake_panel)

    submit = client.post("/tune", json=_global_tune_payload(SKU_SET, *uris))
    assert submit.status_code == 202, submit.text
    handle = submit.json()
    study_id = handle["study_id"]
    session_id = handle["session_id"]

    detail = client.get(f"/studies/{study_id}").json()
    assert detail["status"] == "succeeded"

    # One panel study over the whole panel (not one per uid), fed every uid.
    assert len(tasks) == 1
    assert set(tasks[0].history[UNIQUE_ID].unique()) == set(SKU_SET)

    # The single produced candidate is broadcast verbatim to every uid.
    produced = captured["candidate"]
    assert set(detail["best_candidates"]) == set(SKU_SET)
    for uid in SKU_SET:
        stored = detail["best_candidates"][uid]
        assert stored["model_config_values"] == produced.model_config
        assert stored["conformal_config"]["gamma"] == pytest.approx(
            produced.conformal_config["gamma"]
        )

    # Authoritative result lives in global_tuning_runs; nothing in the per-uid table.
    factory = _factory_for(tuning_db)
    with session_scope(factory) as session:
        global_row = GlobalTuningRunRepo(session).get(session_id)
        per_uid_rows = TuningRunRepo(session).list_for_session(session_id)
    assert global_row is not None
    assert global_row.candidate["model_config"] == produced.model_config
    assert per_uid_rows == []


def test_global_hpo_resumes_from_cache(
    monkeypatch, tuning_db, client, uris, seasonal_search_space
) -> None:
    register_tuning_search_space("seasonal", seasonal_search_space)
    register_tuning_objective("accuracy", Accuracy(metric=smape))

    run_count = {"n": 0}

    def _fake_panel(task) -> TuningCandidate:
        run_count["n"] += 1
        return TuningCandidate(
            model_config={"scope": "global", "season_length": 13},
            conformal_config={"gamma": 0.05},
            ordering_config={},
        )

    monkeypatch.setattr(tuning_service, "optimize_global_task_candidate", _fake_panel)

    payload = _global_tune_payload(SKU_SET, *uris)
    first = client.post("/tune", json=payload).json()
    assert client.get(f"/studies/{first['study_id']}").json()["status"] == "succeeded"
    assert run_count["n"] == 1

    # Same session + identical signature -> reuse the cached global result.
    second = client.post("/tune", json=payload).json()
    detail = client.get(f"/studies/{second['study_id']}").json()
    assert detail["status"] == "succeeded"
    assert run_count["n"] == 1
    assert detail["best_candidates"]["A"]["model_config_values"]["season_length"] == 13

    # A different tuning input (n_trials) changes the signature -> re-run.
    changed = _global_tune_payload(SKU_SET, *uris)
    changed["n_trials"] = 99
    third = client.post("/tune", json=changed).json()
    assert client.get(f"/studies/{third['study_id']}").json()["status"] == "succeeded"
    assert run_count["n"] == 2
