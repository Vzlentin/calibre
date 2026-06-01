"""Tuning orchestration for the ``/tune`` route (issue #70).

This is route orchestration extracted from [calibre/api/main.py](calibre/api/main.py):
it consumes a ``TuneRequest``, runs per-series or panel HPO via
``calibre.tuning.optimize_*``, persists/resumes results through the tuning
repos, and writes study status to the lifecycle store. ``main.py`` keeps the
thin ``/tune`` + ``/studies/{id}`` route handlers and resolves the lifecycle
store + db factory, passing them into :func:`run_tune_job` so the app-state
accessors stay in ``main.py`` and this module never imports back into it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable

import optuna
import pandas as pd
from sqlalchemy.orm import sessionmaker

from calibre.api.errors import format_error
from calibre.api.lifecycle import LifecycleStoreProtocol
from calibre.api.schemas import TuneRequest
from calibre.core.forecast_frame import UNIQUE_ID
from calibre.core.run_status import RunStatus
from calibre.storage.postgres import GlobalTuningRunRepo, TuningRunRepo, session_scope
from calibre.tuning import (
    GlobalTuningTask,
    LocalTuningTask,
    StudyConfig,
    TuningCandidate,
    TuningObjective,
    optimize_global_task_candidate,
    optimize_local_task_candidate,
)

_SEARCH_SPACES: dict[str, Callable[[optuna.Trial], TuningCandidate]] = {}
_OBJECTIVES: dict[str, TuningObjective] = {}


def register_tuning_search_space(
    name: str, search_space: Callable[[optuna.Trial], TuningCandidate]
) -> None:
    _SEARCH_SPACES[name] = search_space


def register_tuning_objective(name: str, objective: TuningObjective) -> None:
    _OBJECTIVES[name] = objective


def has_search_space(name: str) -> bool:
    return name in _SEARCH_SPACES


def has_objective(name: str) -> bool:
    return name in _OBJECTIVES


def _filter_uid(frame: pd.DataFrame, uid: str) -> pd.DataFrame:
    if frame.empty or UNIQUE_ID not in frame.columns:
        return frame
    return frame[frame[UNIQUE_ID] == uid].reset_index(drop=True)


def _candidate_to_payload(candidate: TuningCandidate) -> dict[str, dict]:
    return {
        "model_config": dict(candidate.model_config),
        "conformal_config": dict(candidate.conformal_config),
        "ordering_config": dict(candidate.ordering_config),
    }


def _stored_candidate_payload(candidate: dict) -> dict[str, dict]:
    """Normalise a persisted ``candidate`` blob into the uniform payload shape."""
    return {
        "model_config": dict(candidate.get("model_config", {})),
        "conformal_config": dict(candidate.get("conformal_config", {})),
        "ordering_config": dict(candidate.get("ordering_config", {})),
    }


def _tuning_signature(req: TuneRequest, origins: list[pd.Timestamp]) -> str:
    """Hash the tuning inputs not already encoded in ``session_id``.

    ``session_id`` covers tenant/sku_set/base_model_config/conformal_config, so
    a cached result is only reused when the remaining inputs (search space,
    objective, n_trials, horizon, freq, origins) also match. Both local and
    global scope share this signature so their resume semantics cannot diverge.
    """
    payload = json.dumps(
        {
            "search_space_id": req.search_space_id,
            "objective_id": req.objective_id,
            "n_trials": int(req.n_trials),
            "horizon": int(req.horizon),
            "freq": req.freq,
            "origins": [pd.Timestamp(origin).isoformat() for origin in origins],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_existing_local_runs(
    factory: sessionmaker | None, session_id: str, signature: str
) -> dict[str, dict[str, dict]]:
    """Return cached per-uid candidates for this session in a single query.

    Rows tuned under a different signature (changed search space, n_trials,
    objective, horizon, or origins) are skipped, so a stale candidate is never
    resumed when the tuning inputs change.
    """
    if factory is None:
        return {}
    with session_scope(factory) as session:
        return {
            row.unique_id: _stored_candidate_payload(dict(row.candidate))
            for row in TuningRunRepo(session).list_for_session(session_id)
            if row.config_signature == signature
        }


def _persist_tuning_run(
    factory: sessionmaker | None,
    session_id: str,
    unique_id: str,
    signature: str,
    payload: dict[str, dict],
) -> None:
    if factory is None:
        return
    with session_scope(factory) as session:
        TuningRunRepo(session).upsert(
            session_id,
            unique_id,
            config_signature=signature,
            candidate=payload,
            score=None,
        )


def _load_existing_global_tuning_run(
    factory: sessionmaker | None, session_id: str, signature: str
) -> dict[str, dict] | None:
    if factory is None:
        return None
    with session_scope(factory) as session:
        row = GlobalTuningRunRepo(session).get(session_id)
        if row is None or row.config_signature != signature:
            return None
        return _stored_candidate_payload(dict(row.candidate))


def _persist_global_tuning_run(
    factory: sessionmaker | None,
    session_id: str,
    signature: str,
    payload: dict[str, dict],
) -> None:
    if factory is None:
        return
    with session_scope(factory) as session:
        GlobalTuningRunRepo(session).upsert(
            session_id,
            config_signature=signature,
            candidate=payload,
            score=None,
        )


def _run_local_hpo(
    req: TuneRequest,
    history: pd.DataFrame,
    actuals: pd.DataFrame,
    origins: list[pd.Timestamp],
    factory: sessionmaker | None,
    session_id: str,
) -> dict[str, dict[str, dict]]:
    """Per-series HPO: one candidate per uid, with signature-checked per-uid resume."""
    signature = _tuning_signature(req, origins)
    cached = _load_existing_local_runs(factory, session_id, signature)
    candidates: dict[str, dict[str, dict]] = {}
    for uid in req.sku_set:
        existing = cached.get(uid)
        if existing is not None:
            candidates[uid] = existing
            continue
        task = LocalTuningTask(
            unique_id=uid,
            history=_filter_uid(history, uid),
            horizon=int(req.horizon),
            base_model_config=dict(req.base_model_config),
            search_space=_SEARCH_SPACES[req.search_space_id],
            actuals=_filter_uid(actuals, uid),
            origins=origins,
            objective=_OBJECTIVES[req.objective_id],
            study_config=StudyConfig(n_trials=int(req.n_trials), freq=req.freq),
        )
        payload = _candidate_to_payload(optimize_local_task_candidate(task))
        _persist_tuning_run(factory, session_id, uid, signature, payload)
        candidates[uid] = payload
    return candidates


def _run_global_hpo(
    req: TuneRequest,
    history: pd.DataFrame,
    actuals: pd.DataFrame,
    origins: list[pd.Timestamp],
    factory: sessionmaker | None,
    session_id: str,
) -> dict[str, dict[str, dict]]:
    """Panel/global HPO: one candidate shared across all uids, broadcast on return.

    The single result is the authoritative row in ``global_tuning_runs`` (keyed
    by session + config signature for cross-submission resume); the broadcast
    keeps the study response a uniform ``dict[uid -> candidate]``.
    """
    signature = _tuning_signature(req, origins)
    payload = _load_existing_global_tuning_run(factory, session_id, signature)
    if payload is None:
        task = GlobalTuningTask(
            history=history,
            horizon=int(req.horizon),
            base_model_config=dict(req.base_model_config),
            search_space=_SEARCH_SPACES[req.search_space_id],
            actuals=actuals,
            origins=origins,
            objective=_OBJECTIVES[req.objective_id],
            study_config=StudyConfig(n_trials=int(req.n_trials), freq=req.freq),
        )
        payload = _candidate_to_payload(optimize_global_task_candidate(task))
        _persist_global_tuning_run(factory, session_id, signature, payload)
    return {uid: copy.deepcopy(payload) for uid in req.sku_set}


_HPO_RUNNERS = {"local": _run_local_hpo, "global": _run_global_hpo}


def run_tune_job(
    study_id: str,
    req: TuneRequest,
    history: pd.DataFrame,
    actuals: pd.DataFrame,
    origins: list[pd.Timestamp],
    *,
    store: LifecycleStoreProtocol,
    factory: sessionmaker | None,
) -> None:
    record = store.get_study(study_id)
    if record is None:
        return
    store.update_study(study_id, status=RunStatus.RUNNING)
    session_id = record.session_id
    runner = _HPO_RUNNERS[req.hpo_scope]
    try:
        candidates = runner(req, history, actuals, origins, factory, session_id)
        store.update_study(
            study_id,
            status=RunStatus.SUCCEEDED,
            best_candidates=candidates,
        )
    except Exception as exc:  # pragma: no cover - background task safety net
        store.update_study(study_id, status=RunStatus.FAILED, error=format_error(exc))
