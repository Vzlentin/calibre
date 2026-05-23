"""Tune service: oracle precompute + per-uid Optuna study orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pandas as pd
from sqlalchemy.orm import sessionmaker

from calibre.api.lifecycle import LifecycleStore
from calibre.api.schemas import TuneRequest
from calibre.core.forecast_frame import UNIQUE_ID
from calibre.core.run_status import RunStatus
from calibre.storage.postgres import TuningRunRepo, session_scope
from calibre.tuning import Regret, TuningCandidate, TuningObjective, TuningTask
from calibre.tuning.objectives import perfect_foresight_oracle_cost

logger = logging.getLogger(__name__)

TuningOptimizer = Callable[[TuningTask], TuningCandidate]


def filter_uid(frame: pd.DataFrame, uid: str) -> pd.DataFrame:
    if frame.empty or UNIQUE_ID not in frame.columns:
        return frame
    return frame[frame[UNIQUE_ID] == uid].reset_index(drop=True)


def candidate_to_payload(candidate: TuningCandidate) -> dict[str, dict]:
    return {
        "model_config": dict(candidate.model_config),
        "conformal_config": dict(candidate.conformal_config),
        "ordering_config": dict(candidate.ordering_config),
    }


def load_existing_tuning_run(
    factory: sessionmaker | None, session_id: str, unique_id: str
) -> dict[str, dict] | None:
    if factory is None:
        return None
    with session_scope(factory) as session:
        row = TuningRunRepo(session).get(session_id, unique_id)
        if row is None:
            return None
        candidate = dict(row.candidate)
    return {
        "model_config": dict(candidate.get("model_config", {})),
        "conformal_config": dict(candidate.get("conformal_config", {})),
        "ordering_config": dict(candidate.get("ordering_config", {})),
    }


def persist_tuning_run(
    factory: sessionmaker | None,
    session_id: str,
    unique_id: str,
    payload: dict[str, dict],
) -> None:
    if factory is None:
        return
    with session_scope(factory) as session:
        TuningRunRepo(session).upsert(
            session_id,
            unique_id,
            candidate=payload,
            score=None,
        )


def oracle_cost_for_request(
    req: TuneRequest,
    objective: TuningObjective,
    actuals: pd.DataFrame,
    origins: list[pd.Timestamp],
) -> float | None:
    """Precompute regret oracle cost, or return None for non-regret objectives.

    Raises :class:`TypeError` when ``objective_id == 'regret'`` but the
    registered objective isn't a ``Regret`` instance, and :class:`ValueError`
    on data-shape problems (re-raised from
    :func:`perfect_foresight_oracle_cost`). The HTTP layer translates both
    into 400s.
    """
    if req.objective_id == "regret" and not isinstance(objective, Regret):
        raise TypeError("objective_id 'regret' must be registered with a Regret objective")
    if not isinstance(objective, Regret):
        return None
    return perfect_foresight_oracle_cost(
        objective,
        actuals,
        origins,
        int(req.horizon),
        unique_ids=req.sku_set,
    )


def objective_for_study(objective: TuningObjective, oracle_cost: float | None) -> TuningObjective:
    if not isinstance(objective, Regret):
        return objective
    if oracle_cost is None:
        raise ValueError("regret objective requires precomputed oracle_cost")
    return cast(TuningObjective, replace(objective, oracle_cost=float(oracle_cost)))


def run_tune_job(
    *,
    store: LifecycleStore,
    factory: sessionmaker | None,
    study_id: str,
    req: TuneRequest,
    history: pd.DataFrame,
    actuals: pd.DataFrame,
    origins: list[pd.Timestamp],
    objective: TuningObjective,
    search_space: Callable,
    optimizer: TuningOptimizer,
) -> None:
    """Run a per-uid Optuna study, resume from any persisted candidates, and write results."""
    record = store.get_study(study_id)
    if record is None:
        return
    store.update_study(study_id, status=RunStatus.RUNNING)
    session_id = record.session_id
    candidates: dict[str, dict[str, dict]] = {}
    try:
        for uid in req.sku_set:
            existing = load_existing_tuning_run(factory, session_id, uid)
            if existing is not None:
                candidates[uid] = existing
                continue
            task = TuningTask(
                unique_id=uid,
                history=filter_uid(history, uid),
                horizon=int(req.horizon),
                base_model_config=dict(req.base_model_config),
                search_space=search_space,
                actuals=filter_uid(actuals, uid),
                origins=origins,
                objective=objective,
                n_trials=int(req.n_trials),
                freq=req.freq,
            )
            candidate = optimizer(task)
            payload = candidate_to_payload(candidate)
            persist_tuning_run(factory, session_id, uid, payload)
            candidates[uid] = payload
        store.update_study(
            study_id,
            status=RunStatus.SUCCEEDED,
            best_candidates=candidates,
        )
    except Exception as exc:  # pragma: no cover - background task safety net
        import traceback

        logger.exception("tune job failed", extra={"study_id": study_id})
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        store.update_study(study_id, status=RunStatus.FAILED, error=error)
