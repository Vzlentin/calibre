"""Functions for creating ForecastTask objects from sales data and model configurations."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask, TaskGroups
from calibre.forecasting.adapter_registry import get_adapter_cls, get_scope


def partition_tasks(tasks: list[ForecastTask]) -> TaskGroups:
    """Partition pre-built tasks into a :class:`TaskGroups` by resolved scope.

    For callers that construct ``ForecastTask`` objects directly (benchmarks,
    tests) rather than via :func:`build_tasks`. Scope is resolved exactly once,
    here, through the forecasting registry — the engine never re-interprets it.
    """
    local: list[ForecastTask] = []
    global_: list[ForecastTask] = []
    for task in tasks:
        if get_scope(task.model_config) == "local":
            local.append(task)
        else:
            global_.append(task)
    return TaskGroups(local=local, global_=global_)


def _global_dedup_key(task: ForecastTask) -> tuple[tuple[str, ...], str, int]:
    """Stable content key for global-task dedup.

    Keyed on the panel's unique-id set, the canonical config JSON, and the
    horizon — not ``id(task.history)`` object identity. A defensive copy of the
    history frame therefore dedups identically (the old identity key silently
    failed on a cloned frame).

    The uid set is a sorted tuple rather than a joined string so that
    comma-bearing unique_ids cannot alias (``{"a,b", "c"}`` and
    ``{"a", "b,c"}`` are distinct keys, not both ``"a,b,c"``).
    """
    uids = tuple(sorted(task.history[UNIQUE_ID].astype(str).unique()))
    config = json.dumps(task.model_config, sort_keys=True, default=str)
    return uids, config, task.horizon


def build_tasks(
    sales: pd.DataFrame,
    model_configs: list[dict],
    horizon: int,
    series_filter: list[str] | None = None,
    overrides: Mapping[str, list[dict]] | None = None,
) -> TaskGroups:
    """Create ForecastTask objects from sales data and model configs.

    Scope is resolved exactly once here. For ``scope="local"`` (default),
    emits one task per (unique_id, config) pair into ``TaskGroups.local``. For
    ``scope="global"``, emits one task per config (deduplicated across series)
    into ``TaskGroups.global_``.

    Args:
        sales: Long-format DataFrame with [unique_id, ds, y]
        model_configs: List of model config dicts, each must have a "backend" key
        horizon: Forecast horizon (number of periods)
        series_filter: Optional list of unique_ids to include (None = all)
        overrides: Optional mapping from ``unique_id`` to a per-series list of
            model configs. When present for a series, that list replaces
            ``model_configs`` for that series only. Unknown ``unique_id`` keys
            raise ``ValueError``.

    Returns:
        A :class:`TaskGroups` partition. The engine consumes this directly and
        never re-interprets ``get_scope``. The engine handles origin-based
        history truncation, so full history is passed.
    """
    if series_filter is not None:
        data = sales[sales[UNIQUE_ID].isin(series_filter)].copy()
    else:
        data = sales.copy()

    data = data[[UNIQUE_ID, DS, Y]].sort_values([UNIQUE_ID, DS]).reset_index(drop=True)

    uids = data[UNIQUE_ID].unique().tolist()

    if overrides is not None:
        unknown = set(overrides) - set(uids)
        if unknown:
            raise ValueError(
                f"overrides contains unknown unique_id(s): {sorted(unknown)}. "
                f"Available: {sorted(uids)}"
            )

    local_tasks: list[ForecastTask] = []
    global_tasks: list[ForecastTask] = []
    seen_global: set[tuple[tuple[str, ...], str, int]] = set()

    # Resolve per-series configs up-front so we don't re-validate inside loops.
    def _configs_for(uid: str) -> list[dict]:
        if overrides is not None and uid in overrides:
            return overrides[uid]
        return model_configs

    for uid in uids:
        uid_configs = _configs_for(uid)
        series_data = data[data[UNIQUE_ID] == uid].sort_values(DS).reset_index(drop=True)

        for model_config in uid_configs:
            get_adapter_cls(model_config)  # validate backend early

            if get_scope(model_config) == "local":
                local_tasks.append(
                    ForecastTask(
                        history=series_data,
                        horizon=horizon,
                        model_config=model_config,
                    )
                )
            else:
                # Global scope still sees all series, so we emit one task per
                # config (not per uid). Dedup on a stable content key so the
                # same global config appearing in overrides for multiple uids
                # collapses to a single task.
                task = ForecastTask(
                    history=data,
                    horizon=horizon,
                    model_config=model_config,
                )
                key = _global_dedup_key(task)
                if key not in seen_global:
                    seen_global.add(key)
                    global_tasks.append(task)

    return TaskGroups(local=local_tasks, global_=global_tasks)
