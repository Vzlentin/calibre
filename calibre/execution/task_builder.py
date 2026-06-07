"""Functions for creating ForecastTask objects from sales data and model configurations."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask, TaskGroups
from calibre.forecasting.adapter_registry import get_adapter_cls, get_scope
from calibre.reconciliation.summing import build_summing_matrix


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


def build_node_history(sales: pd.DataFrame, hierarchy: pd.DataFrame | None) -> pd.DataFrame:
    """Expand bottom-level history to the hierarchy's full node set.

    ``hierarchy=None`` preserves the flat-panel input contract. With a hierarchy,
    aggregate rows are real time series whose labels and ordering come directly
    from :func:`build_summing_matrix`.
    """
    if hierarchy is None:
        return sales.copy()

    data = sales[[UNIQUE_ID, DS, Y]].copy()
    data[UNIQUE_ID] = data[UNIQUE_ID].astype(str)
    data[DS] = pd.to_datetime(data[DS]).astype("datetime64[ns]")
    data[Y] = data[Y].astype("float64")

    summing = build_summing_matrix(hierarchy)
    unknown = set(data[UNIQUE_ID].unique()) - set(summing.bottom_ids)
    if unknown:
        raise ValueError(
            f"history contains unique_id values not present in hierarchy: {sorted(unknown)}"
        )

    bottom = data.sort_values([UNIQUE_ID, DS], kind="stable").reset_index(drop=True)
    aggregate_rows: list[pd.DataFrame] = []
    for row_index, node_label in enumerate(
        summing.node_labels[summing.n_bottom :], start=summing.n_bottom
    ):
        member_ids = [
            bottom_id
            for bottom_id, member in zip(summing.bottom_ids, summing.S[row_index], strict=True)
            if member > 0
        ]
        node = (
            data[data[UNIQUE_ID].isin(member_ids)]
            .groupby(DS, sort=True)[Y]
            .sum(min_count=1)
            .reset_index(name=Y)
        )
        node[UNIQUE_ID] = node_label
        aggregate_rows.append(node[[UNIQUE_ID, DS, Y]])

    if not aggregate_rows:
        return bottom

    nodes = pd.concat([bottom, *aggregate_rows], ignore_index=True)
    node_order = {label: i for i, label in enumerate(summing.node_labels)}
    nodes["_node_order"] = nodes[UNIQUE_ID].map(node_order).astype("int64")
    return (
        nodes.sort_values(["_node_order", DS], kind="stable")
        .drop(columns="_node_order")
        .reset_index(drop=True)
    )


def build_tasks(
    sales: pd.DataFrame,
    model_configs: list[dict],
    horizon: int,
    series_filter: list[str] | None = None,
    overrides: Mapping[str, list[dict]] | None = None,
    hierarchy: pd.DataFrame | None = None,
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
        hierarchy: Optional bottom-level hierarchy attributes. When present,
            bottom history is expanded to node-level history before task
            construction so aggregate nodes are forecast independently.

    Returns:
        A :class:`TaskGroups` partition. The engine consumes this directly and
        never re-interprets ``get_scope``. The engine handles origin-based
        history truncation, so full history is passed.
    """
    history = build_node_history(sales, hierarchy)

    if series_filter is not None:
        data = history[history[UNIQUE_ID].isin(series_filter)].copy()
    else:
        data = history.copy()

    uid_order = {
        uid: index for index, uid in enumerate(data[UNIQUE_ID].astype(str).drop_duplicates())
    }
    data = data[[UNIQUE_ID, DS, Y]].copy()
    data["_uid_order"] = data[UNIQUE_ID].astype(str).map(uid_order)
    data = (
        data.sort_values(["_uid_order", DS], kind="stable")
        .drop(columns="_uid_order")
        .reset_index(drop=True)
    )

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
