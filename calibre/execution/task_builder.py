"""Functions for creating ForecastTask objects from sales data and model configurations."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask, TaskGroups
from calibre.forecasting.adapter_registry import get_adapter_cls, get_scope
from calibre.reconciliation.summing import TOTAL_LABEL, HierarchyIndex


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


def build_node_history(sales: pd.DataFrame, hierarchy_index: HierarchyIndex | None) -> pd.DataFrame:
    """Expand bottom-level history to the hierarchy's full node set.

    ``hierarchy_index=None`` preserves the flat-panel input contract. With an
    index, aggregate rows are real time series whose labels and ordering come
    directly from the threaded index's canonical node order — the same instance
    run preparation built, never rebuilt here.
    """
    if hierarchy_index is None:
        return sales.copy()

    data = sales[[UNIQUE_ID, DS, Y]].copy()
    data[UNIQUE_ID] = data[UNIQUE_ID].astype(str)
    data[DS] = pd.to_datetime(data[DS]).astype("datetime64[ns]")
    data[Y] = data[Y].astype("float64")

    unknown = set(data[UNIQUE_ID].unique()) - set(hierarchy_index.bottom_ids)
    if unknown:
        raise ValueError(
            f"history contains unique_id values not present in hierarchy: {sorted(unknown)}"
        )

    node_order = {label: i for i, label in enumerate(hierarchy_index.node_labels)}
    bottom = data.sort_values([UNIQUE_ID, DS], kind="stable").reset_index(drop=True)
    bottom["_node_order"] = bottom[UNIQUE_ID].map(node_order).astype("int64")

    joined = data.merge(
        hierarchy_index.frame[[UNIQUE_ID, *hierarchy_index.attr_cols]],
        on=UNIQUE_ID,
        how="inner",
    )
    expected_members = hierarchy_index.expected_members()
    aggregate_rows: list[pd.DataFrame] = []
    for col in hierarchy_index.attr_cols:
        # Group on the stringified attribute column so the completeness check
        # reads the single stringified counting authority: str-colliding
        # values merge into one coherent node instead of silently duplicating.
        col_str = joined[col].astype(str)
        grouped = (
            joined.assign(**{col: col_str})
            .groupby([col, DS], sort=True)
            .agg(**{Y: (Y, "sum"), "_member_count": (UNIQUE_ID, "nunique")})
            .reset_index()
        )
        complete = grouped[
            grouped["_member_count"] == grouped[col].map(expected_members[col])
        ].copy()
        complete[UNIQUE_ID] = col + "=" + complete[col].astype(str)
        complete["_node_order"] = complete[UNIQUE_ID].map(node_order).astype("int64")
        aggregate_rows.append(complete[[UNIQUE_ID, DS, Y, "_node_order"]])

    total = (
        data.groupby(DS, sort=True)
        .agg(**{Y: (Y, "sum"), "_member_count": (UNIQUE_ID, "nunique")})
        .reset_index()
    )
    total = total[total["_member_count"] == len(hierarchy_index.bottom_ids)].copy()
    total[UNIQUE_ID] = TOTAL_LABEL
    total["_node_order"] = node_order[TOTAL_LABEL]
    aggregate_rows.append(total[[UNIQUE_ID, DS, Y, "_node_order"]])

    nodes = pd.concat([bottom, *aggregate_rows], ignore_index=True)
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

    uid_order = {
        uid: index for index, uid in enumerate(data[UNIQUE_ID].astype(str).drop_duplicates())
    }
    data = data[[UNIQUE_ID, DS, Y]].copy()
    data[UNIQUE_ID] = data[UNIQUE_ID].astype(str)
    data["_uid_order"] = data[UNIQUE_ID].astype(str).map(uid_order)
    data = (
        data.sort_values(["_uid_order", DS], kind="stable")
        .drop(columns="_uid_order")
        .reset_index(drop=True)
    )

    series_by_uid = {
        str(uid): group.reset_index(drop=True) for uid, group in data.groupby(UNIQUE_ID, sort=False)
    }
    uids = list(series_by_uid)

    if overrides is not None:
        unknown = set(overrides) - set(uids)
        if unknown:
            raise ValueError(
                f"overrides contains unknown unique_id(s): {sorted(unknown)}. "
                f"Available: {sorted(uids)}"
            )

    local_tasks: list[ForecastTask] = []
    global_tasks: list[ForecastTask] = []
    panel_uids = tuple(sorted(uids))
    seen_global: set[tuple[tuple[str, ...], str, int]] = set()

    # Resolve per-series configs up-front so we don't re-validate inside loops.
    def _configs_for(uid: str) -> list[dict]:
        if overrides is not None and uid in overrides:
            return overrides[uid]
        return model_configs

    for uid in uids:
        uid_configs = _configs_for(uid)
        series_data = series_by_uid[uid]

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
                config_key = json.dumps(model_config, sort_keys=True, default=str)
                key = (panel_uids, config_key, horizon)
                if key in seen_global:
                    continue
                seen_global.add(key)
                global_tasks.append(
                    ForecastTask(
                        history=data,
                        horizon=horizon,
                        model_config=model_config,
                    )
                )

    return TaskGroups(local=local_tasks, global_=global_tasks)
