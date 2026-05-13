"""Functions for creating ForecastTask objects from sales data and model configurations."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask
from calibre.forecasting.adapter_registry import get_adapter_cls, get_scope


def build_tasks(
    sales: pd.DataFrame,
    model_configs: list[dict],
    horizon: int,
    series_filter: list[str] | None = None,
    overrides: Mapping[str, list[dict]] | None = None,
) -> list[ForecastTask]:
    """Create ForecastTask objects from sales data and model configs.

    For ``scope="local"`` (default), emits one task per (unique_id, config)
    pair. For ``scope="global"``, emits one task per config with all series
    in a single history DataFrame.

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
        Flat list of ForecastTask objects.
        The engine handles origin-based history truncation, so full history is passed.
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

    tasks: list[ForecastTask] = []

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
                tasks.append(
                    ForecastTask(
                        history=series_data,
                        horizon=horizon,
                        model_config=model_config,
                    )
                )
            else:
                # Global scope with per-series overrides: the global model still
                # sees all series, so we emit one task per config (not per uid).
                # To avoid duplicates when multiple uids share the same global
                # override, we deduplicate by config id within this uid's pass.
                tasks.append(
                    ForecastTask(
                        history=data,
                        horizon=horizon,
                        model_config=model_config,
                    )
                )

    # When global configs appeared in overrides for multiple uids, we
    # duplicated the global tasks. Deduplicate by identity on the task tuple
    # (history ref equality is fine since we used the same ``data`` object).
    seen: set[tuple[int, str, int]] = set()
    deduped: list[ForecastTask] = []
    for task in tasks:
        key = id(task.history), json.dumps(task.model_config, sort_keys=True), task.horizon
        if key not in seen:
            seen.add(key)
            deduped.append(task)

    return deduped
