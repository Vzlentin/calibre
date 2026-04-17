"""Functions for creating ForecastTask objects from sales data and model configurations."""

from __future__ import annotations

import pandas as pd

from calibre.contracts.forecast_frame import DS, UNIQUE_ID, Y
from calibre.models.registry import get_adapter_cls
from calibre.tasks.forecast_task import ForecastTask


def build_tasks(
    sales: pd.DataFrame,
    model_configs: list[dict],
    horizon: int,
    series_filter: list[str] | None = None,
) -> list[ForecastTask]:
    """Create ForecastTask objects from sales data and model configs.

    For per-series adapters (``PARALLEL_BY_UID=True``), emits one task per
    (unique_id, config) pair. For global adapters (``PARALLEL_BY_UID=False``),
    emits one task per config with all series in a single history DataFrame.

    Args:
        sales: Long-format DataFrame with [unique_id, ds, y]
        model_configs: List of model config dicts, each must have a "backend" key
        horizon: Forecast horizon (number of periods)
        series_filter: Optional list of unique_ids to include (None = all)

    Returns:
        Flat list of ForecastTask objects.
        The engine handles origin-based history truncation, so full history is passed.
    """
    if series_filter is not None:
        data = sales[sales[UNIQUE_ID].isin(series_filter)].copy()
    else:
        data = sales.copy()

    data = data[[UNIQUE_ID, DS, Y]].sort_values([UNIQUE_ID, DS]).reset_index(drop=True)

    tasks: list[ForecastTask] = []
    for model_config in model_configs:
        adapter_cls = get_adapter_cls(model_config)

        if adapter_cls.PARALLEL_BY_UID:
            for uid in data[UNIQUE_ID].unique():
                series_data = data[data[UNIQUE_ID] == uid].sort_values(DS).reset_index(drop=True)
                tasks.append(
                    ForecastTask(
                        history=series_data,
                        horizon=horizon,
                        model_config=model_config,
                    )
                )
        else:
            tasks.append(
                ForecastTask(
                    history=data,
                    horizon=horizon,
                    model_config=model_config,
                )
            )

    return tasks
