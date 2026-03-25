"""Functions for creating ForecastTask objects from sales data and model configurations."""
from __future__ import annotations

import pandas as pd

from calibre.contracts.forecast_frame import DS, UNIQUE_ID, Y
from calibre.tasks.forecast_task import ForecastTask


def build_tasks(
    sales: pd.DataFrame,
    model_configs: list[dict],
    horizon: int,
    series_filter: list[str] | None = None,
) -> list[ForecastTask]:
    """Create one ForecastTask per (unique_id, model_config) combination.

    Args:
        sales: Long-format DataFrame with [unique_id, ds, y]
        model_configs: List of model config dicts, each must have "backend" and "model" keys
        horizon: Forecast horizon (number of periods)
        series_filter: Optional list of unique_ids to include (None = all)

    Returns:
        Flat list of ForecastTask objects.
        The engine handles origin-based history truncation, so full history is passed.
    """
    tasks = []

    # Determine which series to include
    if series_filter is None:
        series_to_include = sales[UNIQUE_ID].unique().tolist()
    else:
        series_to_include = series_filter

    # For each series and each model config, create a task
    for unique_id in series_to_include:
        # Filter to this series' data
        series_data = sales[sales[UNIQUE_ID] == unique_id].copy()

        # Extract history (only ds and y columns)
        history = series_data[[DS, Y]].sort_values(DS).reset_index(drop=True)

        # Create a task for each model config
        for model_config in model_configs:
            task = ForecastTask(
                unique_id=unique_id,
                history=history,
                horizon=horizon,
                model_config=model_config,
                forecast_origin=None,
                future_x=None,
            )
            tasks.append(task)

    return tasks
