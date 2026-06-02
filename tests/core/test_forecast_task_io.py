from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask, ForecastTaskRef


def test_forecast_task_to_uri_round_trips_from_fsspec_uri() -> None:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            DS: pd.to_datetime(["2024-01-01", "2024-01-08"]),
            Y: [1.0, 2.0],
        }
    )
    task = ForecastTask(
        history=history,
        horizon=2,
        model_config={"backend": "stub", "model": "stub_model"},
        forecast_origin=pd.Timestamp("2024-01-08"),
    )

    ref = task.to_uri("memory://calibre-tests/task-refs")
    materialized = ref.materialize()

    assert isinstance(ref, ForecastTaskRef)
    assert materialized.horizon == task.horizon
    assert materialized.model_config == task.model_config
    assert materialized.forecast_origin == task.forecast_origin
    pd.testing.assert_frame_equal(materialized.history, history)
