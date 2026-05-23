from __future__ import annotations

import pandas as pd

from calibre.conformal.runtime import (
    PartitionedConformalRuntime,
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
)
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)


def test_symmetric_interval_runtime_implements_protocol() -> None:
    config = SymmetricIntervalConfig(method="mscp", coverage=0.9, calibration_window=5)
    runtime = SymmetricIntervalRuntime(config)

    assert isinstance(runtime, PartitionedConformalRuntime)

    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            DS: [pd.Timestamp("2024-01-02")],
            Y: [float("nan")],
            Y_HAT: [10.0],
            H: [1],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-01")],
            MODEL_NAME: ["stub"],
        }
    )
    observed = runtime.apply(frame)
    observed[Y] = 12.0
    runtime.observe(observed)

    states = runtime.get_partition_states()
    restored = SymmetricIntervalRuntime.from_partition_states(config, states)

    assert states
    assert restored.partition_keys == runtime.partition_keys
