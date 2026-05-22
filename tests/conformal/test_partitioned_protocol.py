from __future__ import annotations

from calibre.conformal.runtime import (
    PartitionedConformalRuntime,
    SymmetricIntervalConfig,
    SymmetricIntervalRuntime,
)


def test_symmetric_interval_runtime_implements_protocol():
    config = SymmetricIntervalConfig(method="mscp", coverage=0.9)
    runtime = SymmetricIntervalRuntime(config)
    assert isinstance(runtime, PartitionedConformalRuntime)


def test_set_partition_states_round_trips():
    import pandas as pd

    from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y

    config = SymmetricIntervalConfig(method="mscp", coverage=0.9)
    runtime = SymmetricIntervalRuntime(config)

    # Synthesise one observation to populate score history
    lower_col, upper_col = config.interval_columns
    row = pd.DataFrame(
        {
            UNIQUE_ID: ["sku-a"],
            DS: [pd.Timestamp("2024-01-01")],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-01")],
            MODEL_NAME: ["m1"],
            H: [1],
            Y: [10.0],
            Y_HAT: [9.0],
            lower_col: [7.0],
            upper_col: [11.0],
        }
    )
    runtime.observe(row)

    states = runtime.get_partition_states()
    assert states, "expected non-empty partition states after observation"

    fresh = SymmetricIntervalRuntime(config)
    fresh.set_partition_states(states)
    assert fresh.get_partition_states() == states
