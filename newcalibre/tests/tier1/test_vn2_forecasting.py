"""Exercise the VN2-local lawful native-median seasonal adapter.

Structural, capability, and refusal assertions are tolerance class 1. The
seasonal lookup is hand-derived class 2; repeated bytes are class 4.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from newcalibre.domain import (
    HISTORY_TIMESTAMP,
    POINT_FORECAST,
    SERIES_KEY,
    Calendar,
    ForecastTask,
    GuaranteeClaim,
    InventoryPosition,
    Panel,
    Scope,
    TargetSupport,
    quantile_column,
    validate_forecast_frame,
)
from newcalibre.forecasting import (
    AdapterCapability,
    AdapterCapabilityError,
    AdapterConfigurationError,
    AdapterLifecycleError,
    ForecastAdapter,
    available_backends,
)
from newcalibre.ledger import BoundKey, ForecastIssuance, ForecastKey
from newcalibre.ordering import (
    OrderingInputError,
    OrderingSetup,
    PolicyRequest,
    compile_ordering,
    dispatch_policy,
)
from newcalibre.protocols.vn2 import (
    VN2_SEASONAL_NAIVE_BACKEND,
    VN2SeasonalNaiveQuantileAdapter,
    available_vn2_backends,
    load_vn2_config,
    resolve_vn2_adapter,
)

pytestmark = pytest.mark.tier1

ORIGIN = pd.Timestamp("2025-04-14")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_CONFIG = PROJECT_ROOT / "benchmarks" / "vn2" / "protocol.yaml"


def _config(**overrides: object) -> dict[str, object]:
    return {
        "backend": VN2_SEASONAL_NAIVE_BACKEND,
        "m": 52,
        "model_name": "vn2-native-median",
        "quantile_levels": [0.5],
        "censoring_aware": False,
        **overrides,
    }


def _task(config: Mapping[str, object] | None = None) -> ForecastTask:
    timestamps = pd.date_range(end="2025-04-07", periods=52, freq="W-MON")
    history = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["0_126"] * len(timestamps), dtype="string"),
            HISTORY_TIMESTAMP: pd.Series(timestamps),
            "value": pd.Series(np.arange(1, 53), dtype="float64"),
        }
    )
    return Panel.from_frame(
        history, calendar=Calendar("W-MON"), target_support=TargetSupport.REAL
    ).forecast_tasks(
        horizon=3,
        origin=ORIGIN,
        scope=Scope.GLOBAL,
        model_config=dict(config or _config()),
    )[0]


def _forecast_bytes(task: ForecastTask, config: Mapping[str, object]) -> bytes:
    adapter = resolve_vn2_adapter(config)
    adapter.fit(task)
    return adapter.predict(task).to_csv(index=False, lineterminator="\n").encode()


def test_vn2_registry_is_local_and_global_brick_remains_capability_free() -> None:
    assert available_vn2_backends() == (VN2_SEASONAL_NAIVE_BACKEND,)
    assert VN2_SEASONAL_NAIVE_BACKEND not in available_backends()
    assert "seasonal-naive" not in available_vn2_backends()

    adapter = resolve_vn2_adapter(_config())
    assert isinstance(adapter, ForecastAdapter)
    assert isinstance(adapter, VN2SeasonalNaiveQuantileAdapter)
    assert adapter.capabilities == frozenset({AdapterCapability.NATIVE_QUANTILES})
    assert adapter.requested_capabilities == frozenset({AdapterCapability.NATIVE_QUANTILES})


def test_vn2_variant_emits_canonical_quantile_exactly_equal_to_point() -> None:
    config = _config()
    task = _task(config)
    adapter = resolve_vn2_adapter(config)
    adapter.fit(task)

    frame = adapter.predict(task)
    quantile = quantile_column(0.5)

    pd.testing.assert_frame_equal(frame, validate_forecast_frame(frame, calendar=task.calendar))
    assert quantile == "quantile_0.5"
    assert frame.columns[-1] == quantile
    assert frame[POINT_FORECAST].tolist() == [1.0, 2.0, 3.0]
    pd.testing.assert_series_equal(
        frame[quantile],
        frame[POINT_FORECAST],
        check_names=False,
        check_exact=True,
    )


def test_vn2_variant_is_deterministic_and_task_closed() -> None:
    config = _config()
    task = _task(config)

    assert _forecast_bytes(task, config) == _forecast_bytes(task, config)


def test_vn2_median_drives_lawful_rs_path_without_point_fallback() -> None:
    adapter_config = _config()
    task = _task(adapter_config)
    adapter = resolve_vn2_adapter(adapter_config)
    adapter.fit(task)
    frame = adapter.predict(task)
    protocol = load_vn2_config(PROTOCOL_CONFIG)
    policy = protocol.ordering_policy
    policy_name = policy["name"]
    explicit_quantile = policy["quantile"]
    assert isinstance(policy_name, str)
    assert isinstance(explicit_quantile, float)
    configuration = compile_ordering(
        OrderingSetup(
            policy=policy_name,
            series_keys=("0_126",),
            cost_structure=protocol.cost_structure,
            decision_timing=protocol.timing,
            task_horizon=protocol.task_horizon,
            explicit_quantile=explicit_quantile,
        )
    )
    model_name = adapter_config["model_name"]
    assert isinstance(model_name, str)
    issuances: dict[ForecastKey, dict[BoundKey, ForecastIssuance]] = {
        ("0_126", ORIGIN, step, model_name): {}
        for step in range(1, protocol.timing.protection_period + 1)
    }
    inventory = {"0_126": InventoryPosition(on_hand=0.0, on_order=0.0, backorders=0.0)}

    decision = dispatch_policy(
        PolicyRequest(
            frame=frame,
            issuances=issuances,
            inventory_positions=inventory,
            configuration=configuration,
        )
    )[0]

    assert decision.evidence.raw_target == 6.0
    assert decision.evidence.source_columns == (quantile_column(0.5),)
    assert decision.evidence.source_descriptor.type.claim is GuaranteeClaim.NONE
    assert decision.quantity == 6.0

    point_only = frame.drop(columns=[quantile_column(0.5)])
    with pytest.raises(OrderingInputError, match="quantile"):
        dispatch_policy(
            PolicyRequest(
                frame=point_only,
                issuances=issuances,
                inventory_positions=inventory,
                configuration=configuration,
            )
        )


@pytest.mark.parametrize(
    "levels",
    [None, [], [0.4], [0.5, 0.6], [float("nan")], [True], "0.5"],
    ids=["missing", "empty", "wrong", "multiple", "nan", "boolean", "string"],
)
def test_vn2_variant_requires_the_single_lawful_median_quantile_before_fit(
    levels: object,
) -> None:
    config = _config()
    if levels is None:
        config.pop("quantile_levels")
    else:
        config["quantile_levels"] = levels

    with pytest.raises(AdapterConfigurationError, match="single 0.5"):
        resolve_vn2_adapter(config)


def test_vn2_registry_rejects_unsupported_capabilities_before_fit() -> None:
    with pytest.raises(AdapterCapabilityError, match="censoring_aware_fit"):
        resolve_vn2_adapter(_config(censoring_aware=True))


def test_vn2_variant_keeps_non_quantile_capabilities_loud() -> None:
    config = _config()
    task = _task(config)
    adapter = resolve_vn2_adapter(config)

    with pytest.raises(AdapterLifecycleError, match="successful fit"):
        adapter.predict(task)
    with pytest.raises(AdapterCapabilityError, match="fitted_values"):
        adapter.fit(task, collect_fitted_values=True)
    with pytest.raises(AdapterCapabilityError, match="fitted_values"):
        adapter.fitted_values(task)
    with pytest.raises(AdapterCapabilityError, match="artifact_persistence"):
        adapter.dump_state()
    with pytest.raises(AdapterCapabilityError, match="artifact_persistence"):
        adapter.load_state(b"state")
    with pytest.raises(AdapterCapabilityError, match="incremental_update"):
        adapter.update(task)


def test_vn2_variant_rejects_task_configuration_drift() -> None:
    config = _config()
    adapter = resolve_vn2_adapter(config)

    with pytest.raises(AdapterConfigurationError, match="must match"):
        adapter.fit(_task(_config(m=13)))
