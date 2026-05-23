from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.vn2.replay import (
    CachedRound,
    PolicyApplicationError,
    VN2ReplayCache,
    replay_cached_cost,
)
from benchmarks.vn2.simulator import ProductState
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    quantile_column,
)


def _cache() -> VN2ReplayCache:
    qcol = quantile_column(0.5)
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            DS: [pd.Timestamp("2024-01-08")],
            Y: [float("nan")],
            Y_HAT: [1.0],
            H: [1],
            FORECAST_ORIGIN: [pd.Timestamp("2024-01-01")],
            MODEL_NAME: ["stub"],
            qcol: [1.0],
        }
    )
    return VN2ReplayCache(
        initial_states={"A": ProductState("A", 0.0, 0.0, 0.0)},
        warmup_frames=[],
        rounds={1: CachedRound(round_num=1, origin=pd.Timestamp("2024-01-01"), frame=frame)},
        actuals_by_round={1: {"A": 0.0}},
        model_config={"quantiles": [0.5]},
        quantile_alpha=0.5,
        horizon=1,
        lead_time=1,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=0,
        cumulative_target=False,
    )


def test_policy_error_fails_trial_in_hpo_path(monkeypatch) -> None:
    def _raise_policy_error(*args, **kwargs):
        raise ValueError("broken policy")

    monkeypatch.setattr("benchmarks.vn2.replay.apply_order_policy", _raise_policy_error)

    with pytest.raises(PolicyApplicationError, match="broken policy"):
        replay_cached_cost(_cache(), order_conformal_config=None)


def test_policy_error_degraded_mode_is_explicit(monkeypatch) -> None:
    def _raise_policy_error(*args, **kwargs):
        raise ValueError("broken policy")

    monkeypatch.setattr("benchmarks.vn2.replay.apply_order_policy", _raise_policy_error)

    result = replay_cached_cost(
        _cache(),
        order_conformal_config=None,
        policy_error_mode="degraded",
    )

    assert result.orders_by_round[1] == {"A": 0.0}
