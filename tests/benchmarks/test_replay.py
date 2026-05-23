"""Tests for benchmarks.vn2.replay — policy error behavior in HPO vs degraded mode."""

from __future__ import annotations

import pytest

from benchmarks.vn2.replay import replay_cached_cost


def test_policy_error_fails_trial_in_hpo_path(monkeypatch):
    """With raise_on_policy_error=True, a ValueError from apply_order_policy propagates.

    This is the HPO path behavior: a broken config should fail the trial
    (get inf cost), not silently return zero orders (which would look cheap).
    """

    import pandas as pd

    from benchmarks.vn2.replay import (
        CachedRound,
        VN2ReplayCache,
    )
    from benchmarks.vn2.simulator import ProductState
    from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H, Y

    # Build a minimal cache with one round that will trigger a policy error.
    initial_states = {
        "sku-a": ProductState(
            unique_id="sku-a",
            end_inventory=10.0,
            in_transit_w1=0.0,
            in_transit_w2=0.0,
            cumulative_holding_cost=0.0,
            cumulative_shortage_cost=0.0,
        )
    }
    origin = pd.Timestamp("2024-01-08")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["sku-a"],
            DS: [origin],
            Y_HAT: [5.0],
            H: [1],
            Y: [float("nan")],
        }
    )
    cache = VN2ReplayCache(
        initial_states=initial_states,
        warmup_frames=[],
        rounds={1: CachedRound(round_num=1, origin=origin, frame=frame)},
        actuals_by_round={1: {"sku-a": 3.0}},
        model_config={"quantiles": [0.5]},
        quantile_alpha=0.5,
        horizon=1,
        lead_time=1,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=0,
        cumulative_target=False,
    )

    # Monkeypatch apply_order_policy to raise ValueError
    monkeypatch.setattr(
        "benchmarks.vn2.replay.apply_order_policy",
        lambda frame, config: (_ for _ in ()).throw(ValueError("policy broken")),
    )

    # In HPO mode (raise_on_policy_error=True), the error must propagate.
    with pytest.raises(ValueError, match="policy broken"):
        replay_cached_cost(cache, order_conformal_config=None, raise_on_policy_error=True)


def test_policy_error_uses_zero_orders_in_degraded_mode(monkeypatch):
    """With raise_on_policy_error=False (default), policy errors → zero orders."""
    import pandas as pd

    from benchmarks.vn2.replay import CachedRound, VN2ReplayCache
    from benchmarks.vn2.simulator import ProductState
    from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H, Y

    initial_states = {
        "sku-a": ProductState(
            unique_id="sku-a",
            end_inventory=10.0,
            in_transit_w1=0.0,
            in_transit_w2=0.0,
            cumulative_holding_cost=0.0,
            cumulative_shortage_cost=0.0,
        )
    }
    origin = pd.Timestamp("2024-01-08")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["sku-a"],
            DS: [origin],
            Y_HAT: [5.0],
            H: [1],
            Y: [float("nan")],
        }
    )
    cache = VN2ReplayCache(
        initial_states=initial_states,
        warmup_frames=[],
        rounds={1: CachedRound(round_num=1, origin=origin, frame=frame)},
        actuals_by_round={1: {"sku-a": 3.0}},
        model_config={"quantiles": [0.5]},
        quantile_alpha=0.5,
        horizon=1,
        lead_time=1,
        review_period=1,
        decision_rounds=1,
        delivery_weeks=0,
        cumulative_target=False,
    )

    errors: list[Exception] = []
    monkeypatch.setattr(
        "benchmarks.vn2.replay.apply_order_policy",
        lambda frame, config: (_ for _ in ()).throw(ValueError("policy broken")),
    )

    result = replay_cached_cost(
        cache,
        order_conformal_config=None,
        on_policy_error=lambda rn, exc: errors.append(exc),
        raise_on_policy_error=False,
    )
    assert errors, "expected on_policy_error to be called"
    assert result.orders_by_round[1] == {"sku-a": 0.0}
