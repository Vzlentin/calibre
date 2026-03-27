"""Tests for the conformal prediction module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Imports ──────────────────────────────────────────────────────────────────

def test_conformal_module_importable():
    from calibre.conformal import (  # noqa: F401
        AdaptiveConformalInference,
        ConformalPolicyConfig,
        ConformalRuntime,
        MultiStepSplitConformalInference,
        MultiStepAdaptiveConformalInference,
        OnlineConformalController,
        IntervalPrediction,
        MultiStepIntervalPrediction,
        absolute_error,
        scaled_absolute_error,
        symmetric_interval,
        symmetric_intervals,
    )


def test_conformal_policy_config_exposes_alpha_and_interval_columns():
    from calibre.conformal import ConformalPolicyConfig
    config = ConformalPolicyConfig(method="mscp", coverage=0.9, calibration_window=12)
    assert config.alpha == pytest.approx(0.1)
    assert config.interval_columns == ("lo_0p9", "hi_0p9")


def test_conformal_runtime_stamps_aci_interval_columns_and_state():
    from calibre.conformal import ConformalPolicyConfig, ConformalRuntime
    from calibre.contracts.forecast_frame import (
        CALIBRATION_STATE,
        CONFORMAL_ALPHA,
        CONFORMAL_METHOD,
        FORECAST_ORIGIN,
        H,
        MODEL_NAME,
        NONCONFORMITY_SCORE,
        UNIQUE_ID,
        Y,
        Y_HAT,
    )

    config = ConformalPolicyConfig(method="aci", coverage=0.9, calibration_window=5, gamma=0.05)
    runtime = ConformalRuntime(config)
    origin = pd.Timestamp("2024-01-07")
    future_dates = pd.date_range("2024-01-14", periods=2, freq="W")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001", "SKU_001"],
            "ds": future_dates,
            Y: [np.nan, np.nan],
            Y_HAT: [10.0, 20.0],
            H: [1, 2],
            FORECAST_ORIGIN: [origin, origin],
            MODEL_NAME: ["SeasonalNaive", "SeasonalNaive"],
        }
    )

    enriched = runtime.apply(frame)
    lower_col, upper_col = config.interval_columns
    assert lower_col in enriched.columns
    assert upper_col in enriched.columns
    assert enriched[CONFORMAL_METHOD].eq("aci").all()
    assert enriched[CONFORMAL_ALPHA].notna().all()
    assert enriched[NONCONFORMITY_SCORE].isna().all()
    assert enriched[CALIBRATION_STATE].str.startswith("{").all()


def test_conformal_runtime_updates_only_observed_aci_horizon():
    from calibre.conformal import ConformalPolicyConfig, ConformalRuntime
    from calibre.contracts.forecast_frame import FORECAST_ORIGIN, H, MODEL_NAME, UNIQUE_ID, Y, Y_HAT

    config = ConformalPolicyConfig(method="aci", coverage=0.9, calibration_window=5, gamma=0.05)
    runtime = ConformalRuntime(config)
    first_origin = pd.Timestamp("2024-01-07")
    first_frame = pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001", "SKU_001"],
            "ds": pd.date_range("2024-01-14", periods=2, freq="W"),
            Y: [np.nan, np.nan],
            Y_HAT: [10.0, 20.0],
            H: [1, 2],
            FORECAST_ORIGIN: [first_origin, first_origin],
            MODEL_NAME: ["SeasonalNaive", "SeasonalNaive"],
        }
    )
    first_enriched = runtime.apply(first_frame)
    resolved = first_enriched[first_enriched[H] == 1].copy()
    resolved[Y] = [12.0]
    observed = runtime.observe(resolved)

    second_origin = pd.Timestamp("2024-01-14")
    second_frame = pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001", "SKU_001"],
            "ds": pd.date_range("2024-01-21", periods=2, freq="W"),
            Y: [np.nan, np.nan],
            Y_HAT: [11.0, 21.0],
            H: [1, 2],
            FORECAST_ORIGIN: [second_origin, second_origin],
            MODEL_NAME: ["SeasonalNaive", "SeasonalNaive"],
        }
    )
    second_enriched = runtime.apply(second_frame)
    lower_col, upper_col = config.interval_columns
    widths = second_enriched[upper_col] - second_enriched[lower_col]

    assert observed["nonconformity_score"].iloc[0] == pytest.approx(2.0)
    assert widths.iloc[0] > 0.0
    assert widths.iloc[1] == pytest.approx(0.0)


def test_conformal_runtime_masks_mscp_warmup_bounds():
    from calibre.conformal import ConformalPolicyConfig, ConformalRuntime
    from calibre.contracts.forecast_frame import FORECAST_ORIGIN, H, MODEL_NAME, UNIQUE_ID, Y, Y_HAT

    config = ConformalPolicyConfig(method="mscp", coverage=0.9, calibration_window=5)
    runtime = ConformalRuntime(config)
    origin = pd.Timestamp("2024-01-07")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001", "SKU_001"],
            "ds": pd.date_range("2024-01-14", periods=2, freq="W"),
            Y: [np.nan, np.nan],
            Y_HAT: [10.0, 20.0],
            H: [1, 2],
            FORECAST_ORIGIN: [origin, origin],
            MODEL_NAME: ["SeasonalNaive", "SeasonalNaive"],
        }
    )

    enriched = runtime.apply(frame)
    lower_col, upper_col = config.interval_columns
    assert enriched[lower_col].isna().all()
    assert enriched[upper_col].isna().all()


# ── MultiStepSplitConformalInference ──────────────────────────────────────────

def test_mscp_predict_returns_correct_horizon():
    from calibre.conformal import MultiStepSplitConformalInference
    mscp = MultiStepSplitConformalInference(horizon=3, alpha=0.1, calibration_window=10)
    pred = mscp.predict_interval(point_forecast=np.array([1.0, 2.0, 3.0]))
    assert pred.horizon == 3


def test_mscp_uses_horizon_specific_rolling_score_buffers():
    from calibre.conformal import MultiStepSplitConformalInference
    mscp = MultiStepSplitConformalInference(horizon=2, alpha=0.1, calibration_window=2)
    mscp.observe(horizon=1, y_true=2.0, point_forecast=1.0)
    mscp.observe(horizon=1, y_true=4.0, point_forecast=2.0)
    mscp.observe(horizon=1, y_true=7.0, point_forecast=4.0)
    mscp.observe(horizon=2, y_true=9.0, point_forecast=6.0)
    diag = mscp.get_diagnostics()
    np.testing.assert_array_equal(diag["score_history"][0], np.array([2.0, 3.0]))
    np.testing.assert_array_equal(diag["score_history"][1], np.array([3.0]))


def test_mscp_emits_intervals_from_horizon_specific_quantiles():
    from calibre.conformal import MultiStepSplitConformalInference
    mscp = MultiStepSplitConformalInference(
        horizon=2,
        alpha=0.5,
        calibration_window=5,
        initial_scores=[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
    )
    pred = mscp.predict_interval(point_forecast=np.array([10.0, 20.0]))
    np.testing.assert_array_equal(pred.lower, np.array([8.0, 16.0]))
    np.testing.assert_array_equal(pred.upper, np.array([12.0, 24.0]))


def test_mscp_higher_rule_keeps_infinite_radius_during_warmup():
    from calibre.conformal import MultiStepSplitConformalInference
    mscp = MultiStepSplitConformalInference(
        horizon=1,
        alpha=0.2,
        calibration_window=10,
        initial_scores=[[1.0, 2.0, 3.0]],
    )
    radius = mscp.get_radius()
    assert radius[0] == np.inf


# ── Score functions ───────────────────────────────────────────────────────────

def test_absolute_error_scalar():
    from calibre.conformal import absolute_error
    assert absolute_error(3.0, 1.0) == pytest.approx(2.0)


def test_absolute_error_negative_residual():
    from calibre.conformal import absolute_error
    assert absolute_error(1.0, 3.0) == pytest.approx(2.0)


def test_scaled_absolute_error():
    from calibre.conformal import scaled_absolute_error
    import functools
    score_fn = functools.partial(scaled_absolute_error, scale=2.0)
    assert score_fn(3.0, 1.0) == pytest.approx(1.0)  # |3-1|/2 = 1.0


# ── IntervalPrediction ────────────────────────────────────────────────────────

def test_interval_contains_center():
    from calibre.conformal import symmetric_interval
    pred = symmetric_interval(center=5.0, radius=1.0, alpha=0.1)
    assert pred.contains(5.0)


def test_interval_contains_boundary():
    from calibre.conformal import symmetric_interval
    pred = symmetric_interval(center=5.0, radius=1.0, alpha=0.1)
    assert pred.contains(4.0)  # lower boundary
    assert pred.contains(6.0)  # upper boundary


def test_interval_excludes_outside():
    from calibre.conformal import symmetric_interval
    pred = symmetric_interval(center=5.0, radius=1.0, alpha=0.1)
    assert not pred.contains(3.9)
    assert not pred.contains(6.1)


def test_multistep_interval_contains():
    from calibre.conformal import symmetric_intervals
    pred = symmetric_intervals(
        center=np.array([1.0, 2.0, 3.0]),
        radius=np.array([0.5, 0.5, 0.5]),
        alpha=np.array([0.1, 0.1, 0.1]),
        issued_at=0,
    )
    result = pred.contains(np.array([1.0, 2.6, 3.0]))
    assert result[0]   # center of interval 0
    assert not result[1]  # outside upper of interval 1 (2.6 > 2.5)
    assert result[2]   # center of interval 2


# ── AdaptiveConformalInference ────────────────────────────────────────────────

def test_aci_alpha_decreases_on_miss():
    """When the interval misses, alpha decreases (intervals widen next step)."""
    from calibre.conformal import AdaptiveConformalInference, symmetric_interval
    aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05, alpha_bounds=None)
    alpha_before = aci.current_alpha
    # tiny interval that will miss
    pred = symmetric_interval(center=0.0, radius=0.001, alpha=alpha_before)
    aci.observe(y_true=100.0, prediction=pred)
    assert aci.current_alpha < alpha_before


def test_aci_alpha_increases_on_hit():
    """When the interval covers the truth, alpha increases (intervals narrow next step)."""
    from calibre.conformal import AdaptiveConformalInference, symmetric_interval
    aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05, alpha_bounds=None)
    alpha_before = aci.current_alpha
    # wide interval that will cover everything
    pred = symmetric_interval(center=0.0, radius=1e9, alpha=alpha_before)
    aci.observe(y_true=5.0, prediction=pred)
    assert aci.current_alpha > alpha_before


def test_aci_score_appended_to_history():
    from calibre.conformal import AdaptiveConformalInference, symmetric_interval
    aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
    pred = symmetric_interval(center=0.0, radius=1.0, alpha=aci.current_alpha)
    aci.observe(y_true=0.5, prediction=pred)
    assert len(aci.score_history) == 1
    assert aci.score_history[0] == pytest.approx(0.5)


def test_aci_trim_scores_limits_history():
    from calibre.conformal import AdaptiveConformalInference, symmetric_interval
    aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
    for value in (1.0, 2.0, 3.0):
        pred = symmetric_interval(center=0.0, radius=10.0, alpha=aci.current_alpha)
        aci.observe(y_true=value, prediction=pred)
    aci.trim_scores(2)
    np.testing.assert_array_equal(aci.score_history, np.array([2.0, 3.0]))


def test_aci_get_diagnostics_keys():
    from calibre.conformal import AdaptiveConformalInference
    aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05)
    diag = aci.get_diagnostics()
    for key in ("target_alpha", "gamma", "current_alpha", "alpha_history", "error_history", "score_history", "radius_history"):
        assert key in diag


def test_aci_update_rule_matches_formula():
    """alpha_{t+1} = alpha_t + gamma * (target_alpha - error)."""
    from calibre.conformal import AdaptiveConformalInference, symmetric_interval
    aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05, alpha_bounds=None)
    alpha0 = aci.current_alpha
    # force a miss (error=1)
    pred = symmetric_interval(center=0.0, radius=0.0, alpha=alpha0)
    aci.observe(y_true=1.0, prediction=pred)
    expected = alpha0 + 0.05 * (0.1 - 1)
    assert aci.current_alpha == pytest.approx(expected)


def test_aci_initial_scores_seed_radius():
    """Pre-seeded scores affect the first radius computation."""
    from calibre.conformal import AdaptiveConformalInference
    aci = AdaptiveConformalInference(alpha=0.1, gamma=0.05, initial_scores=[1.0, 2.0, 3.0])
    radius = aci.get_radius()
    assert radius > 0.0


def test_finite_sample_radius_higher_returns_inf_for_small_alpha():
    """'higher' rule returns inf when alpha <= 1/(n+1)."""
    from calibre.conformal.aci import _finite_sample_radius
    scores = [1.0, 2.0, 3.0]  # n=3
    # alpha <= 1/4 should trigger inf
    result = _finite_sample_radius(scores, alpha=0.2, default_radius=0.0, quantile_rule="higher")
    assert result == np.inf


def test_finite_sample_radius_conformal_never_returns_inf():
    """'conformal' rule never returns inf (uses max score instead)."""
    from calibre.conformal.aci import _finite_sample_radius
    scores = [1.0, 2.0, 3.0]  # n=3
    # same small alpha as above
    result = _finite_sample_radius(scores, alpha=0.2, default_radius=0.0, quantile_rule="conformal")
    assert np.isfinite(result)
    assert result == pytest.approx(3.0)  # returns max score


# ── MultiStepAdaptiveConformalInference ───────────────────────────────────────

def test_multistep_predict_returns_correct_horizon():
    from calibre.conformal import MultiStepAdaptiveConformalInference
    aci = MultiStepAdaptiveConformalInference(horizon=3, alpha=0.1, gamma=0.05)
    pred = aci.predict_interval(point_forecast=np.array([1.0, 2.0, 3.0]))
    assert pred.horizon == 3


def test_multistep_horizon_1_matches_single_step_update():
    """With horizon=1 and one cycle, alpha update should match single-step formula."""
    from calibre.conformal import MultiStepAdaptiveConformalInference
    aci = MultiStepAdaptiveConformalInference(
        horizon=1, alpha=0.1, gamma=0.05, alpha_bounds=None
    )
    alpha0 = aci.current_alpha[0]
    aci.predict_interval(point_forecast=np.array([0.0]))
    result = aci.observe(y_true=1000.0)  # guaranteed miss
    # error=1 miss: alpha + gamma*(target - 1) = alpha0 + 0.05*(0.1-1)
    expected = alpha0 + 0.05 * (0.1 - 1.0)
    assert aci.current_alpha[0] == pytest.approx(expected)


def test_multistep_delayed_feedback_horizon_alignment():
    """
    For horizon=2, the observation at t=2 should update:
      - h=1: using prediction issued at t=1 (its 0th element)
      - h=2: using prediction issued at t=0 (its 1st element)
    """
    from calibre.conformal import MultiStepAdaptiveConformalInference
    aci = MultiStepAdaptiveConformalInference(
        horizon=2, alpha=0.1, gamma=0.05, alpha_bounds=None
    )
    # issue prediction at t=0
    pred0 = aci.predict_interval(point_forecast=np.array([10.0, 20.0]))
    # issue prediction at t=1
    pred1 = aci.predict_interval(point_forecast=np.array([10.0, 20.0]))

    # observe at t=1: only h=1 can be resolved (pred issued at t=0, element 0)
    result1 = aci.observe(y_true=10.0)
    assert result1["observed_mask"][0]   # h=1 resolved
    assert not result1["observed_mask"][1]  # h=2 not yet

    # observe at t=2: both h=1 (pred issued t=1, elem 0) and h=2 (pred issued t=0, elem 1) resolved
    result2 = aci.observe(y_true=20.0)
    assert result2["observed_mask"][0]   # h=1 resolved
    assert result2["observed_mask"][1]   # h=2 resolved


def test_multistep_unobserved_steps_are_nan_in_error_history():
    """
    When a horizon step has no resolved prediction, its entry in error_history
    should be np.nan, not the target_alpha float.
    """
    from calibre.conformal import MultiStepAdaptiveConformalInference
    aci = MultiStepAdaptiveConformalInference(
        horizon=2, alpha=0.1, gamma=0.05, alpha_bounds=None
    )
    aci.predict_interval(point_forecast=np.array([0.0, 0.0]))

    # First observation: only h=1 can be resolved (h=2 has no prediction issued 2 steps ago)
    aci.observe(y_true=0.0)
    diag = aci.get_diagnostics()
    error_history = diag["error_history"]  # shape (1, 2)

    assert error_history.shape == (1, 2)
    assert error_history[0, 0] in (0, 1)   # h=1 was observed — must be 0 or 1
    assert np.isnan(error_history[0, 1])   # h=2 was not observed — must be nan


def test_multistep_pending_predictions_cleaned_up():
    """Predictions older than horizon steps are removed from pending."""
    from calibre.conformal import MultiStepAdaptiveConformalInference
    aci = MultiStepAdaptiveConformalInference(
        horizon=2, alpha=0.1, gamma=0.05, alpha_bounds=None
    )
    aci.predict_interval(point_forecast=np.array([0.0, 0.0]))
    aci.predict_interval(point_forecast=np.array([0.0, 0.0]))

    aci.observe(y_true=0.0)  # t=1: pred at t=0 not yet expired (h=2 covers t=2)
    assert aci.get_diagnostics()["pending_predictions"] >= 1

    aci.observe(y_true=0.0)  # t=2: pred at t=0 now expired (horizon=2, issued at 0)
    # after t=2, prediction issued at t=0 should be cleaned up
    assert aci.get_diagnostics()["pending_predictions"] <= 1


def test_multistep_score_history_populated_per_horizon():
    """Each horizon step accumulates scores independently."""
    from calibre.conformal import MultiStepAdaptiveConformalInference
    aci = MultiStepAdaptiveConformalInference(
        horizon=2, alpha=0.1, gamma=0.05, alpha_bounds=None
    )
    aci.predict_interval(point_forecast=np.array([5.0, 10.0]))
    aci.predict_interval(point_forecast=np.array([5.0, 10.0]))

    aci.observe(y_true=6.0)   # t=1: h=1 gets score |6-5|=1.0
    aci.observe(y_true=12.0)  # t=2: h=1 gets score |12-5|=7.0, h=2 gets score |12-10|=2.0

    diag = aci.get_diagnostics()
    h1_scores = diag["score_history"][0]
    h2_scores = diag["score_history"][1]

    assert len(h1_scores) == 2
    assert len(h2_scores) == 1
    assert h1_scores[0] == pytest.approx(1.0)
    assert h1_scores[1] == pytest.approx(7.0)
    assert h2_scores[0] == pytest.approx(2.0)
