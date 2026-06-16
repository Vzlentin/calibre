"""Tests for the conformal prediction module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_conformal_policy_config_exposes_alpha_and_interval_columns():
    from calibre.conformal import SymmetricIntervalConfig

    config = SymmetricIntervalConfig(method="mscp", coverage=0.9, calibration_window=12)
    assert config.alpha == pytest.approx(0.1)
    assert config.interval_columns == ("lo_0p9", "hi_0p9")


def test_conformal_runtime_stamps_aci_interval_columns_and_state():
    from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
    from calibre.core.forecast_frame import (
        CALIBRATION_STATE,
        CALIBRATION_STATE_REF,
        CONFORMAL_ALPHA,
        CONFORMAL_METHOD,
        CONFORMAL_PARTITION,
        FORECAST_ORIGIN,
        MODEL_NAME,
        NONCONFORMITY_SCORE,
        UNIQUE_ID,
        Y_HAT,
        H,
        Y,
    )

    config = SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=5, gamma=0.05)
    runtime = SymmetricIntervalRuntime(config)
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
    assert CALIBRATION_STATE not in enriched.columns
    assert enriched[CALIBRATION_STATE_REF].str.startswith("aci:perhorizon:").all()
    assert enriched[CONFORMAL_PARTITION].str.contains("SeasonalNaive:h").all()


def test_conformal_runtime_updates_only_observed_aci_horizon():
    from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
    from calibre.core.forecast_frame import FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y

    config = SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=5, gamma=0.05)
    runtime = SymmetricIntervalRuntime(config)
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
    from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
    from calibre.core.forecast_frame import FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y

    config = SymmetricIntervalConfig(method="mscp", coverage=0.9, calibration_window=5)
    runtime = SymmetricIntervalRuntime(config)
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


def test_absolute_error_scalar():
    from calibre.conformal import absolute_error

    assert absolute_error(3.0, 1.0) == pytest.approx(2.0)


def test_absolute_error_negative_residual():
    from calibre.conformal import absolute_error

    assert absolute_error(1.0, 3.0) == pytest.approx(2.0)


def test_scaled_absolute_error():
    import functools

    from calibre.conformal import scaled_absolute_error

    score = functools.partial(scaled_absolute_error, scale=2.0)
    assert score(3.0, 1.0) == pytest.approx(1.0)  # |3-1|/2 = 1.0


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
    assert result[0]  # center of interval 0
    assert not result[1]  # outside upper of interval 1 (2.6 > 2.5)
    assert result[2]  # center of interval 2


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
    for key in (
        "target_alpha",
        "gamma",
        "current_alpha",
        "alpha_history",
        "error_history",
        "score_history",
        "radius_history",
    ):
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


def test_higher_quantile_rule_returns_inf_for_small_alpha():
    """'higher' rule yields an infinite radius when alpha <= 1/(n+1).

    Driven through the public ``get_radius`` surface: with 3 seeded scores and
    alpha=0.2 (<= 1/4) the finite-sample 'higher' quantile is unattainable.
    """
    from calibre.conformal import AdaptiveConformalInference

    aci = AdaptiveConformalInference(
        alpha=0.2, gamma=0.05, initial_scores=[1.0, 2.0, 3.0], quantile_rule="higher"
    )
    assert aci.get_radius() == np.inf


def test_conformal_quantile_rule_never_returns_inf():
    """'conformal' rule caps the radius at the max score instead of inf."""
    from calibre.conformal import AdaptiveConformalInference

    aci = AdaptiveConformalInference(
        alpha=0.2, gamma=0.05, initial_scores=[1.0, 2.0, 3.0], quantile_rule="conformal"
    )
    radius = aci.get_radius()
    assert np.isfinite(radius)
    assert radius == pytest.approx(3.0)  # max seeded score


def test_multistep_predict_returns_correct_horizon():
    from calibre.conformal import MultiStepAdaptiveConformalInference

    aci = MultiStepAdaptiveConformalInference(horizon=3, alpha=0.1, gamma=0.05)
    pred = aci.predict_interval(point_forecast=np.array([1.0, 2.0, 3.0]))
    assert pred.horizon == 3


def test_multistep_horizon_1_matches_single_step_update():
    """With horizon=1 and one cycle, alpha update should match single-step formula."""
    from calibre.conformal import MultiStepAdaptiveConformalInference

    aci = MultiStepAdaptiveConformalInference(horizon=1, alpha=0.1, gamma=0.05, alpha_bounds=None)
    alpha0 = aci.current_alpha[0]
    aci.predict_interval(point_forecast=np.array([0.0]))
    _ = aci.observe(y_true=1000.0)  # guaranteed miss
    # error=1 miss: alpha + gamma*(target - 1) = alpha0 + 0.05*(0.1-1)
    expected = alpha0 + 0.05 * (0.1 - 1.0)
    assert aci.current_alpha[0] == pytest.approx(expected)


def test_multistep_delayed_feedback_horizon_alignment():
    """Observation at t=2 resolves both horizons by issue time.

    For horizon=2, the observation at t=2 should update:
      - h=1: using prediction issued at t=1 (its 0th element)
      - h=2: using prediction issued at t=0 (its 1st element)
    """
    from calibre.conformal import MultiStepAdaptiveConformalInference

    aci = MultiStepAdaptiveConformalInference(horizon=2, alpha=0.1, gamma=0.05, alpha_bounds=None)
    # issue prediction at t=0
    _ = aci.predict_interval(point_forecast=np.array([10.0, 20.0]))
    # issue prediction at t=1
    _ = aci.predict_interval(point_forecast=np.array([10.0, 20.0]))

    # observe at t=1: only h=1 can be resolved (pred issued at t=0, element 0)
    result1 = aci.observe(y_true=10.0)
    assert result1["observed_mask"][0]  # h=1 resolved
    assert not result1["observed_mask"][1]  # h=2 not yet

    # observe at t=2: both h=1 (pred issued t=1, elem 0) and h=2 (pred issued t=0, elem 1) resolved
    result2 = aci.observe(y_true=20.0)
    assert result2["observed_mask"][0]  # h=1 resolved
    assert result2["observed_mask"][1]  # h=2 resolved


def test_multistep_unobserved_steps_are_nan_in_error_history():
    """Unobserved horizon steps stay NaN in the error history.

    When a horizon step has no resolved prediction, its entry in error_history
    should be np.nan, not the target_alpha float.
    """
    from calibre.conformal import MultiStepAdaptiveConformalInference

    aci = MultiStepAdaptiveConformalInference(horizon=2, alpha=0.1, gamma=0.05, alpha_bounds=None)
    aci.predict_interval(point_forecast=np.array([0.0, 0.0]))

    # First observation: only h=1 can be resolved (h=2 has no prediction issued 2 steps ago)
    aci.observe(y_true=0.0)
    diag = aci.get_diagnostics()
    error_history = diag["error_history"]  # shape (1, 2)

    assert error_history.shape == (1, 2)
    assert error_history[0, 0] in (0, 1)  # h=1 was observed — must be 0 or 1
    assert np.isnan(error_history[0, 1])  # h=2 was not observed — must be nan


def test_multistep_pending_predictions_cleaned_up():
    """Predictions older than horizon steps are removed from pending."""
    from calibre.conformal import MultiStepAdaptiveConformalInference

    aci = MultiStepAdaptiveConformalInference(horizon=2, alpha=0.1, gamma=0.05, alpha_bounds=None)
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

    aci = MultiStepAdaptiveConformalInference(horizon=2, alpha=0.1, gamma=0.05, alpha_bounds=None)
    aci.predict_interval(point_forecast=np.array([5.0, 10.0]))
    aci.predict_interval(point_forecast=np.array([5.0, 10.0]))

    aci.observe(y_true=6.0)  # t=1: h=1 gets score |6-5|=1.0
    aci.observe(y_true=12.0)  # t=2: h=1 gets score |12-5|=7.0, h=2 gets score |12-10|=2.0

    diag = aci.get_diagnostics()
    h1_scores = diag["score_history"][0]
    h2_scores = diag["score_history"][1]

    assert len(h1_scores) == 2
    assert len(h2_scores) == 1
    assert h1_scores[0] == pytest.approx(1.0)
    assert h1_scores[1] == pytest.approx(7.0)
    assert h2_scores[0] == pytest.approx(2.0)


def test_cumulative_controller_predicts_only_at_terminal_horizon():
    from calibre.conformal import CumulativeSplitConformalInference

    controller = CumulativeSplitConformalInference(
        protection_period=3,
        alpha=0.5,
        calibration_window=5,
        initial_scores=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    prediction = controller.predict_interval(np.array([10.0, 20.0, 30.0]))
    assert np.isnan(prediction.center[0])
    assert np.isnan(prediction.center[1])
    assert prediction.center[2] == pytest.approx(60.0)
    assert np.isfinite(prediction.lower[2])
    assert np.isfinite(prediction.upper[2])


def test_cumulative_controller_appends_window_score():
    from calibre.conformal import CumulativeSplitConformalInference

    controller = CumulativeSplitConformalInference(
        protection_period=3, alpha=0.1, calibration_window=5
    )
    controller.observe(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 4.0]))
    diag = controller.get_diagnostics()
    np.testing.assert_array_equal(diag["score_history"], np.array([abs(6.0 - 8.0)]))


def test_cumulative_controller_rejects_window_size_mismatch():
    from calibre.conformal import CumulativeSplitConformalInference

    controller = CumulativeSplitConformalInference(
        protection_period=3, alpha=0.1, calibration_window=5
    )
    with pytest.raises(ValueError, match="length 3"):
        controller.observe(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_cumulative_controller_ready_mask_only_at_terminal():
    from calibre.conformal import CumulativeSplitConformalInference

    controller = CumulativeSplitConformalInference(
        protection_period=3,
        alpha=0.5,
        calibration_window=5,
        initial_scores=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    mask = controller.ready_mask()
    assert mask.tolist() == [False, False, True]


def test_cumulative_controller_ready_mask_blocked_when_buffer_empty():
    from calibre.conformal import CumulativeSplitConformalInference

    controller = CumulativeSplitConformalInference(
        protection_period=3, alpha=0.5, calibration_window=5
    )
    assert controller.ready_mask().tolist() == [False, False, False]


def _build_cumulative_runtime(
    protection_period: int = 3,
    calibration_window: int = 5,
    coverage: float = 0.5,
):
    from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime

    config = SymmetricIntervalConfig(
        method="mscp",
        coverage=coverage,
        calibration_window=calibration_window,
        mode="cumulative",
        protection_period=protection_period,
    )
    return SymmetricIntervalRuntime(config), config


def _cumulative_frame(
    *,
    unique_id: str,
    origin: pd.Timestamp,
    y_hats: list[float],
    y_actuals: list[float] | None = None,
):
    from calibre.core.forecast_frame import (
        FORECAST_ORIGIN,
        MODEL_NAME,
        UNIQUE_ID,
        Y_HAT,
        H,
        Y,
    )

    horizon = len(y_hats)
    actuals = y_actuals if y_actuals is not None else [np.nan] * horizon
    return pd.DataFrame(
        {
            UNIQUE_ID: [unique_id] * horizon,
            "ds": pd.date_range(origin + pd.Timedelta(weeks=1), periods=horizon, freq="W"),
            Y: actuals,
            Y_HAT: y_hats,
            H: list(range(1, horizon + 1)),
            FORECAST_ORIGIN: [origin] * horizon,
            MODEL_NAME: ["SeasonalNaive"] * horizon,
        }
    )


def test_cumulative_runtime_writes_marker_column_on_every_row():
    from calibre.core.forecast_frame import CONFORMAL_MODE

    runtime, config = _build_cumulative_runtime()
    frame = _cumulative_frame(
        unique_id="SKU_001",
        origin=pd.Timestamp("2024-01-07"),
        y_hats=[10.0, 20.0, 30.0],
    )
    enriched = runtime.apply(frame)
    assert (enriched[CONFORMAL_MODE] == "cumulative").all()


def test_cumulative_runtime_emits_finite_bound_only_at_protection_period():
    runtime, config = _build_cumulative_runtime(protection_period=3, calibration_window=5)
    for week_offset in range(3):
        origin = pd.Timestamp("2024-01-07") + pd.Timedelta(weeks=week_offset)
        frame = _cumulative_frame(
            unique_id="SKU_001",
            origin=origin,
            y_hats=[10.0, 20.0, 30.0],
            y_actuals=[12.0, 22.0, 31.0],
        )
        enriched = runtime.apply(frame)
        runtime.observe(enriched)

    new_origin = pd.Timestamp("2024-02-04")
    fresh_frame = _cumulative_frame(
        unique_id="SKU_001",
        origin=new_origin,
        y_hats=[10.0, 20.0, 30.0],
    )
    enriched = runtime.apply(fresh_frame)
    lower_col, upper_col = config.interval_columns
    assert pd.isna(enriched[lower_col].iloc[0])
    assert pd.isna(enriched[lower_col].iloc[1])
    assert pd.notna(enriched[upper_col].iloc[2])
    assert enriched[upper_col].iloc[2] >= 60.0


def test_cumulative_runtime_observe_appends_one_score_per_window():
    from calibre.core.forecast_frame import NONCONFORMITY_SCORE

    runtime, _ = _build_cumulative_runtime(protection_period=3)
    origin = pd.Timestamp("2024-01-07")
    frame = _cumulative_frame(
        unique_id="SKU_001",
        origin=origin,
        y_hats=[10.0, 20.0, 30.0],
        y_actuals=[12.0, 22.0, 31.0],
    )
    enriched = runtime.apply(frame)
    observed = runtime.observe(enriched)

    scores = observed[NONCONFORMITY_SCORE].dropna().tolist()
    assert len(scores) == 1
    assert scores[0] == pytest.approx(abs(65.0 - 60.0))


def test_cumulative_runtime_observe_skips_partially_resolved_windows():
    from calibre.core.forecast_frame import NONCONFORMITY_SCORE

    runtime, _ = _build_cumulative_runtime(protection_period=3)
    origin = pd.Timestamp("2024-01-07")
    frame = _cumulative_frame(
        unique_id="SKU_001",
        origin=origin,
        y_hats=[10.0, 20.0, 30.0],
        y_actuals=[12.0, np.nan, np.nan],
    )
    enriched = runtime.apply(frame)
    observed = runtime.observe(enriched)
    assert observed[NONCONFORMITY_SCORE].isna().all()


def test_cumulative_config_requires_protection_period():
    from calibre.conformal import SymmetricIntervalConfig

    with pytest.raises(ValueError, match="protection_period"):
        SymmetricIntervalConfig(method="mscp", coverage=0.9, mode="cumulative")


def test_cumulative_config_rejects_aci_method():
    from calibre.conformal import SymmetricIntervalConfig

    with pytest.raises(ValueError, match="cumulative mode"):
        SymmetricIntervalConfig(method="aci", coverage=0.9, mode="cumulative", protection_period=3)


def test_cumulative_controller_rejects_array_alpha():
    from calibre.conformal import CumulativeSplitConformalInference

    with pytest.raises(ValueError, match="scalar"):
        CumulativeSplitConformalInference(
            protection_period=3, alpha=np.array([0.1, 0.2]), calibration_window=5
        )


def test_cumulative_runtime_pads_horizon_beyond_protection_period():
    """When horizon > protection_period, only h=K gets a finite bound; h>K is NaN."""
    from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
    from calibre.core.forecast_frame import FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y

    config = SymmetricIntervalConfig(
        method="mscp",
        coverage=0.5,
        calibration_window=5,
        mode="cumulative",
        protection_period=2,
    )
    runtime = SymmetricIntervalRuntime(config)
    lower_col, upper_col = config.interval_columns

    # Prime buffer with two observations
    for i in range(2):
        origin = pd.Timestamp("2024-01-07") + pd.Timedelta(weeks=i)
        frame = pd.DataFrame(
            {
                UNIQUE_ID: ["A"] * 4,
                "ds": pd.date_range(origin + pd.Timedelta(weeks=1), periods=4, freq="W"),
                Y: [10.0, 11.0, 12.0, 13.0],
                Y_HAT: [10.0, 11.0, 12.0, 13.0],
                H: [1, 2, 3, 4],
                FORECAST_ORIGIN: [origin] * 4,
                MODEL_NAME: ["M"] * 4,
            }
        )
        enriched = runtime.apply(frame)
        runtime.observe(enriched)

    fresh_origin = pd.Timestamp("2024-01-07") + pd.Timedelta(weeks=2)
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A"] * 4,
            "ds": pd.date_range(fresh_origin + pd.Timedelta(weeks=1), periods=4, freq="W"),
            Y: [np.nan] * 4,
            Y_HAT: [10.0, 11.0, 12.0, 13.0],
            H: [1, 2, 3, 4],
            FORECAST_ORIGIN: [fresh_origin] * 4,
            MODEL_NAME: ["M"] * 4,
        }
    )
    enriched = runtime.apply(frame)

    # h=1 and h=2 (protection_period=2): only h=2 gets finite bound
    assert pd.isna(enriched[upper_col].iloc[0]), "h=1 should be NaN"
    assert pd.notna(enriched[upper_col].iloc[1]), "h=2 (terminal) should be finite"
    # h=3, h=4 beyond protection_period: padded to NaN
    assert pd.isna(enriched[upper_col].iloc[2]), "h=3 beyond protection_period should be NaN"
    assert pd.isna(enriched[upper_col].iloc[3]), "h=4 beyond protection_period should be NaN"


def test_cumulative_runtime_observe_raises_on_duplicate_horizons():
    runtime, _ = _build_cumulative_runtime(protection_period=3)
    origin = pd.Timestamp("2024-01-07")
    frame = _cumulative_frame(
        unique_id="SKU_001",
        origin=origin,
        y_hats=[10.0, 20.0, 30.0],
        y_actuals=[12.0, 22.0, 31.0],
    )
    enriched = runtime.apply(frame)
    # Inject a duplicate h=1 row
    dup = enriched.iloc[[0]].copy()
    with pytest.raises(ValueError, match="Duplicate H"):
        runtime.observe(pd.concat([enriched, dup]))


def test_predict_batch_matches_per_row_predict_across_window_lengths():
    import numpy as np

    from calibre.conformal.calibrators import RollingQuantileCalibrator

    for rule in ("higher", "conformal"):
        for ready_on_empty in (False, True):
            calibrator = RollingQuantileCalibrator(
                calibration_window=10,
                quantile_rule=rule,
                ready_on_empty=ready_on_empty,
            )
            rng = np.random.default_rng(11)
            partitions = []
            for n in range(0, 14):
                partition = f"m:h1:series_{n}"
                partitions.append(partition)
                for score in rng.random(n):
                    calibrator.update(float(score), partition)
            partitions.append("m:h1:never_seen")

            for alpha in (0.05, 0.1, 0.5, 1.0 / 11.0):
                radii, ready = calibrator.predict_batch(alpha, partitions)
                for position, partition in enumerate(partitions):
                    expected_radius = calibrator.predict(alpha, partition)
                    expected_issue = calibrator.ready(partition, alpha) and np.isfinite(
                        expected_radius
                    )
                    assert radii[position] == expected_radius or (
                        np.isinf(radii[position]) and np.isinf(expected_radius)
                    ), (rule, ready_on_empty, alpha, partition)
                    assert bool(ready[position] and np.isfinite(radii[position])) == bool(
                        expected_issue
                    ), (rule, ready_on_empty, alpha, partition)


def test_default_runtime_composes_analytic_radius_spread():
    from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
    from calibre.conformal.spread import AnalyticRadius

    config = SymmetricIntervalConfig(method="mscp", coverage=0.9, calibration_window=5)
    runtime = SymmetricIntervalRuntime(config)
    # The seam is wired, not bypassed by a surviving inline copy.
    assert isinstance(runtime.spread, AnalyticRadius)


def test_perhorizon_apply_delegates_interval_arithmetic_to_spread():
    from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
    from calibre.core.forecast_frame import (
        FORECAST_ORIGIN,
        MODEL_NAME,
        UNIQUE_ID,
        Y_HAT,
        H,
        Y,
    )

    config = SymmetricIntervalConfig(method="mscp", coverage=0.5, calibration_window=10)
    runtime = SymmetricIntervalRuntime(config)
    origin = pd.Timestamp("2024-01-07")
    centers = [10.0, 20.0]
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["SKU_001", "SKU_001"],
            "ds": pd.date_range("2024-01-14", periods=2, freq="W"),
            Y: [np.nan, np.nan],
            Y_HAT: centers,
            H: [1, 2],
            FORECAST_ORIGIN: [origin, origin],
            MODEL_NAME: ["SeasonalNaive", "SeasonalNaive"],
        }
    )
    # Seed only h=1's partition so it issues a finite bound while h=2 stays in
    # warmup (NaN) — exercises the issued + not-yet-ready split in one fixture.
    h1_partition = runtime._partition_values(frame)[0]
    for score in (1.0, 2.0, 3.0, 4.0, 5.0):
        runtime.calibrator.update(score, h1_partition)

    alpha = float(runtime.controller.get_alpha())
    radii, ready = runtime.calibrator.predict_batch(alpha, runtime._partition_values(frame))
    issue = ready & np.isfinite(radii)
    # Ground truth = the pre-refactor inline arithmetic, computed here directly.
    with np.errstate(invalid="ignore"):
        expected_lower = np.where(issue, np.asarray(centers) - radii, np.nan)
        expected_upper = np.where(issue, np.asarray(centers) + radii, np.nan)

    enriched = runtime.apply(frame)
    lower_col, upper_col = config.interval_columns

    assert np.isfinite(expected_lower[0]), "fixture must issue h=1 to anchor the mutation check"
    assert np.isnan(expected_upper[1]), "fixture must leave h=2 in warmup"
    np.testing.assert_array_equal(enriched[lower_col].to_numpy(dtype=float), expected_lower)
    np.testing.assert_array_equal(enriched[upper_col].to_numpy(dtype=float), expected_upper)


def test_cumulative_apply_delegates_terminal_arithmetic_to_spread():
    runtime, config = _build_cumulative_runtime(protection_period=2, calibration_window=5)
    lower_col, upper_col = config.interval_columns

    # Warm the cumulative partition so the terminal row issues a finite bound.
    origin = pd.Timestamp("2024-01-07")
    for week in range(5):
        warm_origin = origin + pd.Timedelta(weeks=week)
        warm = _cumulative_frame(
            unique_id="SKU_001",
            origin=warm_origin,
            y_hats=[10.0, 20.0],
            y_actuals=[11.0, 22.0],
        )
        runtime.observe(runtime.apply(warm))

    center = 10.0 + 20.0
    fresh_origin = origin + pd.Timedelta(weeks=10)
    frame = _cumulative_frame(
        unique_id="SKU_001",
        origin=fresh_origin,
        y_hats=[10.0, 20.0],
    )
    # Ground truth: the inline ``center +/- radius`` at the terminal row.
    partition = runtime._partition_for_row(frame.iloc[-1], cumulative=True)
    radius = runtime.calibrator.predict(runtime.controller.get_alpha(), partition)
    assert np.isfinite(radius), "fixture must issue the terminal row to anchor the mutation check"

    enriched = runtime.apply(frame)
    # h=1 (non-terminal) stays NaN; h=2 (terminal) carries center +/- radius.
    assert pd.isna(enriched[lower_col].iloc[0])
    assert pd.isna(enriched[upper_col].iloc[0])
    # Exact equality (not approx): this is a byte-identical refactor, so the
    # delegated cumulative path must reproduce the inline scalar arithmetic to
    # the bit — a 1-ULP drift must fail here.
    assert enriched[lower_col].iloc[1] == center - float(radius)
    assert enriched[upper_col].iloc[1] == center + float(radius)


def _coherent_hierarchy_index():
    from calibre.reconciliation.summing import build_hierarchy_index

    return build_hierarchy_index(pd.DataFrame({"unique_id": ["A", "B"], "group": ["g", "g"]}))


def test_coherent_draws_selector_composes_coherent_spread_with_s_at_construction():
    from calibre.conformal.coherent_draws import CoherentDraws
    from calibre.conformal.runtime import (
        SymmetricIntervalConfig,
        build_symmetric_interval_runtime,
    )

    config = SymmetricIntervalConfig(
        method="aci", coverage=0.9, calibration_window=5, spread="coherent_draws", draw_count=64
    )
    index = _coherent_hierarchy_index()
    runtime = build_symmetric_interval_runtime(config, hierarchy_index=index)

    assert isinstance(runtime.spread, CoherentDraws)
    # S is folded in at construction, keyed to the index node labels.
    assert runtime.spread.summing.node_labels == index.node_labels
    assert runtime.spread.draw_count == 64
    assert runtime.requires_fitted_values is True


def test_analytic_selector_default_does_not_require_fitted_values():
    from calibre.conformal.runtime import (
        SymmetricIntervalConfig,
        build_symmetric_interval_runtime,
    )
    from calibre.conformal.spread import AnalyticRadius

    config = SymmetricIntervalConfig(method="mscp", coverage=0.9, calibration_window=5)
    runtime = build_symmetric_interval_runtime(config)
    assert isinstance(runtime.spread, AnalyticRadius)
    assert runtime.requires_fitted_values is False


def test_coherent_draws_without_hierarchy_index_fails_fast():
    from calibre.conformal.runtime import (
        SymmetricIntervalConfig,
        build_symmetric_interval_runtime,
    )

    config = SymmetricIntervalConfig(
        method="aci", coverage=0.9, calibration_window=5, spread="coherent_draws"
    )
    with pytest.raises(ValueError, match="hierarchy_index"):
        build_symmetric_interval_runtime(config)


def test_coherent_draws_runtime_round_trips_resume_state():
    from calibre.conformal.coherent_draws import CoherentDraws
    from calibre.conformal.runtime import (
        SymmetricIntervalConfig,
        SymmetricIntervalRuntime,
    )

    config = SymmetricIntervalConfig(
        method="aci", coverage=0.9, calibration_window=5, spread="coherent_draws"
    )
    index = _coherent_hierarchy_index()
    runtime = SymmetricIntervalRuntime(config, hierarchy_index=index)
    state = runtime.get_resume_state()

    restored = SymmetricIntervalRuntime.from_state(config, state, hierarchy_index=index)
    assert isinstance(restored.spread, CoherentDraws)
    assert restored.spread.summing.node_labels == index.node_labels

    partition_states = runtime.get_partition_states()
    restored_partitioned = SymmetricIntervalRuntime.from_partition_states(
        config, partition_states, hierarchy_index=index
    )
    assert isinstance(restored_partitioned.spread, CoherentDraws)


def test_config_rejects_unknown_spread_and_nonpositive_draw_count():
    from calibre.conformal.runtime import SymmetricIntervalConfig

    with pytest.raises(ValueError, match="spread"):
        SymmetricIntervalConfig(method="mscp", spread="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="draw_count"):
        SymmetricIntervalConfig(method="mscp", spread="coherent_draws", draw_count=0)


# --- Held-out width ownership (end-to-end reduced-lattice) -------------------

_U4_MODEL = "model"


def _u4_config(window: int = 40):
    from calibre.conformal.partitions import series_partition
    from calibre.conformal.runtime import SymmetricIntervalConfig

    # series_partition keys the per-partition calibrator on the node id, so the
    # held-out radius is per (model, horizon, node) — the key the coherent spread
    # re-anchors each bottom node's width to.
    return SymmetricIntervalConfig(
        method="mscp",
        coverage=0.9,
        calibration_window=window,
        spread="coherent_draws",
        draw_count=400,
        draw_seed=11,
        partition_key=series_partition,
    )


def _u4_fitted(origin, residuals_by_node, *, periods=30):
    from calibre.core.forecast_frame import (
        DS,
        FITTED_Y_HAT,
        MODEL_NAME,
        UNIQUE_ID,
        Y,
    )

    dates = pd.date_range("2022-01-01", periods=periods, freq="W")
    rows = []
    for node, resid in residuals_by_node.items():
        for ds, r in zip(dates, resid, strict=True):
            rows.append(
                {UNIQUE_ID: node, DS: ds, Y: float(r), MODEL_NAME: _U4_MODEL, FITTED_Y_HAT: 0.0}
            )
    return pd.DataFrame(rows)


def _u4_apply_frame(origin, centers_by_node):
    from calibre.core.forecast_frame import (
        DS,
        FORECAST_ORIGIN,
        MODEL_NAME,
        UNIQUE_ID,
        Y_HAT,
        H,
        Y,
    )

    nodes = list(centers_by_node)
    return pd.DataFrame(
        {
            UNIQUE_ID: nodes,
            DS: [origin + pd.Timedelta(weeks=1)] * len(nodes),
            Y: [np.nan] * len(nodes),
            Y_HAT: [centers_by_node[n] for n in nodes],
            H: [1] * len(nodes),
            FORECAST_ORIGIN: [origin] * len(nodes),
            MODEL_NAME: [_U4_MODEL] * len(nodes),
        }
    )


def test_per_node_held_out_coverage_on_reduced_lattice():
    from calibre.conformal.runtime import SymmetricIntervalRuntime
    from calibre.core.forecast_frame import UNIQUE_ID, Y, interval_column_names

    index = _coherent_hierarchy_index()  # nodes A, B + group=g + total
    config = _u4_config()
    runtime = SymmetricIntervalRuntime(config, hierarchy_index=index)
    lower_col, upper_col = interval_column_names(0.9)

    rng = np.random.default_rng(2024)
    # True OOS errors: A wide, B tight. The in-sample residuals deliberately have a
    # DIFFERENT scale, so held-out width ownership is what closes coverage.
    oos_scale = {"A": 6.0, "B": 1.5}
    in_sample_scale = {"A": 1.0, "B": 4.0}
    centers = {"A": 50.0, "B": 80.0}

    inside = {"A": 0, "B": 0}
    total = {"A": 0, "B": 0}
    origin = pd.Timestamp("2023-01-01")
    for step in range(160):
        origin = origin + pd.Timedelta(weeks=1)
        fitted = _u4_fitted(
            origin,
            {n: rng.normal(0.0, in_sample_scale[n], size=30) for n in ("A", "B")},
        )
        runtime.set_fitted_values(fitted)
        applied = runtime.apply(_u4_apply_frame(origin, {"A": centers["A"], "B": centers["B"]}))

        actuals = {n: centers[n] + rng.normal(0.0, oos_scale[n]) for n in ("A", "B")}
        resolved = applied.copy()
        resolved[Y] = resolved[UNIQUE_ID].map(actuals)
        # Score realized coverage on the warmed-up tail only (post-warmup rows).
        if step >= 80:
            for _, row in resolved.iterrows():
                node = row[UNIQUE_ID]
                lo, hi = row[lower_col], row[upper_col]
                if np.isnan(lo) or np.isnan(hi):
                    continue
                total[node] += 1
                if lo <= actuals[node] <= hi:
                    inside[node] += 1
        runtime.observe(resolved)

    # Both bottom nodes must accumulate evaluated rows and land near 90% coverage —
    # achievable only because the width is held-out-owned, not in-sample-shaped.
    assert total["A"] > 30 and total["B"] > 30
    for node in ("A", "B"):
        realized = inside[node] / total[node]
        assert 0.80 <= realized <= 0.99, (node, realized)


def test_held_out_half_widths_engage_under_default_global_partition():
    # Under the DEFAULT global partition the calibrator pools every series into one
    # radius, but the held-out dict must still key by the per-node spread keys (NOT
    # the pooled "__global__" key) so CoherentDraws' per-bottom-node lookups hit,
    # with every node mapping to the single pooled radius. Regression guard: keying
    # by the raw partition base made the per-node lookup miss and silently no-op.
    from calibre.conformal.partitions import global_partition
    from calibre.conformal.runtime import SymmetricIntervalConfig, SymmetricIntervalRuntime
    from calibre.core.forecast_frame import UNIQUE_ID, Y

    index = _coherent_hierarchy_index()
    config = SymmetricIntervalConfig(
        method="mscp",
        coverage=0.9,
        calibration_window=40,
        spread="coherent_draws",
        draw_count=200,
        draw_seed=11,
        partition_key=global_partition,
    )
    runtime = SymmetricIntervalRuntime(config, hierarchy_index=index)
    rng = np.random.default_rng(7)
    origin = pd.Timestamp("2023-01-01")
    for _ in range(30):
        origin = origin + pd.Timedelta(weeks=1)
        runtime.set_fitted_values(
            _u4_fitted(origin, {n: rng.normal(0.0, 2.0, size=30) for n in ("A", "B")})
        )
        applied = runtime.apply(_u4_apply_frame(origin, {"A": 50.0, "B": 80.0}))
        resolved = applied.copy()
        resolved[Y] = resolved[UNIQUE_ID].map(
            {"A": 50.0 + rng.normal(0.0, 5.0), "B": 80.0 + rng.normal(0.0, 5.0)}
        )
        runtime.observe(resolved)

    held = runtime._held_out_half_widths(_u4_apply_frame(origin, {"A": 50.0, "B": 80.0}), 0.1)
    assert set(held) == {f"{_U4_MODEL}:h1:A", f"{_U4_MODEL}:h1:B"}
    pooled = held[f"{_U4_MODEL}:h1:A"]
    assert held[f"{_U4_MODEL}:h1:B"] == pooled
    assert np.isfinite(pooled) and pooled > 0.0


def test_resume_reproduces_held_out_bounds_byte_for_byte():
    from calibre.conformal.runtime import SymmetricIntervalRuntime
    from calibre.core.forecast_frame import UNIQUE_ID, Y, interval_column_names

    index = _coherent_hierarchy_index()
    lower_col, upper_col = interval_column_names(0.9)
    rng_seed = 7

    def run(split_at):
        config = _u4_config()
        runtime = SymmetricIntervalRuntime(config, hierarchy_index=index)
        rng = np.random.default_rng(rng_seed)
        captured = []
        origin = pd.Timestamp("2023-01-01")
        for step in range(24):
            origin = origin + pd.Timedelta(weeks=1)
            # Persist/restore across the partitioned surface at the split boundary.
            if split_at is not None and step == split_at:
                states = runtime.get_partition_states()
                runtime = SymmetricIntervalRuntime.from_partition_states(
                    config, states, hierarchy_index=index
                )
            fitted = _u4_fitted(origin, {n: rng.normal(0.0, 2.0, size=30) for n in ("A", "B")})
            runtime.set_fitted_values(fitted)
            applied = runtime.apply(_u4_apply_frame(origin, {"A": 50.0, "B": 80.0}))
            ordered = applied.sort_values(UNIQUE_ID)
            captured.append(
                (
                    tuple(ordered[UNIQUE_ID]),
                    ordered[[lower_col, upper_col]].to_numpy(dtype=float),
                )
            )
            actuals = {n: c + rng.normal(0.0, 3.0) for n, c in (("A", 50.0), ("B", 80.0))}
            resolved = applied.copy()
            resolved[Y] = resolved[UNIQUE_ID].map(actuals)
            runtime.observe(resolved)
        return captured

    single = run(None)
    resumed = run(12)
    saw_finite = False
    for (nodes_one, bounds_one), (nodes_two, bounds_two) in zip(single, resumed, strict=True):
        assert nodes_one == nodes_two
        np.testing.assert_array_equal(bounds_one, bounds_two)  # NaN positions match too
        saw_finite = saw_finite or bool(np.isfinite(bounds_one).any())
    # Guard against a vacuous all-NaN pass: warmed-up origins must emit real bounds.
    assert saw_finite


def test_analytic_apply_observe_unchanged():
    # The default analytic path ignores SpreadContext entirely; widening the
    # context with held_out_half_width must not perturb its bounds or scores.
    from calibre.conformal.partitions import series_partition
    from calibre.conformal.runtime import SymmetricIntervalConfig, SymmetricIntervalRuntime
    from calibre.core.forecast_frame import (
        DS,
        FORECAST_ORIGIN,
        MODEL_NAME,
        NONCONFORMITY_SCORE,
        UNIQUE_ID,
        Y_HAT,
        H,
        Y,
        interval_column_names,
    )

    lower_col, upper_col = interval_column_names(0.9)

    def run():
        config = SymmetricIntervalConfig(
            method="mscp",
            coverage=0.9,
            calibration_window=10,
            partition_key=series_partition,
        )
        runtime = SymmetricIntervalRuntime(config)
        bounds = []
        scores = []
        origin = pd.Timestamp("2023-01-01")
        for step in range(12):
            origin = origin + pd.Timedelta(weeks=1)
            frame = pd.DataFrame(
                {
                    UNIQUE_ID: ["A", "B"],
                    DS: [origin + pd.Timedelta(weeks=1)] * 2,
                    Y: [np.nan, np.nan],
                    Y_HAT: [10.0, 20.0],
                    H: [1, 1],
                    FORECAST_ORIGIN: [origin, origin],
                    MODEL_NAME: ["model", "model"],
                }
            )
            applied = runtime.apply(frame)
            bounds.append(applied[[lower_col, upper_col]].to_numpy())
            resolved = applied.copy()
            resolved[Y] = [11.0 + step, 18.0 - step]
            observed = runtime.observe(resolved)
            scores.append(observed[NONCONFORMITY_SCORE].to_numpy())
        return bounds, scores, runtime.get_resume_state()

    bounds_a, scores_a, state_a = run()
    bounds_b, scores_b, state_b = run()
    for one, two in zip(bounds_a, bounds_b, strict=True):
        np.testing.assert_array_equal(one, two)
    for one, two in zip(scores_a, scores_b, strict=True):
        np.testing.assert_array_equal(one, two)
    # The analytic resume state carries no held-out-width key (nothing new persisted).
    assert "held_out_half_width" not in state_a
    assert state_a.keys() == state_b.keys()


def test_analytic_resume_state_unchanged():
    # get_resume_state / get_partition_states for an analytic runtime gain no new
    # key for held-out width — it is read live, never serialized.
    from calibre.conformal.partitions import series_partition
    from calibre.conformal.runtime import SymmetricIntervalConfig, SymmetricIntervalRuntime
    from calibre.core.forecast_frame import (
        DS,
        FORECAST_ORIGIN,
        MODEL_NAME,
        UNIQUE_ID,
        Y_HAT,
        H,
        Y,
        interval_column_names,
    )

    lower_col, upper_col = interval_column_names(0.9)
    config = SymmetricIntervalConfig(
        method="mscp", coverage=0.9, calibration_window=8, partition_key=series_partition
    )
    runtime = SymmetricIntervalRuntime(config)
    origin = pd.Timestamp("2023-01-01")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            DS: [origin + pd.Timedelta(weeks=1)] * 2,
            Y: [np.nan, np.nan],
            Y_HAT: [10.0, 20.0],
            H: [1, 1],
            FORECAST_ORIGIN: [origin, origin],
            MODEL_NAME: ["model", "model"],
        }
    )
    applied = runtime.apply(frame)
    resolved = applied.copy()
    resolved[Y] = [12.0, 17.0]
    runtime.observe(resolved)

    expected_keys = {
        "method",
        "mode",
        "coverage",
        "calibration_window",
        "gamma",
        "quantile_rule",
        "protection_period",
        "issued_count",
        "controller",
        "calibrator",
    }
    assert set(runtime.get_resume_state()) == expected_keys
    for state in runtime.get_partition_states().values():
        assert "held_out_half_width" not in state
        assert "held_out_half_width" not in state.get("calibrator", {})


# --- U5 joint/simultaneous coverage (end-to-end reduced lattice) -------------


def _u5_runtime(window: int = 80, *, joint_enabled: bool = True):
    from calibre.conformal.runtime import SymmetricIntervalRuntime

    index = _coherent_hierarchy_index()
    runtime = SymmetricIntervalRuntime(_u4_config(window), hierarchy_index=index)
    if not joint_enabled:
        # The kappa=1 baseline: never inflate, so simultaneous coverage reflects the
        # uncorrected coherent sum-of-member band (the Bonferroni-degrading regime).
        runtime._joint_inflation_map = lambda frame, alpha, held_out: None  # type: ignore[method-assign]
    return runtime


def _run_lattice(runtime, *, steps=400, warm=200, oos_scale, rho=0.5, seed=2024):
    """Drive apply->observe over origins; return per-node and simultaneous coverage.

    Cross-node-dependent OOS errors (shared common shock, coefficient ``rho``) so the
    joint correction has a realised dependence to learn. Coverage is scored on the
    warm tail only, using the lagged estimator the runtime applies.
    """
    from calibre.core.forecast_frame import UNIQUE_ID, Y, interval_column_names

    lower_col, upper_col = interval_column_names(0.9)
    rng = np.random.default_rng(seed)
    centers = {"A": 50.0, "B": 80.0}
    inside = {"A": 0, "B": 0}
    total = {"A": 0, "B": 0}
    simultaneous_inside = 0
    simultaneous_total = 0
    origin = pd.Timestamp("2023-01-01")
    for step in range(steps):
        origin = origin + pd.Timedelta(weeks=1)
        fitted = _u4_fitted(
            origin, {n: rng.normal(0.0, 2.0, size=40) for n in ("A", "B")}, periods=40
        )
        runtime.set_fitted_values(fitted)
        applied = runtime.apply(_u4_apply_frame(origin, dict(centers)))

        common = rng.normal(0.0, 1.0)
        actuals = {}
        for n in ("A", "B"):
            idio = rng.normal(0.0, 1.0)
            shock = (rho * common + np.sqrt(1.0 - rho**2) * idio) * oos_scale[n]
            actuals[n] = centers[n] + shock
        resolved = applied.copy()
        resolved[Y] = resolved[UNIQUE_ID].map(actuals)
        if step >= warm:
            covered_all = True
            scored_any = False
            for _, row in resolved.iterrows():
                node = row[UNIQUE_ID]
                lo, hi = row[lower_col], row[upper_col]
                if np.isnan(lo) or np.isnan(hi):
                    covered_all = False
                    continue
                scored_any = True
                total[node] += 1
                ok = lo <= actuals[node] <= hi
                if ok:
                    inside[node] += 1
                else:
                    covered_all = False
            if scored_any:
                simultaneous_total += 1
                if covered_all:
                    simultaneous_inside += 1
        runtime.observe(resolved)
    return inside, total, simultaneous_inside, simultaneous_total


def test_simultaneous_coverage_two_sided_on_reduced_lattice():
    # On a small cross-node-dependent hierarchy, the joint correction lifts realised
    # SIMULTANEOUS coverage (all bottom nodes covered at the same origin) toward the
    # 1-alpha target where the kappa=1 baseline under-covers (Bonferroni-degraded).
    # The lagged finite-sample estimator approaches but need not perfectly hit the
    # nominal, so the band is two-sided: meaningfully above the baseline, close to
    # the target, and bounded above (never grossly over). Per-LEVEL (bottom)
    # marginal coverage is reported too.
    oos_scale = {"A": 6.0, "B": 1.5}

    inside_j, total_j, sim_in_j, sim_tot_j = _run_lattice(_u5_runtime(), oos_scale=oos_scale)
    _, _, sim_in_b, sim_tot_b = _run_lattice(_u5_runtime(joint_enabled=False), oos_scale=oos_scale)

    assert sim_tot_j > 100 and sim_tot_b > 100
    joint_sim = sim_in_j / sim_tot_j
    baseline_sim = sim_in_b / sim_tot_b
    # The correction strictly and materially helps vs the uncorrected coherent band.
    assert joint_sim >= baseline_sim + 0.03
    # Bounded two-sided: approaches the 1-alpha target from a lagged estimator, and
    # never over-covers grossly (an unbounded kappa would blow past this).
    assert 0.85 <= joint_sim <= 0.99
    # The uncorrected baseline visibly under-covers simultaneously (the gap U5 closes).
    assert baseline_sim < 0.85
    # Per-LEVEL bottom marginal coverage stays valid after the (only-widening) joint
    # inflation.
    for node in ("A", "B"):
        assert inside_j[node] / total_j[node] >= 0.88


def test_per_node_marginal_validity_preserved_under_joint():
    # The joint inflation only widens, so each bottom node's realised marginal
    # coverage stays >= 1-alpha (here >= 0.88 on the finite warm tail).
    oos_scale = {"A": 5.0, "B": 5.0}
    inside, total, _, _ = _run_lattice(_u5_runtime(), oos_scale=oos_scale, rho=0.6)
    for node in ("A", "B"):
        assert total[node] > 30
        assert inside[node] / total[node] >= 0.88


def test_standardisation_balances_scale_per_series():
    # Under per-series partitioning a large-scale node must not dominate kappa: with
    # each node carrying its own sigma_i, the cross-node max of r_i/sigma_i is scale
    # free. Companion: under global partitioning the standardisation collapses
    # (every node shares the pooled sigma), the documented degeneracy.
    from calibre.conformal.partitions import global_partition, series_partition
    from calibre.conformal.runtime import (
        SymmetricIntervalConfig,
        SymmetricIntervalRuntime,
        joint_key,
    )
    from calibre.core.forecast_frame import UNIQUE_ID, Y

    index = _coherent_hierarchy_index()

    def warm_and_kappa(partition_key):
        config = SymmetricIntervalConfig(
            method="mscp",
            coverage=0.9,
            calibration_window=40,
            spread="coherent_draws",
            draw_count=200,
            draw_seed=11,
            partition_key=partition_key,
        )
        runtime = SymmetricIntervalRuntime(config, hierarchy_index=index)
        rng = np.random.default_rng(5)
        origin = pd.Timestamp("2023-01-01")
        # A is ~50x B's scale: a raw-residual max would be dominated by A.
        scale = {"A": 50.0, "B": 1.0}
        for _ in range(60):
            origin = origin + pd.Timedelta(weeks=1)
            runtime.set_fitted_values(
                _u4_fitted(origin, {n: rng.normal(0.0, scale[n], size=30) for n in ("A", "B")})
            )
            applied = runtime.apply(_u4_apply_frame(origin, {"A": 50.0, "B": 80.0}))
            resolved = applied.copy()
            resolved[Y] = resolved[UNIQUE_ID].map(
                {n: c + rng.normal(0.0, scale[n]) for n, c in (("A", 50.0), ("B", 80.0))}
            )
            runtime.observe(resolved)
        frame = _u4_apply_frame(origin, {"A": 50.0, "B": 80.0})
        held = runtime._held_out_half_widths(frame, 0.1)
        sigma_by_node = {"A": held[f"{_U4_MODEL}:h1:A"], "B": held[f"{_U4_MODEL}:h1:B"]}
        jkey = joint_key(_U4_MODEL, "perhorizon", 1)
        # Inspect each node's contribution to the per-origin max under this sigma.
        a_dominates = 0
        b_dominates = 0
        for labels, scores in runtime._joint_history[jkey]:
            std = {
                str(lbl): float(s) / sigma_by_node[str(lbl)]
                for lbl, s in zip(labels, scores, strict=True)
            }
            if std["A"] >= std["B"]:
                a_dominates += 1
            else:
                b_dominates += 1
        return a_dominates, b_dominates

    a_series, b_series = warm_and_kappa(series_partition)
    # Per-series standardisation balances scale: B drives the max a real fraction of
    # the time, not ~never as it would under a raw-residual max.
    assert b_series > 0.20 * (a_series + b_series)

    a_global, b_global = warm_and_kappa(global_partition)
    # Global degeneracy: one pooled sigma, so the max collapses to the raw-residual
    # max and the large-scale node A dominates nearly always.
    assert a_global > b_global


def test_thin_history_and_missing_sigma_fall_back_kappa_one():
    # A too-thin joint window yields kappa = 1.0 (no widening); a node whose sigma_i
    # is +inf/missing is excluded from the max and never produces NaN/inf.
    from calibre.conformal.runtime import joint_key

    runtime = _u5_runtime(window=40)
    jkey = joint_key(_U4_MODEL, "perhorizon", 1)

    # Empty store -> cold-start kappa of exactly 1.0.
    assert runtime._joint_kappa(jkey, 0.1, {"A": 2.0, "B": 3.0}) == 1.0

    # Seed one origin's raw scores, then a missing/inf sigma must drop that node.
    import numpy as _np

    runtime._append_joint_scores(
        jkey, _np.asarray(["A", "B"], dtype=object), _np.asarray([4.0, 9.0], dtype=float)
    )
    # B excluded (inf sigma) -> kappa from A alone = max(4/2, 1) = 2.0, finite.
    kappa = runtime._joint_kappa(jkey, 0.1, {"A": 2.0, "B": _np.inf})
    assert _np.isfinite(kappa) and kappa == pytest.approx(2.0)
    # Both sigmas missing -> no node survives -> kappa falls back to 1.0.
    assert runtime._joint_kappa(jkey, 0.1, {}) == 1.0


# --- U5 node-vector store + persistence/resume -------------------------------


def _warm_coherent_runtime(steps=18, *, split_at=None, window=40):
    """Warm a coherent per-series runtime over origins, optionally persist/restore.

    Returns the (possibly resumed) runtime plus the captured per-origin bounds so
    callers can assert byte-identity across the persisted boundary.
    """
    from calibre.conformal.runtime import SymmetricIntervalRuntime
    from calibre.core.forecast_frame import UNIQUE_ID, Y, interval_column_names

    index = _coherent_hierarchy_index()
    config = _u4_config(window)
    lower_col, upper_col = interval_column_names(0.9)
    runtime = SymmetricIntervalRuntime(config, hierarchy_index=index)
    rng = np.random.default_rng(7)
    origin = pd.Timestamp("2023-01-01")
    captured = []
    for step in range(steps):
        origin = origin + pd.Timedelta(weeks=1)
        if split_at is not None and step == split_at:
            states = runtime.get_partition_states()
            runtime = SymmetricIntervalRuntime.from_partition_states(
                config, states, hierarchy_index=index
            )
        runtime.set_fitted_values(
            _u4_fitted(origin, {n: rng.normal(0.0, 2.0, size=30) for n in ("A", "B")})
        )
        applied = runtime.apply(_u4_apply_frame(origin, {"A": 50.0, "B": 80.0}))
        ordered = applied.sort_values(UNIQUE_ID)
        captured.append(ordered[[lower_col, upper_col]].to_numpy(dtype=float))
        resolved = applied.copy()
        resolved[Y] = resolved[UNIQUE_ID].map(
            {n: c + rng.normal(0.0, 4.0) for n, c in (("A", 50.0), ("B", 80.0))}
        )
        runtime.observe(resolved)
    return runtime, captured


def test_joint_store_round_trips_with_byte_identical_kappa_and_bounds():
    # Run N origins single-pass; run the same with persist/restore across the
    # partitioned surface mid-stream. The joint store rehydrates and the emitted
    # coherent bounds are byte-identical across the boundary.
    _, single = _warm_coherent_runtime(steps=18, split_at=None)
    _, resumed = _warm_coherent_runtime(steps=18, split_at=10)
    saw_finite = False
    for one, two in zip(single, resumed, strict=True):
        np.testing.assert_array_equal(one, two)
        saw_finite = saw_finite or bool(np.isfinite(one).any())
    assert saw_finite


def test_joint_store_is_ragged_no_nan_pad():
    # With varying present-node sets the store stays ragged (present nodes only), so
    # the serialized state is strict-JSON-parseable (no NaN/Infinity tokens).
    import json

    from calibre.conformal.runtime import SymmetricIntervalRuntime, joint_key
    from calibre.core.forecast_frame import UNIQUE_ID, Y

    index = _coherent_hierarchy_index()
    config = _u4_config()
    runtime = SymmetricIntervalRuntime(config, hierarchy_index=index)
    rng = np.random.default_rng(3)
    origin = pd.Timestamp("2023-01-01")
    for step in range(12):
        origin = origin + pd.Timedelta(weeks=1)
        present = ["A", "B"] if step % 2 == 0 else ["A"]  # B absent on odd origins
        runtime.set_fitted_values(
            _u4_fitted(origin, {n: rng.normal(0.0, 2.0, size=30) for n in present})
        )
        applied = runtime.apply(_u4_apply_frame(origin, {n: 50.0 for n in present}))
        resolved = applied.copy()
        resolved[Y] = resolved[UNIQUE_ID].map({n: 50.0 + rng.normal(0.0, 4.0) for n in present})
        runtime.observe(resolved)

    jkey = joint_key(_U4_MODEL, "perhorizon", 1)
    lengths = {len(labels) for labels, _ in runtime._joint_history[jkey]}
    assert lengths == {1, 2}  # ragged: present-node-only vectors, no NaN pad
    serialized = runtime._serialized_joint_history()
    text = json.dumps(serialized)  # default allow_nan=True would still emit tokens
    # A strict parse rejects NaN/Infinity tokens; the ragged invariant guarantees none.
    json.loads(text, parse_constant=_reject_constant)


def _reject_constant(token):
    raise ValueError(f"non-finite JSON token: {token}")


def test_resume_leaves_marginal_keyset_byte_identical():
    # After get_partition_states -> from_partition_states, the rehydrated marginal
    # score_history keyset carries NO ::joint:: key (no phantom marginal partition).
    runtime, _ = _warm_coherent_runtime(steps=14, split_at=None)
    states = runtime.get_partition_states()
    from calibre.conformal.runtime import (
        JOINT_KEY_INFIX,
        SymmetricIntervalRuntime,
    )

    index = _coherent_hierarchy_index()
    rebuilt = SymmetricIntervalRuntime.from_partition_states(
        _u4_config(), states, hierarchy_index=index
    )
    marginal_keys = rebuilt.calibrator.get_state()["score_history"].keys()
    assert all(JOINT_KEY_INFIX not in key for key in marginal_keys)
    assert marginal_keys  # non-empty: the marginal store survived the round-trip


def test_unique_id_lookalike_does_not_collide_with_joint_key():
    # A unique_id literally named to mimic a joint key cannot collide: marginal keys
    # are single-colon, joint keys carry the reserved double-colon infix.
    from calibre.conformal.runtime import JOINT_KEY_INFIX, joint_key

    lookalike = "model::joint::h1"  # a data-controlled node id forging the infix
    marginal_partition = f"model:h1:{lookalike}"  # how it enters the marginal keyspace
    jkey = joint_key("model", "perhorizon", 1)
    assert jkey == "model::joint::h1"
    # The marginal partition key never equals a joint key, and the joint infix never
    # appears as the WHOLE marginal key (only nested inside the node-id segment).
    assert marginal_partition != jkey
    assert not marginal_partition.startswith(f"model{JOINT_KEY_INFIX}")


def test_old_rows_without_joint_key_rehydrate_empty():
    # A persisted state lacking the joint_history key rehydrates to an empty joint
    # store (kappa = 1 until refill) — backward compatibility with pre-U5 rows.
    from calibre.conformal.runtime import (
        JOINT_HISTORY_KEY,
        SymmetricIntervalRuntime,
        joint_key,
    )

    runtime, _ = _warm_coherent_runtime(steps=12, split_at=None)
    states = runtime.get_partition_states()
    stripped = {
        key: {k: v for k, v in state.items() if k != JOINT_HISTORY_KEY}
        for key, state in states.items()
    }
    index = _coherent_hierarchy_index()
    rebuilt = SymmetricIntervalRuntime.from_partition_states(
        _u4_config(), stripped, hierarchy_index=index
    )
    assert rebuilt._joint_history == {}
    jkey = joint_key(_U4_MODEL, "perhorizon", 1)
    assert rebuilt._joint_kappa(jkey, 0.1, {"A": 2.0, "B": 3.0}) == 1.0


def test_joint_store_respects_window_maxlen():
    # The joint deque is bounded by calibration_window: more origins than the window
    # never grows the store past maxlen.
    from calibre.conformal.runtime import joint_key

    window = 6
    runtime, _ = _warm_coherent_runtime(steps=20, window=window)
    jkey = joint_key(_U4_MODEL, "perhorizon", 1)
    assert len(runtime._joint_history[jkey]) == window


def test_marginal_observe_stream_byte_identical():
    # A coherent apply/observe round-trip produces identical marginal score_history
    # and NONCONFORMITY_SCORE to a kappa-disabled baseline: the joint pass is purely
    # additive and never perturbs the marginal stream.
    from calibre.conformal.runtime import SymmetricIntervalRuntime
    from calibre.core.forecast_frame import NONCONFORMITY_SCORE, UNIQUE_ID, Y

    index = _coherent_hierarchy_index()

    def run(joint_enabled):
        config = _u4_config()
        runtime = SymmetricIntervalRuntime(config, hierarchy_index=index)
        if not joint_enabled:
            runtime._joint_inflation_map = lambda frame, alpha, held_out: None  # type: ignore[method-assign]
        rng = np.random.default_rng(7)
        origin = pd.Timestamp("2023-01-01")
        scores = []
        for _ in range(16):
            origin = origin + pd.Timedelta(weeks=1)
            runtime.set_fitted_values(
                _u4_fitted(origin, {n: rng.normal(0.0, 2.0, size=30) for n in ("A", "B")})
            )
            applied = runtime.apply(_u4_apply_frame(origin, {"A": 50.0, "B": 80.0}))
            resolved = applied.copy()
            resolved[Y] = resolved[UNIQUE_ID].map(
                {n: c + rng.normal(0.0, 4.0) for n, c in (("A", 50.0), ("B", 80.0))}
            )
            observed = runtime.observe(resolved)
            scores.append(observed[NONCONFORMITY_SCORE].to_numpy(dtype=float))
        return scores, runtime.calibrator.get_state()["score_history"]

    scores_joint, hist_joint = run(True)
    scores_base, hist_base = run(False)
    for one, two in zip(scores_joint, scores_base, strict=True):
        np.testing.assert_array_equal(one, two)
    assert set(hist_joint) == set(hist_base)
    for key in hist_joint:
        np.testing.assert_array_equal(hist_joint[key], hist_base[key])


def test_analytic_runtime_never_allocates_joint_store():
    # The analytic path's joint store is None and stays None through apply/observe,
    # and its partition states carry no joint_history key (byte-identical default).
    from calibre.conformal.partitions import series_partition
    from calibre.conformal.runtime import (
        JOINT_HISTORY_KEY,
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

    config = SymmetricIntervalConfig(
        method="mscp", coverage=0.9, calibration_window=8, partition_key=series_partition
    )
    runtime = SymmetricIntervalRuntime(config)
    assert runtime._joint_history is None
    origin = pd.Timestamp("2023-01-01")
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            DS: [origin + pd.Timedelta(weeks=1)] * 2,
            Y: [np.nan, np.nan],
            Y_HAT: [10.0, 20.0],
            H: [1, 1],
            FORECAST_ORIGIN: [origin, origin],
            MODEL_NAME: ["model", "model"],
        }
    )
    applied = runtime.apply(frame)
    resolved = applied.copy()
    resolved[Y] = [12.0, 17.0]
    runtime.observe(resolved)
    assert runtime._joint_history is None
    for state in runtime.get_partition_states().values():
        assert JOINT_HISTORY_KEY not in state
