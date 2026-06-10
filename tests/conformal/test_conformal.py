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
    """
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
    """
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
