"""Cumulative split conformal inference over protection-period demand."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Literal

import numpy as np

from calibre.conformal.intervals import symmetric_intervals
from calibre.conformal.numerics import (
    as_1d_array,
    as_scalar_score,
    finite_sample_radius,
    validate_quantile_rule,
)
from calibre.conformal.protocols import Score
from calibre.conformal.scores import absolute_error_score
from calibre.conformal.types import MultiStepIntervalPrediction


class CumulativeSplitConformalInference:
    """Online split conformal prediction on the cumulative protection-period demand.

    The score is ``|sum(y_actual[1..K]) - sum(y_pred[1..K])|`` where K is the
    fixed ``protection_period``. One rolling buffer of these aggregated
    nonconformity scores yields a finite-sample valid interval directly on
    ``sum(y[1..K])`` (Vovk et al., *Algorithmic Learning in a Random World*,
    2005).

    Output layout: ``predict_interval`` emits a ``MultiStepIntervalPrediction``
    of length ``protection_period``; the cumulative bound is placed at index
    ``K-1`` (centred on the cumulative point forecast). All other positions
    carry NaN bounds. ``ready_mask()`` flags only the K-th position as
    emittable, and only once the calibration buffer is non-empty.

    Cross-horizon error autocorrelation (the structure modelled explicitly by
    AcMCP) is absorbed implicitly here through the empirical distribution of
    cumulative scores.
    """

    def __init__(
        self,
        protection_period: int,
        alpha: float,
        calibration_window: int,
        score: Score = absolute_error_score,
        initial_scores: Iterable[float] | None = None,
        initial_radius: float = 0.0,
        quantile_rule: Literal["conformal", "higher"] = "higher",
    ) -> None:
        if protection_period < 1:
            raise ValueError("protection_period must be at least 1")
        if calibration_window < 1:
            raise ValueError("calibration_window must be at least 1")
        self._protection_period = int(protection_period)
        alpha_arr = np.asarray(alpha, dtype=float)
        if alpha_arr.ndim != 0:
            raise ValueError(f"alpha must be a scalar, got shape {alpha_arr.shape}")
        self._alpha = float(alpha_arr)
        self._calibration_window = int(calibration_window)
        self._score = score
        self._initial_radius = float(initial_radius)
        self._quantile_rule = validate_quantile_rule(quantile_rule)
        self._score_history: deque[float] = deque(
            (float(s) for s in (initial_scores or ())),
            maxlen=self._calibration_window,
        )
        self._radius_history: list[float] = []
        self._issued_count = 0

    @property
    def protection_period(self) -> int:
        return self._protection_period

    @property
    def horizon(self) -> int:
        return self._protection_period

    @property
    def current_alpha(self) -> np.ndarray:
        return np.full(self._protection_period, self._alpha, dtype=float)

    @property
    def calibration_window(self) -> int:
        return self._calibration_window

    def get_radius(self, alpha: float | None = None) -> float:
        alpha_value = self._alpha if alpha is None else float(np.asarray(alpha, dtype=float))
        return finite_sample_radius(
            list(self._score_history),
            alpha_value,
            self._initial_radius,
            quantile_rule=self._quantile_rule,
        )

    def ready_mask(self) -> np.ndarray:
        radius = self.get_radius()
        ready = len(self._score_history) > 0 and np.isfinite(radius)
        mask = np.zeros(self._protection_period, dtype=bool)
        mask[-1] = ready
        return mask

    def predict_interval(self, point_forecast) -> MultiStepIntervalPrediction:
        center = as_1d_array(point_forecast, "point_forecast", self._protection_period)
        radius = self.get_radius()
        self._radius_history.append(radius)

        center_vector = np.full(self._protection_period, np.nan, dtype=float)
        radius_vector = np.full(self._protection_period, np.nan, dtype=float)
        center_vector[-1] = float(center.sum())
        radius_vector[-1] = radius

        prediction = symmetric_intervals(
            center=center_vector,
            radius=radius_vector,
            alpha=self.current_alpha,
            issued_at=self._issued_count,
        )
        self._issued_count += 1
        return prediction

    def observe(
        self,
        y_actual_window: Iterable[float],
        y_hat_window: Iterable[float],
    ) -> dict[str, float | int]:
        actual_arr = np.asarray(list(y_actual_window), dtype=float)
        hat_arr = np.asarray(list(y_hat_window), dtype=float)
        if actual_arr.size != self._protection_period or hat_arr.size != self._protection_period:
            raise ValueError(
                f"Cumulative observe expects windows of length {self._protection_period}; "
                f"got y_actual={actual_arr.size}, y_hat={hat_arr.size}"
            )
        score = as_scalar_score(self._score(actual_arr.sum(), hat_arr.sum()))
        self._score_history.append(score)
        return {
            "score": score,
            "n_scores": len(self._score_history),
        }

    def get_diagnostics(self) -> dict:
        return {
            "target_alpha": float(self._alpha),
            "current_alpha": self.current_alpha,
            "calibration_window": self._calibration_window,
            "quantile_rule": self._quantile_rule,
            "protection_period": self._protection_period,
            "score_history": np.asarray(list(self._score_history), dtype=float),
            "radius_history": np.asarray(self._radius_history, dtype=float),
            "issued_count": self._issued_count,
        }


class MultiStepSplitConformalInference:
    """Online multi-step split conformal prediction with rolling score buffers.

    This follows the paper's MSCP structure at the controller level:
    horizon-specific nonconformity scores, a rolling calibration window, and
    split-conformal quantile semantics via the ``"higher"`` rule.
    """

    def __init__(
        self,
        horizon: int,
        alpha: float,
        calibration_window: int,
        score: Score = absolute_error_score,
        initial_scores: Iterable[Iterable[float]] | None = None,
        initial_radius=0.0,
        quantile_rule: Literal["conformal", "higher"] = "higher",
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if calibration_window < 1:
            raise ValueError("calibration_window must be at least 1")
        self._horizon = int(horizon)
        self._alpha = float(np.asarray(alpha, dtype=float))
        self._calibration_window = int(calibration_window)
        self._score = score
        self._initial_radius = as_1d_array(initial_radius, "initial_radius", self._horizon)
        self._quantile_rule = validate_quantile_rule(quantile_rule)
        self._score_history = self._normalize_score_histories(initial_scores)
        self._radius_history: list[np.ndarray] = []
        self._issued_count = 0

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def current_alpha(self) -> np.ndarray:
        return np.full(self._horizon, self._alpha, dtype=float)

    @property
    def calibration_window(self) -> int:
        return self._calibration_window

    def _normalize_score_histories(self, initial_scores) -> list[deque[float]]:
        if initial_scores is None:
            return [deque(maxlen=self._calibration_window) for _ in range(self._horizon)]

        histories = list(initial_scores)
        if len(histories) != self._horizon:
            raise ValueError("initial_scores must provide one iterable per horizon step")

        normalized: list[deque[float]] = []
        for scores in histories:
            history = deque((float(score) for score in scores), maxlen=self._calibration_window)
            normalized.append(history)
        return normalized

    def get_radius(self, alpha=None) -> np.ndarray:
        alpha_value = self._alpha if alpha is None else float(np.asarray(alpha, dtype=float))
        return np.asarray(
            [
                finite_sample_radius(
                    list(self._score_history[idx]),
                    alpha_value,
                    self._initial_radius[idx],
                    quantile_rule=self._quantile_rule,
                )
                for idx in range(self._horizon)
            ],
            dtype=float,
        )

    def ready_mask(self) -> np.ndarray:
        radii = self.get_radius()
        return np.asarray(
            [
                len(self._score_history[idx]) > 0 and np.isfinite(radii[idx])
                for idx in range(self._horizon)
            ],
            dtype=bool,
        )

    def predict_interval(self, point_forecast) -> MultiStepIntervalPrediction:
        center = as_1d_array(point_forecast, "point_forecast", self._horizon)
        radius = self.get_radius()
        self._radius_history.append(radius.copy())
        prediction = symmetric_intervals(
            center=center,
            radius=radius,
            alpha=self.current_alpha,
            issued_at=self._issued_count,
        )
        self._issued_count += 1
        return prediction

    def observe(self, horizon: int, y_true: float, point_forecast: float) -> dict[str, float | int]:
        horizon_idx = int(horizon) - 1
        if not 0 <= horizon_idx < self._horizon:
            raise ValueError(f"horizon must be in [1, {self._horizon}]")

        score = as_scalar_score(self._score(y_true, point_forecast))
        self._score_history[horizon_idx].append(score)
        return {
            "horizon": int(horizon),
            "score": score,
            "n_scores": len(self._score_history[horizon_idx]),
        }

    def get_diagnostics(self) -> dict:
        return {
            "target_alpha": float(self._alpha),
            "current_alpha": self.current_alpha,
            "calibration_window": self._calibration_window,
            "quantile_rule": self._quantile_rule,
            "score_history": [
                np.asarray(list(scores), dtype=float) for scores in self._score_history
            ],
            "radius_history": (
                np.vstack(self._radius_history)
                if self._radius_history
                else np.empty((0, self._horizon), dtype=float)
            ),
            "issued_count": self._issued_count,
        }
