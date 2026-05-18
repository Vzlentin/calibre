from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from calibre.conformal.intervals import symmetric_interval, symmetric_intervals
from calibre.conformal.numerics import (
    _as_1d_array,
    _as_scalar_score,
    _clip_alpha,
    _finite_sample_radius,
    _validate_bounds,
    _validate_quantile_rule,
)
from calibre.conformal.policies import OnlineConformalController
from calibre.conformal.protocols import Score
from calibre.conformal.scores import absolute_error_score
from calibre.conformal.types import IntervalPrediction, MultiStepIntervalPrediction


class AdaptiveConformalInference(OnlineConformalController):
    """One-step adaptive conformal inference controller."""

    def __init__(
        self,
        alpha: float,
        gamma: float,
        initial_alpha: float | None = None,
        score: Score = absolute_error_score,
        initial_scores: Iterable[float] | None = None,
        initial_radius: float = 0.0,
        alpha_bounds: tuple[float, float] | None = (1e-6, 1.0 - 1e-6),
        quantile_rule: Literal["conformal", "higher"] = "conformal",
    ):
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self._bounds = _validate_bounds(alpha_bounds)
        self._quantile_rule = _validate_quantile_rule(quantile_rule)
        self._target_alpha = float(alpha)
        self._gamma = float(gamma)
        self._alpha: float = float(
            _clip_alpha(
                self._target_alpha if initial_alpha is None else float(initial_alpha),
                self._bounds,
            )
        )
        self._score = score
        self._initial_radius = float(initial_radius)
        self._score_history = [float(s) for s in ([] if initial_scores is None else initial_scores)]
        self._error_history: list[int] = []
        self._alpha_history = [float(self._alpha)]
        self._radius_history: list[float] = []

    @property
    def current_alpha(self) -> float:
        return float(self._alpha)

    @property
    def score_history(self) -> np.ndarray:
        return np.asarray(self._score_history, dtype=float)

    @property
    def error_history(self) -> np.ndarray:
        return np.asarray(self._error_history, dtype=int)

    def trim_scores(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        if len(self._score_history) > window:
            self._score_history = self._score_history[-window:]

    def get_radius(self, alpha: float | None = None) -> float:
        alpha = self._alpha if alpha is None else float(alpha)
        return _finite_sample_radius(
            self._score_history,
            alpha,
            self._initial_radius,
            quantile_rule=self._quantile_rule,
        )

    def predict_interval(self, point_forecast) -> IntervalPrediction:
        radius = self.get_radius()
        self._radius_history.append(radius)
        return symmetric_interval(
            center=float(point_forecast),
            radius=radius,
            alpha=self._alpha,
            issued_at=len(self._error_history),
        )

    def update(self, error: int) -> float:
        self._alpha = float(
            _clip_alpha(
                self._alpha + self._gamma * (self._target_alpha - int(error)),
                self._bounds,
            )
        )
        self._alpha_history.append(float(self._alpha))
        return float(self._alpha)

    def observe(self, y_true: float, prediction: IntervalPrediction) -> dict:
        error = int(not prediction.contains(float(y_true)))
        score = _as_scalar_score(self._score(y_true, prediction.center))
        self._score_history.append(score)
        self._error_history.append(error)
        alpha_before = float(self._alpha)
        alpha_after = self.update(error)
        return {
            "error": error,
            "score": score,
            "alpha_before": alpha_before,
            "alpha_after": alpha_after,
        }

    def get_diagnostics(self) -> dict:
        return {
            "target_alpha": self._target_alpha,
            "gamma": self._gamma,
            "current_alpha": float(self._alpha),
            "alpha_bounds": self._bounds,
            "quantile_rule": self._quantile_rule,
            "alpha_history": np.asarray(self._alpha_history, dtype=float),
            "error_history": self.error_history,
            "score_history": self.score_history,
            "radius_history": np.asarray(self._radius_history, dtype=float),
        }


class MultiStepAdaptiveConformalInference(OnlineConformalController):
    """Multi-step ACI with delayed feedback and horizon-wise control."""

    def __init__(
        self,
        horizon: int,
        alpha,
        gamma,
        initial_alpha=None,
        score: Score = absolute_error_score,
        initial_scores: Iterable[Iterable[float]] | None = None,
        initial_radius=0.0,
        alpha_bounds: tuple[float, float] | None = (1e-6, 1.0 - 1e-6),
        quantile_rule: Literal["conformal", "higher"] = "conformal",
    ):
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        self._horizon = int(horizon)
        self._bounds = _validate_bounds(alpha_bounds)
        self._quantile_rule = _validate_quantile_rule(quantile_rule)
        self._target_alpha: np.ndarray = _clip_alpha(
            _as_1d_array(alpha, "alpha", self._horizon), self._bounds
        )  # type: ignore[assignment]
        self._gamma: np.ndarray = _as_1d_array(gamma, "gamma", self._horizon)
        if np.any(self._gamma < 0):
            raise ValueError("gamma must be non-negative")
        self._alpha: np.ndarray = _clip_alpha(
            self._target_alpha
            if initial_alpha is None
            else _as_1d_array(initial_alpha, "initial_alpha", self._horizon),
            self._bounds,
        )  # type: ignore[assignment]
        self._score = score
        self._initial_radius = _as_1d_array(initial_radius, "initial_radius", self._horizon)
        self._score_history = self._normalize_score_histories(initial_scores)
        self._error_history: list[np.ndarray] = []
        self._alpha_history: list[np.ndarray] = [self._alpha.copy()]  # type: ignore[union-attr]
        self._radius_history: list[np.ndarray] = []
        self._pending_predictions: dict[int, MultiStepIntervalPrediction] = {}
        self._issued_count = 0
        self._observed_count = 0

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def current_alpha(self) -> np.ndarray:
        return self._alpha.copy()

    def _normalize_score_histories(self, initial_scores) -> list[list[float]]:
        if initial_scores is None:
            return [[] for _ in range(self._horizon)]
        histories = list(initial_scores)
        if len(histories) != self._horizon:
            raise ValueError("initial_scores must provide one iterable per horizon step")
        return [[float(score) for score in scores] for scores in histories]

    def get_radius(self, alpha=None) -> np.ndarray:
        alpha = self._alpha if alpha is None else _as_1d_array(alpha, "alpha", self._horizon)
        return np.asarray(
            [
                _finite_sample_radius(
                    self._score_history[idx],
                    alpha[idx],
                    self._initial_radius[idx],
                    quantile_rule=self._quantile_rule,
                )
                for idx in range(self._horizon)
            ],
            dtype=float,
        )

    def predict_interval(self, point_forecast) -> MultiStepIntervalPrediction:
        center = _as_1d_array(point_forecast, "point_forecast", self._horizon)
        radius = self.get_radius()
        self._radius_history.append(radius.copy())
        prediction = symmetric_intervals(
            center=center,
            radius=radius,
            alpha=self._alpha,
            issued_at=self._issued_count,
        )
        self._pending_predictions[self._issued_count] = prediction
        self._issued_count += 1
        return prediction

    def update(self, error) -> np.ndarray:
        error = _as_1d_array(error, "error", self._horizon)
        self._alpha = np.asarray(
            _clip_alpha(self._alpha + self._gamma * (self._target_alpha - error), self._bounds),
            dtype=float,
        )
        self._alpha_history.append(self._alpha.copy())
        return self._alpha.copy()

    def observe(self, y_true: float) -> dict:
        self._observed_count += 1
        actual_index = self._observed_count
        error_vector = self._target_alpha.copy()
        scores = np.full(self._horizon, np.nan, dtype=float)
        observed_mask = np.zeros(self._horizon, dtype=bool)

        for horizon_idx in range(1, self._horizon + 1):
            issued_at = actual_index - horizon_idx
            if issued_at < 0:
                continue
            prediction = self._pending_predictions.get(issued_at)
            if prediction is None:
                continue

            point_forecast = prediction.center[horizon_idx - 1]
            score = _as_scalar_score(self._score(y_true, point_forecast))
            error = int(
                not (
                    prediction.lower[horizon_idx - 1] <= y_true <= prediction.upper[horizon_idx - 1]
                )
            )

            self._score_history[horizon_idx - 1].append(score)
            error_vector[horizon_idx - 1] = error
            scores[horizon_idx - 1] = score
            observed_mask[horizon_idx - 1] = True

        expired_origin = actual_index - self._horizon
        if expired_origin in self._pending_predictions:
            del self._pending_predictions[expired_origin]

        alpha_before = self._alpha.copy()
        alpha_after = self.update(error_vector)
        history_vector = np.where(observed_mask, error_vector, np.nan)
        self._error_history.append(history_vector)
        return {
            "error_vector": error_vector.copy(),
            "scores": scores,
            "observed_mask": observed_mask,
            "alpha_before": alpha_before,
            "alpha_after": alpha_after,
            "observed_index": actual_index,
        }

    def get_diagnostics(self) -> dict:
        error_history = (
            np.vstack(self._error_history)
            if self._error_history
            else np.empty((0, self._horizon), dtype=float)
        )
        return {
            "target_alpha": self._target_alpha.copy(),
            "gamma": self._gamma.copy(),
            "current_alpha": self._alpha.copy(),
            "alpha_bounds": self._bounds,
            "quantile_rule": self._quantile_rule,
            "alpha_history": np.vstack(self._alpha_history),
            "error_history": error_history,
            "score_history": [np.asarray(scores, dtype=float) for scores in self._score_history],
            "radius_history": np.vstack(self._radius_history)
            if self._radius_history
            else np.empty((0, self._horizon)),
            "pending_predictions": len(self._pending_predictions),
        }
