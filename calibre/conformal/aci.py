from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from calibre.conformal.intervals import symmetric_interval, symmetric_intervals
from calibre.conformal.policies import OnlineConformalController
from calibre.conformal.protocols import Score
from calibre.conformal.scores import absolute_error_score
from calibre.conformal.types import IntervalPrediction, MultiStepIntervalPrediction


def _as_scalar_score(score) -> float:
    arr = np.asarray(score, dtype=float).reshape(-1)
    if arr.size != 1:
        raise ValueError("Expected Score to return a scalar score")
    return float(arr[0])


def _as_1d_array(values, name: str, length: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        if length is None:
            return arr.reshape(1)
        return np.full(length, float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D array")
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}")
    return arr.astype(float, copy=True)


def _validate_bounds(bounds) -> tuple[float, float] | None:
    if bounds is None:
        return None
    lower, upper = bounds
    if lower > upper:
        raise ValueError("alpha_bounds must satisfy lower <= upper")
    return float(lower), float(upper)


def _clip_alpha(alpha, bounds) -> np.ndarray | float:
    if bounds is None:
        arr = np.asarray(alpha, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        return arr.astype(float, copy=True)
    lower, upper = bounds
    clipped = np.clip(alpha, lower, upper)
    if np.ndim(clipped) == 0:
        return float(clipped)
    return clipped


def _validate_quantile_rule(quantile_rule: str) -> Literal["conformal", "higher"]:
    if quantile_rule not in {"conformal", "higher"}:
        raise ValueError("quantile_rule must be 'conformal' or 'higher'")
    return quantile_rule  # type: ignore[return-value]


def _finite_sample_radius(
    scores: Iterable[float],
    alpha: float,
    default_radius: float,
    quantile_rule: Literal["conformal", "higher"] = "conformal",
) -> float:
    """Compute the (1-alpha) quantile of *scores* under the chosen rule.

    **Coverage semantics differ between the two rules:**

    - ``"higher"`` (standard split-conformal): returns ``np.inf`` when
      ``alpha <= 1/(n+1)``, i.e. the calibration set is too small to provide
      a finite radius at the requested coverage level.  This gives exact
      finite-sample coverage guarantees.

    - ``"conformal"`` (clipped rank formula): clamps the rank to ``[0, n-1]``
      and always returns a finite value (the maximum observed score).  Use
      this rule when ``np.inf`` radii are unacceptable (e.g. for plotting or
      downstream arithmetic), but note that coverage guarantees no longer
      hold for very small calibration sets with small ``alpha``.
    """
    scores_arr = np.asarray(list(scores), dtype=float)
    if scores_arr.size == 0:
        return float(default_radius)
    ordered = np.sort(scores_arr)
    quantile_rule = _validate_quantile_rule(quantile_rule)
    alpha = float(np.asarray(alpha, dtype=float))

    if quantile_rule == "higher":
        if alpha <= 1.0 / (ordered.size + 1):
            return float(np.inf)
        clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(np.quantile(ordered, 1.0 - clipped_alpha, method="higher"))

    clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
    rank = int(np.ceil((ordered.size + 1) * (1.0 - clipped_alpha))) - 1
    rank = min(max(rank, 0), ordered.size - 1)
    return float(ordered[rank])


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
        self._alpha = _clip_alpha(  # type: ignore[assignment]
            self._alpha + self._gamma * (self._target_alpha - error), self._bounds
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
