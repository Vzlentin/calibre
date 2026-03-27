from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Literal

import numpy as np

from calibre.conformal.aci import (
    _as_1d_array,
    _as_scalar_score,
    _finite_sample_radius,
    _validate_quantile_rule,
)
from calibre.conformal.intervals import symmetric_intervals
from calibre.conformal.scores import absolute_error
from calibre.conformal.types import MultiStepIntervalPrediction


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
        score_fn: Callable = absolute_error,
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
        self._score_fn = score_fn
        self._initial_radius = _as_1d_array(initial_radius, "initial_radius", self._horizon)
        self._quantile_rule = _validate_quantile_rule(quantile_rule)
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
                _finite_sample_radius(
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
        center = _as_1d_array(point_forecast, "point_forecast", self._horizon)
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

        score = _as_scalar_score(self._score_fn(y_true, point_forecast))
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
            "score_history": [np.asarray(list(scores), dtype=float) for scores in self._score_history],
            "radius_history": (
                np.vstack(self._radius_history)
                if self._radius_history
                else np.empty((0, self._horizon), dtype=float)
            ),
            "issued_count": self._issued_count,
        }
