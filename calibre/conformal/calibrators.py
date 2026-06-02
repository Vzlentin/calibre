from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Literal

import numpy as np

from calibre.conformal.numerics import finite_sample_radius, validate_quantile_rule


class RollingQuantileCalibrator:
    def __init__(
        self,
        *,
        calibration_window: int,
        initial_scores: dict[str, Iterable[float]] | None = None,
        initial_radius: float = 0.0,
        quantile_rule: Literal["conformal", "higher"] = "higher",
        ready_on_empty: bool = False,
    ) -> None:
        if calibration_window < 1:
            raise ValueError("calibration_window must be at least 1")
        self._calibration_window = int(calibration_window)
        self._initial_radius = float(initial_radius)
        self._quantile_rule = validate_quantile_rule(quantile_rule)
        self._ready_on_empty = bool(ready_on_empty)
        self._scores: dict[str, deque[float]] = {}
        self.fit(
            {
                partition: [float(score) for score in scores]
                for partition, scores in (initial_scores or {}).items()
            }
        )

    def fit(self, scores: dict[str, list[float]]) -> None:
        self._scores = {
            str(partition): deque(
                (float(score) for score in values),
                maxlen=self._calibration_window,
            )
            for partition, values in scores.items()
        }

    def predict(self, alpha: float, partition: str = "__global__") -> float:
        history = self._scores.get(str(partition), deque(maxlen=self._calibration_window))
        return finite_sample_radius(
            list(history),
            float(alpha),
            self._initial_radius,
            quantile_rule=self._quantile_rule,
        )

    def update(self, new_score: float, partition: str = "__global__") -> None:
        key = str(partition)
        self._scores.setdefault(key, deque(maxlen=self._calibration_window)).append(
            float(new_score)
        )

    def ready(self, partition: str = "__global__", alpha: float | None = None) -> bool:
        if alpha is None:
            alpha = 0.5
        history = self._scores.get(str(partition))
        if history is None or not history:
            return self._ready_on_empty
        return bool(np.isfinite(self.predict(float(alpha), partition)))

    def get_state(self) -> dict:
        return {
            "calibration_window": self._calibration_window,
            "quantile_rule": self._quantile_rule,
            "ready_on_empty": self._ready_on_empty,
            "score_history": {
                partition: np.asarray(list(scores), dtype=float)
                for partition, scores in self._scores.items()
            },
        }

    def set_state(self, state: dict) -> None:
        """Restore the fields emitted by get_state()."""
        self._calibration_window = int(state.get("calibration_window", self._calibration_window))
        self._quantile_rule = validate_quantile_rule(
            state.get("quantile_rule", self._quantile_rule)
        )
        self._ready_on_empty = bool(state.get("ready_on_empty", self._ready_on_empty))
        score_history = state.get("score_history", {})
        if not isinstance(score_history, dict):
            raise ValueError("calibrator score_history must be a mapping")
        self._scores = {
            str(partition): deque(
                (float(score) for score in values),
                maxlen=self._calibration_window,
            )
            for partition, values in score_history.items()
        }
