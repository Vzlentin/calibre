from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
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

    def predict_batch(
        self,
        alpha: float,
        partitions: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized :meth:`predict` + issue-readiness over many partitions.

        Returns ``(radii, ready)`` aligned with ``partitions``. Row ``i`` is
        exactly ``predict(alpha, partitions[i])`` and the issue condition the
        per-row path computes (``ready(partition, alpha)`` with a finite
        radius). Windows are bucketed by length so each bucket is one
        ``np.quantile`` call, which keeps the math bit-identical to
        :func:`finite_sample_radius`.
        """
        alpha = float(alpha)
        count = len(partitions)
        radii = np.full(count, self._initial_radius, dtype=float)
        nonempty = np.zeros(count, dtype=bool)

        empty: deque[float] = deque()
        windows = [self._scores.get(str(partition)) or empty for partition in partitions]
        by_length: dict[int, list[int]] = {}
        for position, window in enumerate(windows):
            if window:
                nonempty[position] = True
                by_length.setdefault(len(window), []).append(position)

        clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
        for length, positions in by_length.items():
            if self._quantile_rule == "higher" and alpha <= 1.0 / (length + 1):
                radii[positions] = np.inf
                continue
            matrix = np.array([windows[position] for position in positions], dtype=float)
            if self._quantile_rule == "higher":
                radii[positions] = np.quantile(matrix, 1.0 - clipped_alpha, axis=1, method="higher")
            else:
                rank = int(np.ceil((length + 1) * (1.0 - clipped_alpha))) - 1
                rank = min(max(rank, 0), length - 1)
                radii[positions] = np.sort(matrix, axis=1)[:, rank]

        ready = np.isfinite(radii) & (nonempty | self._ready_on_empty)
        return radii, ready

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
