"""Structural protocols for conformal scores, calibrators, and controllers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Score(Protocol):
    """Callable that maps actuals and predictions to nonconformity scores."""

    def __call__(self, y_true, y_pred, *, mask=None, weights=None) -> np.ndarray: ...


class Calibrator(Protocol):
    """Maps a target ``alpha`` and partition to a conformal radius."""

    def fit(self, scores: dict[str, list[float]]) -> None: ...

    def predict(self, alpha: float, partition: str = "__global__") -> float: ...

    def predict_batch(
        self,
        alpha: float,
        partitions: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def update(self, new_score: float, partition: str = "__global__") -> None: ...

    def ready(self, partition: str = "__global__", alpha: float | None = None) -> bool: ...

    def get_state(self) -> dict: ...

    def set_state(self, state: dict) -> None: ...


class Controller(Protocol):
    """Tracks observed errors and supplies the working ``alpha``."""

    def observe(self, y_true, y_pred, h: int) -> None: ...

    def get_alpha(self) -> float: ...

    def drift(self) -> float | None: ...

    def get_state(self) -> dict: ...

    def set_state(self, state: dict) -> None: ...
