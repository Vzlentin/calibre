from __future__ import annotations

from typing import Protocol

import numpy as np


class Score(Protocol):
    def __call__(self, y_true, y_pred, *, mask=None, weights=None) -> np.ndarray: ...


class Calibrator(Protocol):
    def fit(self, scores: dict[str, list[float]]) -> None: ...

    def predict(self, alpha: float, partition: str = "__global__") -> float: ...

    def update(self, new_score: float, partition: str = "__global__") -> None: ...


class Controller(Protocol):
    def observe(self, y_true, y_pred, h: int) -> None: ...

    def get_alpha(self) -> float: ...

    def get_state(self) -> dict: ...
