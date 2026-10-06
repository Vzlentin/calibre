"""The calibrator contract: what a calibration method receives, keeps, and returns.

A calibrator turns the scores known so far into a threshold per node and column. Its
state is a nested dict of numpy arrays, and its methods are functions of that state.
So a product can save the state after each origin and continue later, and a backtest
is the same loop as production. A call owns the state it receives: it can write into
those arrays, and the caller continues only with the returned state.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

State = dict[str, Any]
"""Nested dict of numpy arrays. `calibre.online.state` flattens it for storage."""


@dataclass(frozen=True)
class Feedback:
    """Scores that became known at one origin. Each row `[K]` is one score column of
    one issuing origin, for all nodes.

    Rows are in origin order. `scores` is `[K, N]`, NaN where the actual is missing.
    `issued` is the threshold that was issued for each score, and `censored` marks
    targets that are lower bounds.
    """

    origin: np.ndarray
    column: np.ndarray
    scores: np.ndarray
    issued: np.ndarray
    censored: np.ndarray

    @abstractmethod
    def misses(self, n_columns: int) -> tuple[np.ndarray, np.ndarray]:
        """Counts `[N, C]` of misses (score above the issued threshold) and known scores."""
        known = np.isfinite(self.scores)
        miss = (self.scores > self.issued) & known
        columns = np.eye(n_columns)[self.column]  # [K, C]
        return miss.T @ columns, known.T @ columns


class Calibrator(ABC):
    @abstractmethod
    def init(self, n_nodes: int, n_columns: int) -> State:
        """Return the empty state."""

    @abstractmethod
    def update(self, state: State, feedback: Feedback, level: float) -> State:
        """Return the state after the feedback. It can reuse the input arrays."""

    @abstractmethod
    def threshold(self, state: State, level: float) -> np.ndarray:
        """Return thresholds `[N, C]`. inf means not ready."""
