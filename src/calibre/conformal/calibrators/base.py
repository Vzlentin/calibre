"""The calibrator contract: what a calibration method receives, keeps, and returns.

A calibrator turns the targets known so far into a threshold per node and column. A
threshold indexes a nested family of bounds, the `Score`: a quantile calibrator ranks
the scores, and a risk calibrator picks the threshold whose bounds have the least loss. Its
state is a nested dict of numpy arrays, and its methods are functions of that state.
So a product can save the state after each origin and continue later, and a backtest
is the same loop as production. A call owns the state it receives: it can write into
those arrays, and the caller continues only with the returned state.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from calibre.conformal.scores import Score

State = dict[str, Any]
"""Nested dict of numpy arrays. `calibre.online.state` flattens it for storage."""

Level = float | np.ndarray
"""Target level: a scalar, or `[N, C]` per node and column. `[N, 1]` is one per node."""


@dataclass(frozen=True)
class Feedback:
    """Targets that became known at one origin. Each row `[K]` is one target column of
    one issuing origin, for all nodes.

    Rows are in origin order. `point` and `target` are the column sums `[K, N]`, NaN
    where the actual is missing. `issued` is the threshold that was issued for each
    row, `censored` marks targets that are lower bounds, and `score` is the family of
    bounds the thresholds index.
    """

    origin: np.ndarray
    column: np.ndarray
    point: np.ndarray
    target: np.ndarray
    issued: np.ndarray
    censored: np.ndarray
    score: Score

    @property
    def scores(self) -> np.ndarray:
        """Nonconformity scores `[K, N]` float32: the smallest threshold that holds the target."""
        return self.score.score(self.target, self.point).astype(np.float32)

    def bounds(self, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bounds `[K, N, G]` that thresholds `[G]` or `[N, G]` would have issued."""
        return self.score.bound(self.point[..., None], threshold)

    def misses(self, n_columns: int) -> tuple[np.ndarray, np.ndarray]:
        """Counts `[N, C]` of misses (score above the issued threshold) and known scores."""
        scores = self.scores
        known = np.isfinite(scores)
        miss = (scores > self.issued) & known
        columns = np.eye(n_columns)[self.column]  # [K, C]
        return miss.T @ columns, known.T @ columns


class Calibrator(ABC):
    """A calibration method. It owns its target level, set at construction."""

    @abstractmethod
    def initial_state(self, n_nodes: int, n_columns: int) -> State:
        """Return the empty state."""

    @abstractmethod
    def update(self, state: State, feedback: Feedback) -> State:
        """Return the state after the feedback. It can reuse the input arrays."""

    @abstractmethod
    def threshold(self, state: State) -> np.ndarray:
        """Return thresholds `[N, C]` at the target level. inf means not ready."""


class QuantileCalibrator(Calibrator):
    """A calibrator that can give its threshold at any level, not only its own.

    Wrappers such as `ACI` move the level and ask the base for the threshold there.
    """

    level: Level

    @abstractmethod
    def threshold_at(self, state: State, level: Level) -> np.ndarray:
        """Return thresholds `[N, C]` at `level` instead of the target level."""


def check_level(level: Level) -> Level:
    """A scalar or `[N, C]` level strictly between zero and one."""
    values = np.asarray(level, dtype=np.float64)
    if values.ndim not in (0, 2):
        raise ValueError(f"level must be a scalar or [N, C], got shape {values.shape}")
    if not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
        raise ValueError("level must be strictly between zero and one")
    return float(values) if values.ndim == 0 else values
