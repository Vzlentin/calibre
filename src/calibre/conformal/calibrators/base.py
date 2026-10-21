"""The calibrator contract: what a calibration method receives, keeps, and returns.

A calibrator turns known targets into bound parameters per node and column. Scalar
thresholds index nested bounds; asymmetric intervals use a lower/upper pair instead.
Its state is a nested dict of numpy arrays, and its methods are functions of that state.
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


class Scorecaster(ABC):
    """Stateless anticipation from revealed score histories, with no forecast dependency."""

    @abstractmethod
    def predict(
        self, scores: np.ndarray, counts: np.ndarray, scale: np.ndarray, ahead: np.ndarray
    ) -> np.ndarray:
        """Return finite predictions [N, C] without changing the read-only inputs.

        scores [N, C, K] are chronological and right-aligned, with leading NaN;
        counts [N, C] give retained lengths, scale [N, C] the fixed seed scale,
        and ahead [C] positive forecast horizons in the retained-score clock.
        Empty cells predict zero. Recompute fits; retain no learned instance state.
        """


@dataclass(frozen=True)
class Feedback:
    """Targets that became known at one origin. Each row `[K]` is one target column of
    one issuing origin, for all nodes.

    Rows are in origin order. `point` and `target` are column sums `[K, N]`, NaN
    where the actual is missing. `issued` is `[K, N]`, or `[K, N, 2]` for paired
    corrections. `censored` marks targets that are lower bounds, and `score` defines
    bounds the thresholds index, so a calibrator can find the bounds of any threshold.
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
        """Scores `[K, N]` float32, signed residuals for asymmetric intervals."""
        return self.score.score(self.target, self.point).astype(np.float32)

    def misses(self, n_columns: int) -> tuple[np.ndarray, np.ndarray]:
        """Counts `[N, C]` of misses against issued parameters and known scores."""
        scores = self.scores
        known = np.isfinite(scores)
        if self.score.parameter_shape():
            lower, upper = self.score.bound(self.point, self.issued)
            miss = ((self.target < lower) | (self.target > upper)) & known
        else:
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
        """Return scalar `[N, C]` or paired `[N, C, 2]` bound parameters.

        An unready cell returns scalar inf or the pair [-inf, inf]. Adaptive
        controllers can also intentionally issue signed infinity after initialization.
        """


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
