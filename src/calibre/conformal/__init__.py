"""Conformal calibration: a Score, a Calibrator, and one driver that owns causality.

A `Score` says how wrong a forecast was, in columns: one per step, or one per window
of steps. A `Calibrator` turns the scores known so far into a threshold per node and
column. `Conformal` runs them origin by origin and gives each calibrator only the
scores whose targets are known, with the threshold that was issued for them.

Calibrator state is a nested dict of numpy arrays, and calibrators are functions
of it. So a product can save the state after each origin and continue later, and a
backtest is the same loop as production. A call owns the state it receives: it can
write into those arrays, and the caller continues only with the returned state.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

State = dict[str, Any]
"""Nested dict of numpy arrays. `calibre.conformal.state` flattens it for storage."""


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


class Score(Protocol):
    def cover(self, horizon: int) -> np.ndarray:
        """`[C, H]` bool: the forecast steps that each score column sums."""
        ...

    def score(self, target: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Nonconformity per column from targets and points of the same shape."""
        ...

    def interval(self, point: np.ndarray, threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds `[N, C]` that hold exactly when score <= threshold."""
        ...


class Calibrator(Protocol):
    def init(self, n_nodes: int, n_columns: int) -> State:
        """Return the empty state."""
        ...

    def update(self, state: State, feedback: Feedback, level: float) -> State:
        """Return the state after the feedback. It can reuse the input arrays."""
        ...

    def threshold(self, state: State, level: float) -> np.ndarray:
        """Return thresholds `[N, C]`. inf means not ready."""
        ...
