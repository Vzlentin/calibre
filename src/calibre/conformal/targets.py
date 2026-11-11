"""Targets: the steps that each column sums."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def _size(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _steps(value: object, name: str) -> int | np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iu":
            raise TypeError(f"{name} must be an int or an array of int")
        if value.ndim != 1:
            raise ValueError(f"{name} must be one value per node")
        array = np.array(value, dtype=np.int64)
        array.flags.writeable = False
        return array
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int or an array of int")
    return value


class Target(ABC):
    """Which steps each column sums: `cover` is `[N, C, H]` bool."""

    @abstractmethod
    def cover(self, n_nodes: int, horizon: int) -> np.ndarray: ...


@dataclass(frozen=True)
class Step(Target):
    """H columns: column c covers step c + 1."""

    def cover(self, n_nodes: int, horizon: int) -> np.ndarray:
        n_nodes, horizon = _size(n_nodes, "n_nodes"), _size(horizon, "horizon")
        return np.broadcast_to(np.eye(horizon, dtype=bool), (n_nodes, horizon, horizon)).copy()


@dataclass(frozen=True)
class Span(Target):
    """One column: the steps from `first` to `last` of each node, 1-based."""

    first: int | np.ndarray  # [N] int64 when per node
    last: int | np.ndarray  # [N] int64 when per node

    def __post_init__(self) -> None:
        object.__setattr__(self, "first", _steps(self.first, "first"))
        object.__setattr__(self, "last", _steps(self.last, "last"))

    def cover(self, n_nodes: int, horizon: int) -> np.ndarray:
        n_nodes, horizon = _size(n_nodes, "n_nodes"), _size(horizon, "horizon")
        first = np.broadcast_to(np.asarray(self.first, dtype=np.int64), (n_nodes,))
        last = np.broadcast_to(np.asarray(self.last, dtype=np.int64), (n_nodes,))
        for value in (self.first, self.last):
            if isinstance(value, np.ndarray) and len(value) != n_nodes:
                raise ValueError("a Span array needs one value per node")
        if (first < 1).any() or (first > last).any() or (last > horizon).any():
            raise ValueError("a Span needs 1 <= first <= last <= horizon")
        step = np.arange(1, horizon + 1)
        inside = (step >= first[:, None]) & (step <= last[:, None])
        return inside[:, None, :]
