"""Targets: which quantity a score column bounds, as a set of forecast steps."""

from abc import ABC, abstractmethod

import numpy as np


class Target(ABC):
    @abstractmethod
    def cover(self, horizon: int) -> np.ndarray:
        """`[C, H]` bool: the forecast steps that each column sums."""


class Step(Target):
    """One column per forecast step."""

    def cover(self, horizon: int) -> np.ndarray:
        return np.eye(horizon, dtype=bool)


class LeadTime(Target):
    """One column: the total over the first `steps` steps, for example lead-time demand.

    A column is known only when every step it covers is known.
    """

    def __init__(self, steps: int) -> None:
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        self.steps = steps

    def cover(self, horizon: int) -> np.ndarray:
        if self.steps > horizon:
            raise ValueError(f"lead time of {self.steps} steps is longer than horizon {horizon}")
        return (np.arange(horizon) < self.steps)[None, :]


def columns(values: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """Steps `[..., H]` to columns `[..., C]`: a sum, or any for flags.

    A missing covered step makes the column missing. Uncovered steps are ignored.
    """
    if cover.shape[0] == cover.shape[1] and (cover == np.eye(len(cover), dtype=bool)).all():
        return values
    reduce = np.any if values.dtype == bool else np.sum
    out = [reduce(values[..., steps], axis=-1) for steps in cover]
    return np.stack(out, axis=-1)
