"""Structural protocols for conformal spreads, scores, calibrators, and controllers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpreadContext:
    """Per-origin context a draws-based spread reads; ``None`` for the point path.

    Carries the slice of the origin frame whose rows align positionally with the
    ``centers``/``radii``/``issue`` arrays passed to :meth:`Spread.to_interval`,
    the working ``alpha``, the in-sample fitted-value sidecar (``unique_id``,
    ``ds``, ``y``, ``model_name``, ``fitted_y_hat``) the residual bootstrap reads,
    a base seed for per-cross-section RNG derivation, the cumulative protection
    period (``None`` in per-horizon mode), and the held-out half-widths a coherent
    spread re-anchors its draw spread to. :class:`AnalyticRadius` ignores it
    entirely, so the default path never constructs one.

    ``held_out_half_width`` maps ``"{model}:h{h}:{node}"`` (and the cumulative
    ``"{model}:cumulative:{node}"``) to the held-out ``(1-alpha)`` radius the
    per-partition marginal calibrator already tracks for that partition. It is
    per-origin and depends on accumulated out-of-sample state, so it threads
    through context rather than construction; ``None`` leaves the in-sample draw
    geometry unscaled.

    ``joint_inflation`` maps the same per-node keys to a joint/simultaneous
    coverage factor ``kappa >= 1`` applied *after* the held-out factor clears the
    ``MAX_HELD_OUT_FACTOR`` clamp, so it can only ever widen a node and never flip
    a clamped node back to unscaled. ``None`` (the U4 / default behaviour) leaves
    the held-out width un-inflated.
    """

    frame: pd.DataFrame
    alpha: float
    fitted_values: pd.DataFrame | None = None
    seed: int = 0
    protection_period: int | None = None
    held_out_half_width: dict[str, float] | None = None
    joint_inflation: dict[str, float] | None = None


class Spread(Protocol):
    """Turns per-origin centers and calibrated radii into interval bounds.

    The fourth interchangeable part of the conformal runtime, alongside
    :class:`Score`, :class:`Calibrator`, and :class:`Controller`. It owns the
    final step of "what form the predictive uncertainty takes": mapping
    ``(center, radius)`` into ``(lower, upper)``. The interface is vectorised;
    the cumulative call site passes length-1 arrays.

    ``context`` is a keyword-only, defaulted :class:`SpreadContext` carrying the
    per-origin frame slice a draws-based adapter needs. It defaults to ``None``
    so the point adapter (:class:`~calibre.conformal.spread.AnalyticRadius`) is
    unchanged and the default path stays byte-identical.
    """

    def to_interval(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        issue: np.ndarray,
        *,
        context: SpreadContext | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...


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
