"""Spread adapters: turn per-origin centers and radii into interval bounds.

A spread is the fourth interchangeable part of the conformal runtime (see
:class:`~calibre.conformal.protocols.Spread`). :class:`AnalyticRadius` is the
point-forecast adapter — ``center +/- radius`` — and the only one today; a
draws-based adapter joins it later behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibre.conformal.protocols import SpreadContext


@dataclass(frozen=True, slots=True)
class AnalyticRadius:
    """Symmetric ``center +/- radius`` spread for point forecasts.

    Reproduces the runtime's inline interval arithmetic exactly: bounds are
    ``centers +/- radii`` where ``issue`` is true and ``NaN`` elsewhere, under the
    same ``np.errstate(invalid="ignore")`` so non-finite radii propagate to
    ``NaN`` bit-identically. Stateless and construction-argument-free.
    """

    def to_interval(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
        issue: np.ndarray,
        *,
        context: SpreadContext | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` as ``centers +/- radii`` gated by ``issue``.

        ``context`` is accepted to satisfy the widened :class:`Spread` interface
        and is deliberately ignored — the point path takes no per-origin frame —
        so the bounds stay bit-identical to the runtime's original arithmetic.
        """
        with np.errstate(invalid="ignore"):
            lower = np.where(issue, centers - radii, np.nan)
            upper = np.where(issue, centers + radii, np.nan)
        return lower, upper
