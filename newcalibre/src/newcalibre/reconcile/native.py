"""Implement the native no-op and all-members-present bottom-up strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from newcalibre.domain import HierarchyIndex
from newcalibre.reconcile.apply import apply_bottom_up, apply_none
from newcalibre.reconcile.protocol import (
    MatrixCapability,
    ReconcilerDeclaration,
    ReconciliationContext,
    ReconciliationInputFamily,
)

NONE: Final = "none"
BOTTOM_UP: Final = "bottom_up"

NONE_DECLARATION: Final = ReconcilerDeclaration(
    name=NONE,
    input_family=ReconciliationInputFamily.SYNTHESIS,
    requires_fitted_values=False,
    matrix_capability=MatrixCapability.SPARSE_CAPABLE,
)
BOTTOM_UP_DECLARATION: Final = ReconcilerDeclaration(
    name=BOTTOM_UP,
    input_family=ReconciliationInputFamily.SYNTHESIS,
    requires_fitted_values=False,
    matrix_capability=MatrixCapability.SPARSE_CAPABLE,
)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class NoReconciliation:
    """Validate point rows and enforce support without reconciliation math."""

    @property
    def declaration(self) -> ReconcilerDeclaration:
        """Return the native no-op declaration."""
        return NONE_DECLARATION

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: HierarchyIndex | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        """Return support-valid points without applying reconciliation math."""
        return apply_none(frame, hierarchy, context, declaration=self.declaration)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class BottomUpReconciler:
    """Synthesize only aggregates whose complete bottom membership is present."""

    @property
    def declaration(self) -> ReconcilerDeclaration:
        """Return the native bottom-up declaration."""
        return BOTTOM_UP_DECLARATION

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: HierarchyIndex | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        """Preserve bottom rows and append deterministic coherent aggregates."""
        return apply_bottom_up(frame, hierarchy, context, declaration=self.declaration)


def build_none() -> NoReconciliation:
    """Build a fresh native no-op reconciler."""
    return NoReconciliation()


def build_bottom_up() -> BottomUpReconciler:
    """Build a fresh native bottom-up reconciler."""
    return BottomUpReconciler()
