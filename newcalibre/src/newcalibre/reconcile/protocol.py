"""Define the fixed point-reconciliation contract and declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import pandas as pd

from newcalibre.domain import FittedValues, HierarchyIndex, TargetSupport


class ReconciliationInputFamily(StrEnum):
    """Declare which hierarchy rows a strategy requires as input."""

    SYNTHESIS = "synthesis"
    PROJECTION = "projection"


class MatrixCapability(StrEnum):
    """Declare the summing-matrix representation a strategy can consume."""

    SPARSE_CAPABLE = "sparse_capable"
    DENSE_ONLY = "dense_only"


@dataclass(frozen=True, slots=True)
class ReconcilerDeclaration:
    """Describe run-preparation requirements without executing a strategy."""

    name: str
    input_family: ReconciliationInputFamily
    requires_fitted_values: bool
    matrix_capability: MatrixCapability

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("reconciliation strategy name must be a non-empty trimmed string")
        if self.name != self.name.casefold():
            raise ValueError("reconciliation strategy name must use canonical lowercase")
        if not isinstance(self.input_family, ReconciliationInputFamily):
            raise TypeError("reconciliation input family must be a ReconciliationInputFamily")
        if not isinstance(self.requires_fitted_values, bool):
            raise TypeError("requires_fitted_values must be boolean")
        if not isinstance(self.matrix_capability, MatrixCapability):
            raise TypeError("matrix capability must be a MatrixCapability")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReconciliationContext:
    """Carry optional per-origin sidecars outside the forecast frame."""

    fitted_values: FittedValues | None = None
    target_support: TargetSupport

    def __post_init__(self) -> None:
        if self.fitted_values is not None and not isinstance(self.fitted_values, FittedValues):
            raise TypeError("reconciliation fitted values must be FittedValues or None")
        if not isinstance(self.target_support, TargetSupport):
            raise TypeError("reconciliation target support must be a TargetSupport")


@runtime_checkable
class Reconciler(Protocol):
    """Reconcile point forecasts through one fixed frame-level signature."""

    @property
    def declaration(self) -> ReconcilerDeclaration:
        """Return immutable run-preparation requirements."""
        ...

    def __call__(
        self,
        frame: pd.DataFrame,
        hierarchy: HierarchyIndex | None,
        context: ReconciliationContext,
    ) -> pd.DataFrame:
        """Return point forecasts reconciled inside context.target_support."""
        ...
