"""Generic discrete-period inventory simulator.

Domain-agnostic building blocks for evaluating ordering decisions: the
``Simulator`` advances per-product state via a pluggable ``InventoryRule`` and
accumulates per-component costs via a pluggable ``CostModel``.
"""

from calibre.simulation.costs import CostModel, LinearCostModel
from calibre.simulation.results import PeriodResult
from calibre.simulation.rules import InventoryRule, LostSalesRule
from calibre.simulation.simulator import Simulator
from calibre.simulation.state import ProductState, make_pipeline

__all__ = [
    "CostModel",
    "InventoryRule",
    "LinearCostModel",
    "LostSalesRule",
    "PeriodResult",
    "ProductState",
    "Simulator",
    "make_pipeline",
]
