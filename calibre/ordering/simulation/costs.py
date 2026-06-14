"""Cost models for the inventory simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from calibre.ordering.simulation.results import PeriodResult
from calibre.ordering.simulation.state import ProductState


class CostModel(Protocol):
    """Compute per-period cost components for a single product."""

    def cost(self, state: ProductState, result: PeriodResult) -> dict[str, float]: ...


_DEFAULT_ATTRIBUTES: dict[str, str] = {
    "holding": "end_inventory",
    "shortage": "missed_sales",
}


@dataclass(frozen=True, slots=True)
class LinearCostModel:
    """Linear cost model: each component is a rate multiplied by a result attribute.

    ``rates`` maps a cost component name to a rate; ``attributes`` maps the same
    component name to a ``PeriodResult`` field whose value is multiplied by the
    rate. The default mapping covers VN2's holding (on end_inventory) and
    shortage (on missed_sales) costs.
    """

    rates: dict[str, float]
    attributes: dict[str, str] | None = None

    def __post_init__(self) -> None:
        attributes = dict(self.attributes if self.attributes is not None else _DEFAULT_ATTRIBUTES)
        unknown = set(self.rates) - set(attributes)
        if unknown:
            raise ValueError(
                f"LinearCostModel rates reference unknown components {sorted(unknown)}; "
                f"add them to attributes={sorted(attributes)}"
            )
        object.__setattr__(self, "attributes", attributes)

    def cost(self, state: ProductState, result: PeriodResult) -> dict[str, float]:
        del state
        assert self.attributes is not None
        return {
            component: float(rate) * float(getattr(result, self.attributes[component]))
            for component, rate in self.rates.items()
        }
