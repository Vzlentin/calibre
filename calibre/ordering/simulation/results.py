"""Per-period result record for the inventory simulator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PeriodResult:
    """Snapshot of one product over one simulation period.

    Inventory accounting is filled by the inventory rule; ``costs`` is filled
    by the cost model after the rule has produced this result.
    """

    unique_id: str
    period: int
    start_inventory: float
    arrivals: float
    demand: float
    sales: float
    missed_sales: float
    end_inventory: float
    order: float
    costs: dict[str, float] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return float(sum(self.costs.values()))

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "unique_id": self.unique_id,
            "period": self.period,
            "start_inventory": self.start_inventory,
            "arrivals": self.arrivals,
            "demand": self.demand,
            "sales": self.sales,
            "missed_sales": self.missed_sales,
            "end_inventory": self.end_inventory,
            "order": self.order,
        }
        for component, value in self.costs.items():
            record[f"{component}_cost"] = value
        return record
