"""Per-product inventory state for the simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ProductState:
    """Inventory state for a single product at the start of a simulation period.

    The in-transit pipeline is a deque of length ``lead_time_depth``. Each
    period, the leftmost slot becomes available as arrivals and the new order
    is appended on the right. With ``lead_time_depth=N``, an order placed at
    the end of period ``t`` is available for sales starting period ``t+N+1``.

    Attributes:
        unique_id: Series identifier (matches forecast frame ``unique_id``).
        end_inventory: On-hand inventory at end of last period.
        pipeline: In-transit quantities, ordered by arrival period (leftmost
            arrives next period).
        cumulative_costs: Per-component cumulative cost dictionary keyed by
            cost-model component name (e.g. ``{"holding": ..., "shortage": ...}``).
    """

    unique_id: str
    end_inventory: float
    pipeline: deque[float]
    cumulative_costs: dict[str, float] = field(default_factory=dict)

    @property
    def lead_time_depth(self) -> int:
        return self.pipeline.maxlen if self.pipeline.maxlen is not None else len(self.pipeline)

    @property
    def in_transit_total(self) -> float:
        return float(sum(self.pipeline))

    @property
    def inventory_position(self) -> float:
        """End inventory plus all in-transit quantities."""
        return float(self.end_inventory) + self.in_transit_total

    @property
    def total_cost(self) -> float:
        return float(sum(self.cumulative_costs.values()))

    def copy(self) -> ProductState:
        return ProductState(
            unique_id=self.unique_id,
            end_inventory=float(self.end_inventory),
            pipeline=deque(self.pipeline, maxlen=self.pipeline.maxlen),
            cumulative_costs=dict(self.cumulative_costs),
        )


def make_pipeline(in_transit: list[float], lead_time_depth: int) -> deque[float]:
    """Build a fixed-length pipeline deque, padding with zeros if needed.

    ``in_transit[0]`` is the quantity arriving next period, ``in_transit[-1]``
    is the most recently placed order.
    """
    if lead_time_depth < 0:
        raise ValueError("lead_time_depth must be non-negative")
    if len(in_transit) > lead_time_depth:
        raise ValueError(
            f"in_transit length {len(in_transit)} exceeds lead_time_depth {lead_time_depth}"
        )
    padded = list(in_transit) + [0.0] * (lead_time_depth - len(in_transit))
    return deque((float(x) for x in padded), maxlen=lead_time_depth)
