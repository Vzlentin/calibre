"""Lost-sales inventory: order up to a bound, then settle one period at a time."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Week:
    """One settled period, per series `[B]`."""

    arrivals: np.ndarray
    demand: np.ndarray
    sales: np.ndarray
    missed: np.ndarray
    end_inventory: np.ndarray
    holding_cost: np.ndarray
    shortage_cost: np.ndarray


def order_up_to(bound: np.ndarray, on_hand: np.ndarray, in_transit: np.ndarray) -> np.ndarray:
    """Order enough to raise the inventory position to `bound`, never below zero."""
    position = on_hand + in_transit.sum(axis=1)
    return np.maximum(bound - position, 0).astype(np.float32)


def settle(
    on_hand: np.ndarray,
    in_transit: np.ndarray,
    demand: np.ndarray,
    order: np.ndarray,
    holding: float,
    shortage: float,
) -> tuple[np.ndarray, np.ndarray, Week]:
    """Receive the first pipeline slot, sell, then add `order` at the end of the pipeline.

    `in_transit` is `[B, L]`: slot 0 arrives this period. Unmet demand is lost. Returns
    the new on-hand stock, the new pipeline, and the period's record.
    """
    if not np.isfinite(demand).all():
        raise ValueError("settlement needs finite demand for every series")
    arrivals = in_transit[:, 0]
    start = on_hand + arrivals
    sales = np.minimum(start, demand)
    missed = demand - sales
    end = start - sales
    pipeline = np.column_stack([in_transit[:, 1:], order])
    week = Week(arrivals, demand, sales, missed, end, end * holding, missed * shortage)
    return end, pipeline, week
