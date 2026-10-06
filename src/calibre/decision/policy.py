"""Ordering policies: from a calibrated demand bound to an order quantity."""

import numpy as np


def critical_ratio(holding: float | np.ndarray, shortage: float | np.ndarray) -> np.ndarray:
    """Newsvendor level `shortage / (holding + shortage)`, per node when costs are arrays.

    An order at this quantile of demand minimizes the expected holding and shortage cost.
    """
    holding, shortage = (
        np.asarray(holding, dtype=np.float64),
        np.asarray(shortage, dtype=np.float64),
    )
    if (holding <= 0).any() or (shortage <= 0).any():
        raise ValueError("holding and shortage costs must be positive")
    return shortage / (holding + shortage)


def order_up_to(bound: np.ndarray, on_hand: np.ndarray, in_transit: np.ndarray) -> np.ndarray:
    """Order enough to raise the inventory position to `bound`, never below zero.

    The position is the stock on hand `[B]` plus the pipeline `in_transit` `[B, L]`. The
    bound covers demand over the lead time and the review period.
    """
    position = on_hand + in_transit.sum(axis=1)
    return np.maximum(bound - position, 0).astype(np.float32)
