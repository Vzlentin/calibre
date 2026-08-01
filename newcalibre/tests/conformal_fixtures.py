"""Share canonical conformal batch fixtures across test tiers."""

from collections.abc import Iterable

from newcalibre.conformal import DeliveryBatch, ResolvedObservation


def delivery_batch(
    label: str,
    observations: Iterable[ResolvedObservation],
) -> DeliveryBatch:
    """Build one semantic partition row through the batch constructor."""
    return DeliveryBatch({label: observations})
