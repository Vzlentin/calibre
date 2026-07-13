"""Apply suite-wide numeric gate/witness collection invariants."""

from __future__ import annotations

import pytest

from oracle.witnesses import (
    WitnessDeclaration,
    WitnessPairingError,
    require_exact_witnesses,
)

_TIERS = tuple(f"tier{number}" for number in range(5))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Fail collection for every orphaned, cross-tier, or self-owned witness."""
    gates = _marker_declarations(items, "oracle_gate")
    witnesses = _marker_declarations(items, "oracle_witness")
    try:
        require_exact_witnesses(gates, witnesses)
    except WitnessPairingError as error:
        raise pytest.UsageError(str(error)) from error


def _marker_declarations(
    items: list[pytest.Item], marker_name: str
) -> tuple[WitnessDeclaration, ...]:
    declarations: list[WitnessDeclaration] = []
    for item in items:
        markers = tuple(item.iter_markers(marker_name))
        if len(markers) > 1:
            raise pytest.UsageError(f"{item.nodeid} declares {marker_name} more than once")
        if not markers:
            continue
        tiers = tuple(tier for tier in _TIERS if item.get_closest_marker(tier) is not None)
        if len(tiers) != 1:
            raise pytest.UsageError(
                f"{item.nodeid} declares {marker_name} without exactly one tier marker"
            )
        marker = markers[0]
        if len(marker.args) != 1 or marker.kwargs or not isinstance(marker.args[0], str):
            raise pytest.UsageError(
                f"{item.nodeid} must declare {marker_name} with one string gate ID"
            )
        identifier = marker.args[0]
        if not identifier or identifier != identifier.strip():
            raise pytest.UsageError(f"{item.nodeid} declares an invalid {marker_name} gate ID")
        declarations.append(WitnessDeclaration(tiers[0], identifier, item.nodeid))
    return tuple(declarations)
