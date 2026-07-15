"""Enforce exact same-tier numeric-gate/witness pairing during collection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


class WitnessPairingError(ValueError):
    """Report an orphaned or multiply owned numeric gate identifier."""


@dataclass(frozen=True, slots=True)
class WitnessDeclaration:
    """Bind a stable gate ID to its tier and owning pytest node."""

    tier: str
    identifier: str
    nodeid: str


def require_exact_witnesses(
    gates: Iterable[WitnessDeclaration],
    witnesses: Iterable[WitnessDeclaration],
) -> None:
    """Require exactly one same-tier witness for every stable gate ID."""
    gate_values = tuple(gates)
    witness_values = tuple(witnesses)
    gate_counts = Counter((item.tier, item.identifier) for item in gate_values)
    witness_counts = Counter((item.tier, item.identifier) for item in witness_values)
    invalid_gates = sorted(key for key, count in gate_counts.items() if count != 1)
    invalid_witnesses = sorted(key for key, count in witness_counts.items() if count != 1)
    missing = sorted(set(gate_counts) - set(witness_counts))
    orphaned = sorted(set(witness_counts) - set(gate_counts))
    gate_nodes = {(item.tier, item.identifier, item.nodeid) for item in gate_values}
    witness_nodes = {(item.tier, item.identifier, item.nodeid) for item in witness_values}
    overlapping_nodes = sorted(gate_nodes & witness_nodes)
    if invalid_gates or invalid_witnesses or missing or orphaned or overlapping_nodes:
        raise WitnessPairingError(
            "numeric gate/witness mismatch: "
            f"duplicate_gates={invalid_gates!r}, duplicate_witnesses={invalid_witnesses!r}, "
            f"missing={missing!r}, orphaned={orphaned!r}, "
            f"same_node={overlapping_nodes!r}"
        )


def require_exact_inventory(
    gates: Iterable[WitnessDeclaration],
    witnesses: Iterable[WitnessDeclaration],
    *,
    required_gates: Iterable[WitnessDeclaration],
    required_witnesses: Iterable[WitnessDeclaration],
) -> None:
    """Require the collected declarations to equal one stable named inventory."""

    def counts(values: Iterable[WitnessDeclaration]) -> Counter[tuple[str, str, str]]:
        return Counter((item.tier, item.identifier, item.nodeid) for item in values)

    gate_counts = counts(gates)
    witness_counts = counts(witnesses)
    required_gate_counts = counts(required_gates)
    required_witness_counts = counts(required_witnesses)
    missing_gates = sorted((required_gate_counts - gate_counts).elements())
    unexpected_gates = sorted((gate_counts - required_gate_counts).elements())
    missing_witnesses = sorted((required_witness_counts - witness_counts).elements())
    unexpected_witnesses = sorted((witness_counts - required_witness_counts).elements())
    if missing_gates or unexpected_gates or missing_witnesses or unexpected_witnesses:
        raise WitnessPairingError(
            "required Tier 3 oracle inventory mismatch: "
            f"missing_gates={missing_gates!r}, unexpected_gates={unexpected_gates!r}, "
            f"missing_witnesses={missing_witnesses!r}, "
            f"unexpected_witnesses={unexpected_witnesses!r}"
        )
