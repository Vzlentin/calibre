"""Prove the independent oracle reference and mechanical witness contract."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from oracle import reference as _reference
from oracle.reference import (
    ReferenceInputError,
    ReferenceOrder,
    ReferenceSeries,
    calculate_reference_trajectory,
)
from oracle.witnesses import (
    WitnessDeclaration,
    WitnessPairingError,
    require_exact_witnesses,
)

pytestmark = pytest.mark.tier1


def test_closed_form_reference_recomputes_arrival_conservation_and_cost_by_hand() -> None:
    periods = ("p0", "p1", "p2", "p3")
    series = (
        ReferenceSeries("a", initial_on_hand=3.0, holding_rate=1.0, shortage_rate=4.0),
        ReferenceSeries("b", initial_on_hand=1.0, holding_rate=2.0, shortage_rate=5.0),
    )
    demand = {
        ("a", "p0"): 2.0,
        ("a", "p1"): 2.0,
        ("a", "p2"): 4.0,
        ("a", "p3"): 1.0,
        ("b", "p0"): 0.0,
        ("b", "p1"): 2.0,
        ("b", "p2"): 1.0,
        ("b", "p3"): 1.0,
    }
    orders = (
        ReferenceOrder("a", origin_index=0, quantity=5.0),
        ReferenceOrder("b", origin_index=1, quantity=2.0),
    )

    trajectory = calculate_reference_trajectory(
        periods=periods,
        series=series,
        demand=demand,
        orders=orders,
        lead_time=2,
    )

    rows = {(row.series_key, row.period): row for row in trajectory.rows}
    assert rows[("a", "p2")].arrivals == 5.0
    assert rows[("b", "p3")].arrivals == 2.0
    assert rows[("a", "p1")].shortage == 1.0
    assert rows[("b", "p1")].shortage == 1.0
    assert rows[("a", "p2")].closing == 1.0
    assert rows[("b", "p3")].closing == 1.0
    assert trajectory.cost_by_period == {"p0": 3.0, "p1": 9.0, "p2": 6.0, "p3": 2.0}
    assert trajectory.total_cost == 20.0
    assert demand[("a", "p2")] == 4.0
    assert orders[0].quantity == 5.0


def test_closed_form_reference_refuses_incomplete_shared_inputs() -> None:
    with pytest.raises(ReferenceInputError, match="demand keys mismatch"):
        calculate_reference_trajectory(
            periods=("p0",),
            series=(ReferenceSeries("a", 1.0, 1.0, 1.0),),
            demand={},
            orders=(),
            lead_time=1,
        )

    with pytest.raises(ReferenceInputError, match="arrival drain"):
        calculate_reference_trajectory(
            periods=("p0", "p1"),
            series=(ReferenceSeries("a", 1.0, 1.0, 1.0),),
            demand={("a", "p0"): 0.0, ("a", "p1"): 0.0},
            orders=(ReferenceOrder("a", origin_index=1, quantity=99.0),),
            lead_time=1,
        )


def test_closed_form_reference_imports_no_production_settlement_code() -> None:
    path = Path(_reference.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert all(not name.startswith("newcalibre") for name in imported)


def test_deleting_numeric_gate_witness_fails_collection_contract() -> None:
    project_root = Path(__file__).parents[2]
    paired = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/meta/paired_gate.py",
            "--collect-only",
            "-q",
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert paired.returncode == 0, paired.stdout + paired.stderr

    orphan = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/meta/orphan_gate.py",
            "--collect-only",
            "-q",
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert orphan.returncode != 0
    assert "meta-orphan" in orphan.stdout + orphan.stderr

    self_owned = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/meta/self_owned_gate.py",
            "--collect-only",
            "-q",
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert self_owned.returncode != 0
    assert "same_node" in self_owned.stdout + self_owned.stderr


@pytest.mark.parametrize(
    ("gate_nodes", "witness_nodes", "message"),
    [
        (("gate-a", "gate-b"), ("witness",), "duplicate_gates"),
        (("gate",), ("witness-a", "witness-b"), "duplicate_witnesses"),
        ((), ("witness",), "orphaned"),
    ],
)
def test_witness_contract_requires_exactly_one_same_tier_pair(
    gate_nodes: tuple[str, ...],
    witness_nodes: tuple[str, ...],
    message: str,
) -> None:
    gates = tuple(WitnessDeclaration("tier3", "a", node) for node in gate_nodes)
    witnesses = tuple(WitnessDeclaration("tier3", "a", node) for node in witness_nodes)
    with pytest.raises(WitnessPairingError, match=message):
        require_exact_witnesses(gates, witnesses)
