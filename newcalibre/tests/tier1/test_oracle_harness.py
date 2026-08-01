"""Prove the independent oracle reference and mechanical witness contract."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from newcalibre.domain import (
    ActualsSemantics,
    Calendar,
    CostStructure,
    DecisionTiming,
    InventoryPosition,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.engine import (
    InMemoryLedgerSink,
    SettlementRequest,
    SettlementResult,
    settle,
)
from newcalibre.ledger import BookedCost, SettlementRecord, StockoutTransition
from oracle import reference as _reference
from oracle.reference import (
    ReferenceInputError,
    ReferenceOrder,
    ReferenceRow,
    ReferenceSeries,
    calculate_reference_trajectory,
)
from oracle.witnesses import (
    WitnessDeclaration,
    WitnessPairingError,
    require_exact_inventory,
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
    assert tuple(row.on_order for row in trajectory.rows) == (
        5.0,
        0.0,
        5.0,
        2.0,
        0.0,
        2.0,
        0.0,
        0.0,
    )
    assert trajectory.cost_by_period == {"p0": 3.0, "p1": 9.0, "p2": 6.0, "p3": 2.0}
    assert trajectory.total_cost == 20.0
    assert demand[("a", "p2")] == 4.0
    assert orders[0].quantity == 5.0


def test_closed_form_reference_seeds_every_initial_pipeline_period() -> None:
    trajectory = calculate_reference_trajectory(
        periods=("p0", "p1", "p2", "p3"),
        series=(ReferenceSeries("a", 1.0, 1.0, 4.0),),
        demand={
            ("a", "p0"): 2.0,
            ("a", "p1"): 2.0,
            ("a", "p2"): 4.0,
            ("a", "p3"): 0.0,
        },
        orders=(ReferenceOrder("a", origin_index=0, quantity=5.0),),
        lead_time=2,
        initial_arrivals={
            ("a", "p0"): 2.0,
            ("a", "p1"): 3.0,
        },
    )

    rows = {row.period: row for row in trajectory.rows}
    assert tuple(row.arrivals for row in trajectory.rows) == (2.0, 3.0, 5.0, 0.0)
    assert tuple(row.closing for row in trajectory.rows) == (1.0, 2.0, 3.0, 3.0)
    assert tuple(row.on_order for row in trajectory.rows) == (8.0, 5.0, 0.0, 0.0)
    assert trajectory.cost_by_period == {"p0": 1.0, "p1": 2.0, "p2": 3.0, "p3": 3.0}
    assert rows["p2"].opening == 2.0

    mutable_view = cast(dict[str, float], trajectory.cost_by_period)
    with pytest.raises(TypeError):
        mutable_view["p0"] = 99.0


def test_closed_form_reference_refuses_incomplete_shared_inputs() -> None:
    with pytest.raises(ReferenceInputError, match="demand keys mismatch"):
        calculate_reference_trajectory(
            periods=("p0",),
            series=(ReferenceSeries("a", 1.0, 1.0, 1.0),),
            demand={},
            orders=(),
            lead_time=1,
        )

    with pytest.raises(ReferenceInputError, match="initial arrival keys mismatch"):
        calculate_reference_trajectory(
            periods=("p0", "p1", "p2"),
            series=(ReferenceSeries("a", 1.0, 1.0, 1.0),),
            demand={("a", period): 0.0 for period in ("p0", "p1", "p2")},
            orders=(),
            lead_time=2,
            initial_arrivals={("a", "p0"): 1.0},
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


def test_vn2_protocol_defines_no_second_local_settlement_implementation() -> None:
    protocol_root = Path(__file__).parents[2] / "src" / "newcalibre" / "protocols" / "vn2"
    forbidden_modules = {"replay.py", "settlement.py", "simulator.py"}
    assert forbidden_modules.isdisjoint(path.name for path in protocol_root.glob("*.py"))

    forbidden_definitions: list[str] = []
    for path in protocol_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden_definitions.extend(
            f"{path.name}:{node.name}"
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"calculate_reference_trajectory", "settle"}
        )
    assert forbidden_definitions == []


def test_successor_demand_and_reference_sales_have_separately_proven_results() -> None:
    series_key = "stockout"
    calendar = Calendar("W-MON", phase=pd.Timestamp("2026-01-05"))
    periods = tuple(pd.date_range("2026-01-05", periods=2, freq="W-MON"))
    timing = DecisionTiming(lead_time=1, review_period=1)
    session = SessionIdentity.derive(
        tenant="synthetic-censoring",
        series_keys=(series_key,),
        calendar=calendar,
        horizon=timing.protection_period,
        model_config={"backend": "seasonal-naive", "m": 1},
        ordering_policy={"name": "newsvendor"},
        decision_series_keys=(series_key,),
        cost_structure=CostStructure(1.0, 1.0, 0.0, 1.0),
        decision_timing=timing,
        stockout_rule=StockoutRule.LOST_SALES,
    )
    sink = InMemoryLedgerSink(session=session, calendar=calendar)
    demand_scored = settle(
        SettlementRequest(
            session=session,
            snapshot=sink.settlement_snapshot(periods),
            actuals={(series_key, periods[0]): 7, (series_key, periods[1]): 1},
            inventory_positions={series_key: InventoryPosition(4, 0, 0)},
            actuals_semantics=ActualsSemantics.DEMAND,
        )
    )
    literal_demand_expectation = SettlementResult(
        records=(
            SettlementRecord(
                session=session,
                series_key=series_key,
                period=periods[0],
                arrivals=0,
                actuals_semantics=ActualsSemantics.DEMAND,
                transition=StockoutTransition(
                    rule=StockoutRule.LOST_SALES,
                    demand=7,
                    fulfilled_demand=4,
                    unmet_demand=3,
                    closing_on_hand=0,
                    closing_backorders=0,
                ),
                inventory_position=InventoryPosition(0, 0, 0),
                holding=BookedCost(rate=0, basis=0, amount=0),
                shortage=BookedCost(rate=1, basis=3, amount=3),
            ),
            SettlementRecord(
                session=session,
                series_key=series_key,
                period=periods[1],
                arrivals=0,
                actuals_semantics=ActualsSemantics.DEMAND,
                transition=StockoutTransition(
                    rule=StockoutRule.LOST_SALES,
                    demand=1,
                    fulfilled_demand=0,
                    unmet_demand=1,
                    closing_on_hand=0,
                    closing_backorders=0,
                ),
                inventory_position=InventoryPosition(0, 0, 0),
                holding=BookedCost(rate=0, basis=0, amount=0),
                shortage=BookedCost(rate=1, basis=1, amount=1),
            ),
        ),
        inventory_positions={series_key: InventoryPosition(0, 0, 0)},
    )

    assert demand_scored == literal_demand_expectation

    sales_scored = calculate_reference_trajectory(
        periods=("p0", "p1"),
        series=(ReferenceSeries(series_key, 4, 0, 1),),
        demand={(series_key, "p0"): 4, (series_key, "p1"): 0},
        orders=(),
        lead_time=1,
    )
    literal_sales_expectation = (
        ReferenceRow(
            series_key=series_key,
            period="p0",
            opening=4,
            arrivals=0,
            demand=4,
            fulfilled=4,
            closing=0,
            on_order=0,
            shortage=0,
            holding_cost=0,
            shortage_cost=0,
        ),
        ReferenceRow(
            series_key=series_key,
            period="p1",
            opening=0,
            arrivals=0,
            demand=0,
            fulfilled=0,
            closing=0,
            on_order=0,
            shortage=0,
            holding_cost=0,
            shortage_cost=0,
        ),
    )

    assert sales_scored.rows == literal_sales_expectation
    assert sales_scored.cost_by_period == {"p0": 0.0, "p1": 0.0}
    assert sales_scored.total_cost == 0.0

    demand_outcome = tuple(
        (
            record.transition.demand,
            record.transition.fulfilled_demand,
            record.transition.unmet_demand,
            record.inventory_position.on_hand,
            record.realized_cost,
        )
        for record in demand_scored.records
    )
    sales_outcome = tuple(
        (row.demand, row.fulfilled, row.shortage, row.closing, row.total_cost)
        for row in sales_scored.rows
    )
    assert demand_outcome != sales_outcome


def test_deleting_numeric_gate_witness_fails_collection_contract() -> None:
    project_root = Path(__file__).parents[2]
    tier3 = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/tier3/vn2",
            "--collect-only",
            "-q",
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tier3.returncode == 0, tier3.stdout + tier3.stderr

    one_half = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            (
                "tests/tier3/vn2/test_conditional_replay.py::"
                "test_promoted_orders_match_independent_conditional_replay"
            ),
            "--collect-only",
            "-q",
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert one_half.returncode != 0
    assert "vn2-conditional-replay" in one_half.stdout + one_half.stderr

    whole_module = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/tier3/vn2",
            "--ignore=tests/tier3/vn2/test_conditional_replay.py",
            "--collect-only",
            "-q",
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert whole_module.returncode != 0
    assert "required Tier 3 oracle inventory mismatch" in (
        whole_module.stdout + whole_module.stderr
    )

    m5 = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/tier3/m5",
            "--collect-only",
            "-q",
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert m5.returncode == 0, m5.stdout + m5.stderr

    m5_one_half = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            (
                "tests/tier3/m5/test_m5_frozen_scorer_parity.py::"
                "test_successor_and_frozen_m5_scorers_have_exact_count_parity"
            ),
            "--collect-only",
            "-q",
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert m5_one_half.returncode != 0
    assert "m5-frozen-scorer-parity" in m5_one_half.stdout + m5_one_half.stderr

    m5_without_pair = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/tier3/m5",
            "--ignore=tests/tier3/m5/test_m5_frozen_scorer_parity.py",
            "--collect-only",
            "-q",
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert m5_without_pair.returncode != 0
    assert "required Tier 3 oracle inventory mismatch" in (
        m5_without_pair.stdout + m5_without_pair.stderr
    )

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


@pytest.mark.parametrize(
    ("gates", "witnesses", "message"),
    [
        ((), (), "missing_gates"),
        (
            (
                WitnessDeclaration(
                    "tier3",
                    "vn2-conditional-replay",
                    (
                        "tests.tier3.vn2.test_conditional_replay::"
                        "test_promoted_orders_match_independent_conditional_replay"
                    ),
                ),
            ),
            (),
            "missing_witnesses",
        ),
        (
            (),
            (
                WitnessDeclaration(
                    "tier3",
                    "vn2-conditional-replay",
                    (
                        "tests.tier3.vn2.test_conditional_replay::"
                        "test_conditional_replay_rejects_one_successor_order_unit"
                    ),
                ),
            ),
            "missing_gates",
        ),
        (
            (
                WitnessDeclaration(
                    "tier3",
                    "vn2-conditional-replay",
                    "tests.tier3.vn2.renamed_replay::test_promoted_orders",
                ),
            ),
            (
                WitnessDeclaration(
                    "tier3",
                    "vn2-conditional-replay",
                    (
                        "tests.tier3.vn2.test_conditional_replay::"
                        "test_conditional_replay_rejects_one_successor_order_unit"
                    ),
                ),
            ),
            "unexpected_gates",
        ),
    ],
)
def test_required_tier3_inventory_refuses_empty_half_or_renamed_declarations(
    gates: tuple[WitnessDeclaration, ...],
    witnesses: tuple[WitnessDeclaration, ...],
    message: str,
) -> None:
    required_gates = (
        WitnessDeclaration(
            "tier3",
            "vn2-conditional-replay",
            (
                "tests.tier3.vn2.test_conditional_replay::"
                "test_promoted_orders_match_independent_conditional_replay"
            ),
        ),
    )
    required_witnesses = (
        WitnessDeclaration(
            "tier3",
            "vn2-conditional-replay",
            (
                "tests.tier3.vn2.test_conditional_replay::"
                "test_conditional_replay_rejects_one_successor_order_unit"
            ),
        ),
    )

    with pytest.raises(WitnessPairingError, match=message):
        require_exact_inventory(
            gates,
            witnesses,
            required_gates=required_gates,
            required_witnesses=required_witnesses,
        )
