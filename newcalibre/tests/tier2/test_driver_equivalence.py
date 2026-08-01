"""Prove time-loop and event-frontier equality over one transactional store."""

from __future__ import annotations

import math

import pytest
from tests.tier2.driver_scenarios import (
    CALENDAR,
    EXPECTED_BOOKED_COSTS,
    EXPECTED_FINAL_INVENTORY,
    ORIGINS,
    RUNTIME_CASES,
    build_event_driver,
    drive_origins,
    make_world,
    run_event_world,
    run_time_world,
    runtime_witness,
    seed_event_history,
)
from tests.tier2.durable_state import project_durable_state

pytestmark = pytest.mark.tier2


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_time_loop_and_event_frontier_publish_identical_domain_state(
    runtime_name: str | None,
) -> None:
    """Compare every durable domain family while excluding driver commit grouping."""
    time_world = run_time_world(runtime_name)
    event_world = run_event_world(runtime_name)

    assert project_durable_state(
        event_world.store,
        include_journal=False,
    ) == project_durable_state(time_world.store, include_journal=False)
    assert len(time_world.store.forecasts) == 48
    assert len(time_world.store.orders) == 8
    assert len(time_world.store.settlements) == 16
    assert len(time_world.store.observed_history) == 24
    assert time_world.store.revision < event_world.store.revision


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_event_actual_order_and_matching_rechunk_schedules_are_deterministic(
    runtime_name: str | None,
) -> None:
    """Keep actual transaction boundaries outside the logical state projection."""
    canonical = run_event_world(runtime_name)
    reversed_batch = run_event_world(runtime_name, reverse=True)
    rechunked = run_event_world(runtime_name, rechunk=True)
    repeated_rechunk = run_event_world(runtime_name, rechunk=True, reverse=True)

    assert project_durable_state(
        reversed_batch.store,
        include_journal=False,
    ) == project_durable_state(canonical.store, include_journal=False)
    assert project_durable_state(
        repeated_rechunk.store,
        include_journal=False,
    ) == project_durable_state(rechunked.store, include_journal=False)
    assert rechunked.store.revision > canonical.store.revision


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_reconstructed_event_driver_continues_the_same_store(runtime_name: str | None) -> None:
    """Rebuild the process-local engine halfway through the event schedule."""
    expected = run_event_world(runtime_name)
    resumed = make_world(runtime_name)
    first = build_event_driver(resumed)
    seed_event_history(resumed, first)
    drive_origins(resumed, first, origins=ORIGINS[:4])

    second = build_event_driver(resumed)
    drive_origins(resumed, second, origins=ORIGINS[4:])

    assert project_durable_state(resumed.store) == project_durable_state(expected.store)


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES[1:])
def test_conformal_runtime_leaves_positive_durable_witnesses(runtime_name: str) -> None:
    """Show that equivalence exercises issued bounds, delivery, and dirty state rows."""
    world = run_event_world(runtime_name)
    witness = runtime_witness(world)

    assert witness.delivered_scores
    assert witness.finite_issuances
    assert witness.state_after


def test_decision_run_finishes_with_expected_inventory_and_costs() -> None:
    """Retain the hand-checked settlement witness across the store cutover."""
    world = run_time_world(None)
    probe = CALENDAR.advance(max(row.period for row in world.store.settlements), 1)
    snapshot = world.store.settlement_snapshot((probe,))
    holding = math.fsum(row.holding.amount for row in world.store.settlements)
    shortage = math.fsum(row.shortage.amount for row in world.store.settlements)

    assert dict(snapshot.current_positions) == dict(EXPECTED_FINAL_INVENTORY)
    assert (holding, shortage, math.fsum((holding, shortage))) == EXPECTED_BOOKED_COSTS
