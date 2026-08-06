"""Exercise durable event idempotency across driver reconstruction."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.tier2.driver_scenarios import (
    INITIAL_INVENTORY,
    ORIGINS,
    RUNTIME_CASES,
    actual_records,
    actuals_event,
    build_event_driver,
    drive_origins,
    make_world,
    origin_event,
    run_event_world,
    seed_event_history,
)
from tests.tier2.durable_state import project_durable_state

from newcalibre.domain import Scope
from newcalibre.engine import EventDriverError, OriginEvent

pytestmark = pytest.mark.tier2


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_identical_event_retries_replay_exact_receipts(runtime_name: str | None) -> None:
    """Reconstruct between retries without duplicating any durable domain fact."""
    expected = run_event_world(runtime_name)
    world = make_world(runtime_name, source_actuals=False)
    driver = build_event_driver(world)
    seed_event_history(world, driver)

    origin = origin_event(world, ORIGINS[0], seed=True)
    first_origin = driver.handle(origin)
    origin_state = project_durable_state(world.store)
    origin_revision = world.store.revision
    replayed_origin = build_event_driver(world).handle(origin)
    assert replayed_origin == first_origin
    assert project_durable_state(world.store) == origin_state
    assert world.store.revision == origin_revision

    event = actuals_event(world, actual_records(world, timestamps=(ORIGINS[0],)))
    first_actuals = driver.handle(event)
    actuals_state = project_durable_state(world.store)
    actuals_revision = world.store.revision
    replayed_actuals = build_event_driver(world).handle(event)
    assert replayed_actuals == first_actuals
    assert project_durable_state(world.store) == actuals_state
    assert world.store.revision == actuals_revision

    drive_origins(world, build_event_driver(world), origins=ORIGINS[1:])
    assert project_durable_state(world.store) == project_durable_state(expected.store)


@pytest.mark.parametrize("runtime_name", RUNTIME_CASES)
def test_changed_facts_at_committed_natural_keys_are_rejected(runtime_name: str | None) -> None:
    """Reject changed input facts at a committed origin or actuals natural key."""
    world = make_world(runtime_name, source_actuals=False)
    driver = build_event_driver(world)
    seed_event_history(world, driver)
    origin = origin_event(world, ORIGINS[0], seed=True)
    driver.handle(origin)
    before_origin_conflict = project_durable_state(world.store)

    with pytest.raises(EventDriverError, match="different input facts"):
        driver.handle(
            OriginEvent(
                session=world.session,
                origin=ORIGINS[0],
                scope=Scope.LOCAL,
                initial_inventory_positions=INITIAL_INVENTORY,
            )
        )
    assert project_durable_state(world.store) == before_origin_conflict

    records = actual_records(world, timestamps=(ORIGINS[0],))
    event = actuals_event(world, records)
    driver.handle(event)
    before_actual_conflict = project_durable_state(world.store)
    changed = (
        replace(records[0], recorded_value=float(records[0].recorded_value) + 1.0),
        *records[1:],
    )
    with pytest.raises(EventDriverError, match="different input facts"):
        driver.handle(actuals_event(world, changed))
    assert project_durable_state(world.store) == before_actual_conflict
