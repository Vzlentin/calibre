"""Prove restart equivalence at origin and actuals transaction boundaries."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from tests.tier2.driver_scenarios import (
    CALENDAR,
    ORIGINS,
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

from newcalibre.domain import ActualsSemantics, Panel, SessionIdentity
from newcalibre.engine import (
    ActualsCommit,
    InMemoryIndexedRunStore,
    OriginCommit,
    PhaseError,
)

pytestmark = pytest.mark.tier2

type _Write = OriginCommit | ActualsCommit


class _InterruptingRunStore(InMemoryIndexedRunStore):
    """Fail once immediately before or after a selected atomic publication."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        actuals: Panel,
        selected: Callable[[_Write], bool],
        fail_after_commit: bool,
    ) -> None:
        super().__init__(
            session=session,
            calendar=CALENDAR,
            actuals=actuals,
            actuals_semantics=ActualsSemantics.DEMAND,
        )
        self._selected = selected
        self._fail_after_commit = fail_after_commit
        self._failed = False

    def commit(self, write: _Write):
        """Interrupt the selected write once while retaining real store semantics."""
        if self._failed or not self._selected(write):
            return super().commit(write)
        self._failed = True
        if not self._fail_after_commit:
            raise RuntimeError("interrupted before transactional publication")
        super().commit(write)
        raise RuntimeError("lost transactional commit response")


@pytest.mark.parametrize("fail_after_commit", [False, True], ids=["before", "after"])
def test_origin_commit_restart_matches_uninterrupted_run(fail_after_commit: bool) -> None:
    """Replay an origin naturally after either invisible failure or lost response."""
    expected = run_event_world(None)
    target = ORIGINS[2]
    world = make_world(
        None,
        store_factory=lambda session, panel: _InterruptingRunStore(
            session=session,
            actuals=panel,
            selected=lambda write: isinstance(write, OriginCommit) and write.origin == target,
            fail_after_commit=fail_after_commit,
        ),
    )
    driver = build_event_driver(world)
    seed_event_history(world, driver)
    drive_origins(world, driver, origins=ORIGINS[:2])
    before = project_durable_state(world.store)

    with pytest.raises(PhaseError, match="transactional"):
        driver.handle(origin_event(world, target))
    if fail_after_commit:
        assert world.store.receipt(target) is not None
        assert project_durable_state(world.store) != before
    else:
        assert world.store.receipt(target) is None
        assert project_durable_state(world.store) == before

    resumed = build_event_driver(world)
    resumed.handle(origin_event(world, target))
    resumed.handle(actuals_event(world, actual_records(world, timestamps=(target,))))
    drive_origins(world, resumed, origins=ORIGINS[3:])

    assert project_durable_state(world.store) == project_durable_state(expected.store)


@pytest.mark.parametrize("fail_after_commit", [False, True], ids=["before", "after"])
def test_actuals_commit_restart_matches_uninterrupted_run(fail_after_commit: bool) -> None:
    """Replay an actuals natural key and deliver each affected forecast only once."""
    expected = run_event_world(None)
    target = ORIGINS[2]
    target_keys = tuple(record.key for record in actual_records(expected, timestamps=(target,)))
    world = make_world(
        None,
        store_factory=lambda session, panel: _InterruptingRunStore(
            session=session,
            actuals=panel,
            selected=lambda write: (
                isinstance(write, ActualsCommit) and write.actual_keys == target_keys
            ),
            fail_after_commit=fail_after_commit,
        ),
    )
    driver = build_event_driver(world)
    seed_event_history(world, driver)
    drive_origins(world, driver, origins=ORIGINS[:2])
    driver.handle(origin_event(world, target))
    event = actuals_event(world, actual_records(world, timestamps=(target,)))
    before = project_durable_state(world.store)

    with pytest.raises(RuntimeError, match="transactional"):
        driver.handle(event)
    if fail_after_commit:
        assert project_durable_state(world.store) != before
    else:
        assert project_durable_state(world.store) == before

    resumed = build_event_driver(world)
    resumed.handle(event)
    delivered = tuple(world.store.observation_resolutions)
    resumed.handle(event)
    assert world.store.observation_resolutions == delivered
    drive_origins(world, resumed, origins=ORIGINS[3:])

    assert project_durable_state(world.store) == project_durable_state(expected.store)
