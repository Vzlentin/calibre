"""Exercise the engine-owned revision-snapshot forecasting lifecycle."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    Calendar,
    CycleToken,
    Panel,
    Scope,
    SessionIdentity,
    TargetSupport,
)
from newcalibre.domain._canonical_json import canonical_json_bytes
from newcalibre.engine import (
    ForecastLifecycle,
    ForecastLifecycleError,
    IndexedPanel,
    InProcessDispatch,
)
from newcalibre.forecasting import SeasonalNaiveAdapter, resolve_adapter

pytestmark = pytest.mark.tier1


def _world():
    history = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku"] * 16, dtype="string"),
            TIMESTAMP: pd.date_range("2026-01-01", periods=16, freq="D"),
            OBSERVED_VALUE: pd.Series(range(1, 17), dtype="float64"),
        }
    )
    panel = IndexedPanel.from_panel(
        Panel.from_frame(
            history,
            calendar=Calendar("D"),
            target_support=TargetSupport.NONNEGATIVE,
        )
    )
    config = {"backend": "seasonal-naive", "m": 7}
    session = SessionIdentity.derive(
        tenant="test",
        series_keys=panel.series_keys,
        calendar=panel.calendar,
        horizon=2,
        model_config=config,
    )
    return panel, config, session


def _task(panel: IndexedPanel, config: dict[str, object], origin: str, *, previous=None):
    return panel.tasks(
        origin=pd.Timestamp(origin),
        horizon=2,
        scope=Scope.GLOBAL,
        model_config=config,
        previous_cursors=previous,
    )[0]


def _run(
    lifecycle: ForecastLifecycle,
    *,
    session: SessionIdentity,
    task,
    token: CycleToken,
    checkpoints: dict[str, bytes],
    indexes: dict[str, bytes],
) -> pd.DataFrame:
    result = _execute(
        lifecycle,
        session=session,
        task=task,
        token=token,
        checkpoints=checkpoints,
        indexes=indexes,
    )
    staged_checkpoints, staged_indexes = lifecycle.staged_updates((result,))
    checkpoints.update(staged_checkpoints)
    indexes.update(staged_indexes)
    return result.frame


def _execute(
    lifecycle: ForecastLifecycle,
    *,
    session: SessionIdentity,
    task,
    token: CycleToken,
    checkpoints: dict[str, bytes],
    indexes: dict[str, bytes],
):
    dispatch = InProcessDispatch()
    work = lifecycle.prepare_work(
        session=session,
        task=task,
        token=token,
        checkpoints=checkpoints,
        checkpoint_indexes=indexes,
        backend=dispatch.backend,
        budget=dispatch.budget,
    )
    return lifecycle.complete_work(work, dispatch.dispatch(work, lifecycle))


def test_first_fit_exact_load_and_incremental_resume_stage_deterministic_checkpoints() -> None:
    """Restore exact and predecessor state from one revision-bound mapping pair."""
    panel, config, session = _world()
    checkpoints: dict[str, bytes] = {}
    indexes: dict[str, bytes] = {}
    first = _task(panel, config, "2026-01-15")
    lifecycle = ForecastLifecycle(adapter_resolver=resolve_adapter)
    first_frame = _run(
        lifecycle,
        session=session,
        task=first,
        token=CycleToken(session, first.origin, 1, 1, "0" * 32),
        checkpoints=checkpoints,
        indexes=indexes,
    )
    first_checkpoints = dict(checkpoints)

    exact_frame = _run(
        ForecastLifecycle(adapter_resolver=resolve_adapter),
        session=session,
        task=first,
        token=CycleToken(session, first.origin, 2, 1, "0" * 32),
        checkpoints=checkpoints,
        indexes=indexes,
    )
    second = _task(panel, config, "2026-01-16", previous={first.series_keys: first.cursor})
    second_frame = _run(
        ForecastLifecycle(adapter_resolver=resolve_adapter),
        session=session,
        task=second,
        token=CycleToken(session, second.origin, 3, 1, "0" * 32),
        checkpoints=checkpoints,
        indexes=indexes,
    )

    pd.testing.assert_frame_equal(first_frame, exact_frame)
    assert checkpoints.items() >= first_checkpoints.items()
    assert len(checkpoints) == 2
    assert len(indexes) == 1
    assert second_frame["point_forecast"].tolist() == [9.0, 10.0]


def test_staged_updates_remain_unpublished_until_the_caller_commits() -> None:
    """Return checkpoint effects without mutating snapshot mappings."""
    panel, config, session = _world()
    task = _task(panel, config, "2026-01-15")
    lifecycle = ForecastLifecycle(adapter_resolver=resolve_adapter)
    result = _execute(
        lifecycle,
        session=session,
        task=task,
        token=CycleToken(session, task.origin, 1, 1, "0" * 32),
        checkpoints={},
        indexes={},
    )

    checkpoints, indexes = lifecycle.staged_updates((result,))

    assert checkpoints
    assert indexes
    assert not hasattr(lifecycle, "publish")


def test_malformed_exact_checkpoint_fails_closed() -> None:
    """Reject malformed opaque state before adapter prediction."""
    panel, config, session = _world()
    checkpoints: dict[str, bytes] = {}
    indexes: dict[str, bytes] = {}
    task = _task(panel, config, "2026-01-15")
    lifecycle = ForecastLifecycle(adapter_resolver=resolve_adapter)
    _run(
        lifecycle,
        session=session,
        task=task,
        token=CycleToken(session, task.origin, 1, 1, "0" * 32),
        checkpoints=checkpoints,
        indexes=indexes,
    )
    checkpoints[next(iter(checkpoints))] = b"malformed"

    with pytest.raises(ForecastLifecycleError, match="malformed"):
        _execute(
            ForecastLifecycle(adapter_resolver=resolve_adapter),
            session=session,
            task=task,
            token=CycleToken(session, task.origin, 2, 1, "0" * 32),
            checkpoints=checkpoints,
            indexes=indexes,
        )


def test_run_requires_the_exact_task_cycle() -> None:
    """Bind dispatched work to its opened store revision token."""
    panel, config, session = _world()
    task = _task(panel, config, "2026-01-15")
    with pytest.raises(ForecastLifecycleError, match="does not match"):
        _execute(
            ForecastLifecycle(adapter_resolver=resolve_adapter),
            session=session,
            task=task,
            token=CycleToken(
                session,
                task.calendar.advance(task.origin, 1),
                1,
                1,
                "0" * 32,
            ),
            checkpoints={},
            indexes={},
        )


@pytest.mark.parametrize(
    ("field", "foreign"),
    [
        ("session", "0" * 64),
        ("task_identity", "0" * 64),
        ("lineage_identity", "0" * 64),
        ("config_digest", "0" * 64),
        ("capabilities", ["artifact_persistence"]),
        (
            "cursor",
            {
                "panel_identity": "0" * 64,
                "series_start": 0,
                "series_stop": 1,
                "time_bound": 14,
            },
        ),
    ],
)
def test_well_formed_foreign_checkpoint_bindings_fail_closed(
    field: str,
    foreign: object,
) -> None:
    """Reject checkpoint bytes bound to different task or lineage facts."""
    panel, config, session = _world()
    checkpoints: dict[str, bytes] = {}
    indexes: dict[str, bytes] = {}
    task = _task(panel, config, "2026-01-15")
    _run(
        ForecastLifecycle(adapter_resolver=resolve_adapter),
        session=session,
        task=task,
        token=CycleToken(session, task.origin, 1, 1, "0" * 32),
        checkpoints=checkpoints,
        indexes=indexes,
    )
    key = next(iter(checkpoints))
    payload = json.loads(checkpoints[key])
    payload[field] = foreign
    checkpoints[key] = canonical_json_bytes(payload, path="foreign checkpoint")

    with pytest.raises(ForecastLifecycleError):
        _execute(
            ForecastLifecycle(adapter_resolver=resolve_adapter),
            session=session,
            task=task,
            token=CycleToken(session, task.origin, 2, 1, "0" * 32),
            checkpoints=checkpoints,
            indexes=indexes,
        )


def test_well_formed_foreign_checkpoint_index_target_fails_closed() -> None:
    """Reject an index that points outside its derived checkpoint lineage."""
    panel, config, session = _world()
    checkpoints: dict[str, bytes] = {}
    indexes: dict[str, bytes] = {}
    first = _task(panel, config, "2026-01-08")
    _run(
        ForecastLifecycle(adapter_resolver=resolve_adapter),
        session=session,
        task=first,
        token=CycleToken(session, first.origin, 1, 1, "0" * 32),
        checkpoints=checkpoints,
        indexes=indexes,
    )
    index_key = next(iter(indexes))
    index = json.loads(indexes[index_key])
    index["checkpoint_key"] = "foreign"
    indexes[index_key] = canonical_json_bytes(index, path="foreign checkpoint index")
    later = _task(panel, config, "2026-01-16", previous={first.series_keys: first.cursor})

    with pytest.raises(ForecastLifecycleError, match="invalid artifact"):
        _execute(
            ForecastLifecycle(adapter_resolver=resolve_adapter),
            session=session,
            task=later,
            token=CycleToken(session, later.origin, 2, 1, "0" * 32),
            checkpoints=checkpoints,
            indexes=indexes,
        )


def test_declared_update_failure_leaves_snapshot_mappings_unchanged() -> None:
    """Propagate adapter failures without publishing staged checkpoint effects."""
    calls: list[str] = []

    class FailingUpdateAdapter(SeasonalNaiveAdapter):
        def fit(self, task) -> None:
            calls.append("fit")
            super().fit(task)

        def update(self, delta) -> None:
            calls.append("update")
            raise RuntimeError("declared update failed")

    panel, config, session = _world()
    checkpoints: dict[str, bytes] = {}
    indexes: dict[str, bytes] = {}
    first = _task(panel, config, "2026-01-08")
    lifecycle = ForecastLifecycle(
        adapter_resolver=lambda model_config: FailingUpdateAdapter(model_config)
    )
    _run(
        lifecycle,
        session=session,
        task=first,
        token=CycleToken(session, first.origin, 1, 1, "0" * 32),
        checkpoints=checkpoints,
        indexes=indexes,
    )
    published = (dict(checkpoints), dict(indexes))
    calls.clear()
    later = _task(panel, config, "2026-01-16", previous={first.series_keys: first.cursor})

    with pytest.raises(RuntimeError, match="declared update failed"):
        _execute(
            lifecycle,
            session=session,
            task=later,
            token=CycleToken(session, later.origin, 2, 1, "0" * 32),
            checkpoints=checkpoints,
            indexes=indexes,
        )

    assert calls == ["update"]
    assert (checkpoints, indexes) == published
