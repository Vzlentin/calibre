"""Exercise the engine-owned checkpointed forecasting lifecycle."""

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
    InMemoryArtifactStore,
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


def _task(
    panel: IndexedPanel,
    config: dict[str, object],
    origin: str,
    *,
    previous=None,
):
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
) -> pd.DataFrame:
    result = lifecycle.run_item((session, task, token))
    lifecycle.publish((result,))
    return result.frame


def test_first_fit_exact_load_and_incremental_resume_publish_deterministic_checkpoints() -> None:
    panel, config, session = _world()
    store = InMemoryArtifactStore()
    first = _task(panel, config, "2026-01-15")
    first_token = CycleToken(session, first.origin, 1)
    lifecycle = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    first_frame = _run(lifecycle, session=session, task=first, token=first_token)
    first_artifacts = dict(store.artifacts)

    exact = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    exact_frame = _run(
        exact,
        session=session,
        task=first,
        token=CycleToken(session, first.origin, 2),
    )

    second = _task(panel, config, "2026-01-16", previous={first.series_keys: first.cursor})
    resumed = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    second_token = CycleToken(session, second.origin, 3)
    second_frame = _run(resumed, session=session, task=second, token=second_token)

    pd.testing.assert_frame_equal(first_frame, exact_frame)
    assert dict(store.artifacts).items() >= first_artifacts.items()
    assert len(store.artifacts) == 2
    assert len(store.artifact_indexes) == 1
    assert second_frame["point_forecast"].tolist() == [9.0, 10.0]


def test_incremental_resume_loads_one_index_and_one_prior_checkpoint() -> None:
    class CountingStore(InMemoryArtifactStore):
        def __init__(self) -> None:
            super().__init__()
            self.artifact_loads = 0
            self.index_loads = 0

        def load(self, key: str) -> bytes | None:
            self.artifact_loads += 1
            return super().load(key)

        def load_index(self, key: str) -> bytes | None:
            self.index_loads += 1
            return super().load_index(key)

    panel, config, session = _world()
    store = CountingStore()
    first = _task(panel, config, "2026-01-08")
    first_token = CycleToken(session, first.origin, 1)
    lifecycle = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    _run(lifecycle, session=session, task=first, token=first_token)

    store.artifact_loads = 0
    store.index_loads = 0
    later = _task(panel, config, "2026-01-16", previous={first.series_keys: first.cursor})
    later_token = CycleToken(session, later.origin, 2)
    resumed = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    resumed.run_item((session, later, later_token))

    assert store.index_loads == 1
    assert store.artifact_loads == 2  # exact miss plus the indexed predecessor


def test_malformed_exact_checkpoint_fails_closed() -> None:
    panel, config, session = _world()
    store = InMemoryArtifactStore()
    task = _task(panel, config, "2026-01-15")
    token = CycleToken(session, task.origin, 1)
    lifecycle = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    _run(lifecycle, session=session, task=task, token=token)
    key = next(iter(store._artifacts))
    store._artifacts[key] = b"malformed"

    retry = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    with pytest.raises(ForecastLifecycleError, match="malformed"):
        retry.run_item(
            (session, task, CycleToken(session, task.origin, 2)),
        )


def test_run_requires_the_exact_task_cycle() -> None:
    panel, config, session = _world()
    task = _task(panel, config, "2026-01-15")
    lifecycle = ForecastLifecycle(
        artifact_store=InMemoryArtifactStore(),
        adapter_resolver=resolve_adapter,
    )
    with pytest.raises(ForecastLifecycleError, match="does not match"):
        lifecycle.run_item(
            (
                session,
                task,
                CycleToken(session, task.calendar.advance(task.origin, 1), 1),
            ),
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
    panel, config, session = _world()
    store = InMemoryArtifactStore()
    task = _task(panel, config, "2026-01-15")
    token = CycleToken(session, task.origin, 1)
    lifecycle = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    _run(lifecycle, session=session, task=task, token=token)
    key = next(iter(store._artifacts))
    payload = json.loads(store._artifacts[key])
    payload[field] = foreign
    store._artifacts[key] = canonical_json_bytes(payload, path="foreign checkpoint")

    retry = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    with pytest.raises(ForecastLifecycleError):
        retry.run_item(
            (session, task, CycleToken(session, task.origin, 2)),
        )


def test_well_formed_foreign_checkpoint_index_target_fails_closed() -> None:
    panel, config, session = _world()
    store = InMemoryArtifactStore()
    first = _task(panel, config, "2026-01-08")
    lifecycle = ForecastLifecycle(artifact_store=store, adapter_resolver=resolve_adapter)
    token = CycleToken(session, first.origin, 1)
    _run(lifecycle, session=session, task=first, token=token)
    index_key = next(iter(store._artifact_indexes))
    index = json.loads(store._artifact_indexes[index_key])
    index["checkpoint_key"] = "foreign"
    store._artifact_indexes[index_key] = canonical_json_bytes(
        index,
        path="foreign checkpoint index",
    )
    later = _task(panel, config, "2026-01-16", previous={first.series_keys: first.cursor})

    with pytest.raises(ForecastLifecycleError, match="invalid artifact"):
        ForecastLifecycle(
            artifact_store=store,
            adapter_resolver=resolve_adapter,
        ).run_item(
            (session, later, CycleToken(session, later.origin, 2)),
        )


def test_declared_update_failure_propagates_without_fit_or_publication() -> None:
    calls: list[str] = []

    class FailingUpdateAdapter(SeasonalNaiveAdapter):
        def fit(self, task) -> None:
            calls.append("fit")
            super().fit(task)

        def update(self, delta) -> None:
            calls.append("update")
            raise RuntimeError("declared update failed")

    panel, config, session = _world()
    store = InMemoryArtifactStore()
    first = _task(panel, config, "2026-01-08")
    lifecycle = ForecastLifecycle(
        artifact_store=store,
        adapter_resolver=lambda model_config: FailingUpdateAdapter(model_config),
    )
    first_result = lifecycle.run_item((session, first, CycleToken(session, first.origin, 1)))
    lifecycle.publish((first_result,))
    published = (dict(store.artifacts), dict(store.artifact_indexes))
    calls.clear()
    later = _task(panel, config, "2026-01-16", previous={first.series_keys: first.cursor})

    with pytest.raises(RuntimeError, match="declared update failed"):
        lifecycle.run_item((session, later, CycleToken(session, later.origin, 2)))

    assert calls == ["update"]
    assert (dict(store.artifacts), dict(store.artifact_indexes)) == published
    assert not hasattr(lifecycle, "_prepared")
