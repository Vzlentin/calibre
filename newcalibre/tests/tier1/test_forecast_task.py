"""Exercise indexed forecast-task identity and temporal hygiene."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from newcalibre.domain import (
    KNOWN_AT,
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    Calendar,
    ForecastTask,
    ForecastTaskError,
    HistoryView,
    Panel,
    Scope,
    TargetSupport,
)
from newcalibre.engine import IndexedPanel, IndexedPanelError

pytestmark = pytest.mark.tier1


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(
                ["sku-b", "sku-a", "sku-a", "sku-b", "sku-a", "sku-b"],
                dtype="string",
            ),
            TIMESTAMP: pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-12",
                    "2026-01-12",
                    "2026-01-19",
                    "2026-01-19",
                ]
            ),
            OBSERVED_VALUE: pd.Series([20.0, 10.0, 15.0, 25.0, 30.0, 35.0]),
        }
    )


def _panel(frame: pd.DataFrame | None = None) -> IndexedPanel:
    return IndexedPanel.from_panel(
        Panel.from_frame(
            _frame() if frame is None else frame,
            calendar=Calendar("W-MON"),
            target_support=TargetSupport.NONNEGATIVE,
        )
    )


def _future(*, known_at: str = "2026-01-19") -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku-b", "sku-a"], dtype="string"),
            TIMESTAMP: pd.to_datetime(["2026-01-26", "2026-01-19"]),
            KNOWN_AT: pd.to_datetime([known_at, known_at]),
            "promotion": pd.Series([0, 1], dtype="int64"),
        }
    )


def _tasks(
    *,
    panel: IndexedPanel | None = None,
    origin: str = "2026-01-19",
    scope: Scope = Scope.GLOBAL,
    config: dict[str, object] | None = None,
    future: pd.DataFrame | None = None,
) -> tuple[ForecastTask, ...]:
    return (panel or _panel()).tasks(
        origin=pd.Timestamp(origin),
        horizon=2,
        scope=scope,
        model_config=config or {"backend": "seasonal-naive", "m": 2},
        future_exogenous=future,
    )


def test_task_history_is_an_opaque_strictly_pre_origin_view() -> None:
    task = _tasks()[0]

    assert isinstance(task.history, HistoryView)
    history = task.history.materialize()
    assert history[TIMESTAMP].max() == pd.Timestamp("2026-01-12")
    assert history[TIMESTAMP].lt(task.origin).all()
    assert not isinstance(task.history, pd.DataFrame)


def test_scope_is_resolved_once_into_deterministic_tasks() -> None:
    global_tasks = _tasks(scope=Scope.GLOBAL)
    local_tasks = _tasks(scope=Scope.LOCAL)

    assert [task.series_keys for task in global_tasks] == [("sku-a", "sku-b")]
    assert [task.series_keys for task in local_tasks] == [("sku-a",), ("sku-b",)]
    assert all(task.scope is Scope.LOCAL for task in local_tasks)


def test_task_identity_is_stable_under_source_row_permutation() -> None:
    first = _tasks()[0]
    permuted = _frame().iloc[[1, 0, 3, 2, 5, 4]].reset_index(drop=True)
    second = _tasks(panel=_panel(permuted))[0]

    assert first.identity == second.identity
    assert first.cursor == second.cursor


def test_task_identity_binds_scope_cursor_and_configuration() -> None:
    baseline = _tasks()[0]
    local = _tasks(scope=Scope.LOCAL)[0]
    later = _tasks(origin="2026-01-26")[0]
    configured = _tasks(config={"backend": "seasonal-naive", "m": 1})[0]

    assert len({baseline.identity, local.identity, later.identity, configured.identity}) == 4


def test_future_exogenous_is_canonical_and_defensively_owned() -> None:
    source = _future()
    task = _tasks(future=source)[0]
    source.loc[:, "promotion"] = 9
    returned = task.future_exogenous
    assert returned is not None
    returned.loc[:, "promotion"] = 8

    restored = task.future_exogenous
    assert restored is not None
    assert restored[SERIES_KEY].tolist() == ["sku-a", "sku-b"]
    assert restored["promotion"].tolist() == [1, 0]


@pytest.mark.parametrize(
    ("future", "pattern"),
    [
        (_future(known_at="2026-01-20"), "known at or before"),
        (_future().assign(timestamp=pd.to_datetime(["2026-02-02", "2026-01-19"])), "horizon"),
        (_future().assign(series_key=pd.Series(["unknown", "sku-a"], dtype="string")), "unknown"),
    ],
)
def test_future_exogenous_rejects_temporal_or_series_leakage(
    future: pd.DataFrame,
    pattern: str,
) -> None:
    with pytest.raises((ForecastTaskError, IndexedPanelError), match=pattern):
        _tasks(future=future)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_chunk_size_requires_a_positive_integer(chunk_size: object) -> None:
    with pytest.raises(IndexedPanelError, match="positive integer"):
        _panel().tasks(
            origin=pd.Timestamp("2026-01-19"),
            horizon=2,
            scope=Scope.LOCAL,
            series_chunk_size=chunk_size,  # type: ignore[arg-type]
            model_config={"backend": "seasonal-naive", "m": 2},
        )


def test_model_configuration_is_canonical_and_scope_remains_engine_owned() -> None:
    task = _tasks(config={"m": 2, "backend": "seasonal-naive"})[0]
    returned = task.model_config
    assert returned == {"backend": "seasonal-naive", "m": 2}
    returned["m"] = 3  # type: ignore[index]
    assert task.model_config == {"backend": "seasonal-naive", "m": 2}
    with pytest.raises(ForecastTaskError, match="scope is engine configuration"):
        _tasks(config={"backend": "seasonal-naive", "m": 2, "scope": "global"})


def test_old_copied_history_transport_surface_is_absent() -> None:
    source = inspect.getsource(ForecastTask)

    assert "to_bytes" not in ForecastTask.__dict__
    assert "from_bytes" not in ForecastTask.__dict__
    assert "pd.DataFrame" not in inspect.signature(ForecastTask.history.fget).return_annotation
    assert "_serialized" not in source
    assert "forecast_tasks" not in Panel.__dict__
