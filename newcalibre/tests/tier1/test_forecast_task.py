"""Exercise panel-owned task construction and exact task transport."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pytest

from newcalibre.domain import (
    AVAILABILITY_BOUND,
    CENSOR_STATUS,
    KNOWN_AT,
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    UNDECLARED_CENSORING,
    Calendar,
    CensoringAssertion,
    ForecastTask,
    ForecastTaskError,
    Panel,
    PanelError,
    Scope,
)

pytestmark = pytest.mark.tier1


def _panel_frame(*, resolution: str = "us") -> pd.DataFrame:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2026-01-05",
                "2026-01-05",
                "2026-01-12",
                "2026-01-12",
                "2026-01-19",
                "2026-01-19",
            ]
        ).astype(f"datetime64[{resolution}]")
    )
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(
                ["sku-a", "sku-b", "sku-a", "sku-b", "sku-b", "sku-a"],
                dtype="string",
            ),
            TIMESTAMP: timestamps,
            OBSERVED_VALUE: pd.Series([10, 20, np.nan, 25, 30, 15], dtype="float64"),
            CENSOR_STATUS: pd.Series(
                ["uncensored", pd.NA, "censored", "uncensored", pd.NA, "censored"],
                dtype="string",
            ),
            AVAILABILITY_BOUND: pd.Series([10, 20, 9, 25, 30, 14], dtype="int64"),
            "planned_price": pd.Series([1, 1, 2, 2, 3, 3], dtype="int32"),
        }
    )


def _panel(*, resolution: str = "us") -> Panel:
    return Panel.from_frame(_panel_frame(resolution=resolution), calendar=Calendar("W-MON"))


def _tasks(
    *,
    scope: Scope = Scope.GLOBAL,
    origin: str = "2026-01-19",
    horizon: int = 2,
    config: dict[str, object] | None = None,
    future: pd.DataFrame | None = None,
    resolution: str = "us",
) -> tuple[ForecastTask, ...]:
    return _panel(resolution=resolution).forecast_tasks(
        origin=pd.Timestamp(origin),
        horizon=horizon,
        scope=scope,
        model_config=config or {"backend": "seasonal-naive", "m": 2},
        future_exogenous=future,
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


def test_panel_canonicalizes_rows_columns_and_numeric_values_without_fabricating_missing() -> None:
    panel = _panel()

    assert panel.series_keys == ("sku-a", "sku-b")
    assert panel.frame.columns.tolist() == [
        SERIES_KEY,
        TIMESTAMP,
        OBSERVED_VALUE,
        CENSOR_STATUS,
        AVAILABILITY_BOUND,
        "planned_price",
    ]
    assert panel.frame[[SERIES_KEY, TIMESTAMP]].values.tolist() == [
        ["sku-a", pd.Timestamp("2026-01-05")],
        ["sku-a", pd.Timestamp("2026-01-12")],
        ["sku-a", pd.Timestamp("2026-01-19")],
        ["sku-b", pd.Timestamp("2026-01-05")],
        ["sku-b", pd.Timestamp("2026-01-12")],
        ["sku-b", pd.Timestamp("2026-01-19")],
    ]
    assert panel.frame[OBSERVED_VALUE].isna().sum() == 1
    assert panel.frame[AVAILABILITY_BOUND].dtype == np.dtype("int64")
    assert panel.frame["planned_price"].dtype == np.dtype("int32")
    assert panel.frame[CENSOR_STATUS].tolist().count(UNDECLARED_CENSORING) == 2
    assert panel.frame[CENSOR_STATUS].dtype.storage == "pyarrow"


def test_valid_interleavings_produce_the_same_canonical_snapshot() -> None:
    frame = _panel_frame()
    first = Panel.from_frame(frame, calendar=Calendar("W-MON"))
    second = Panel.from_frame(
        frame.iloc[[0, 2, 1, 3, 5, 4]][list(reversed(frame.columns))],
        calendar=Calendar("W-MON"),
    )

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.series_keys == second.series_keys


def test_panel_rejects_unique_out_of_order_timestamps_within_a_series() -> None:
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(
                ["sku-a", "sku-b", "sku-a", "sku-b", "sku-b", "sku-a"],
                dtype="string",
            ),
            TIMESTAMP: pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-12",
                    "2026-01-19",
                    "2026-01-12",
                    "2026-01-19",
                ]
            ),
            OBSERVED_VALUE: pd.Series([1.0, 10.0, 2.0, 30.0, 20.0, 3.0]),
        }
    )

    with pytest.raises(PanelError, match="strictly increasing"):
        Panel.from_frame(frame, calendar=Calendar("W-MON"))


def test_panel_accepts_an_increasing_series_with_a_calendar_gap() -> None:
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku", "sku"], dtype="string"),
            TIMESTAMP: pd.to_datetime(["2026-01-05", "2026-01-19"]),
            OBSERVED_VALUE: pd.Series([1.0, 3.0]),
        }
    )

    panel = Panel.from_frame(frame, calendar=Calendar("W-MON"))

    pd.testing.assert_frame_equal(panel.frame, frame)


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        (lambda frame: frame.drop(columns=OBSERVED_VALUE), "missing required"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate"),
        (lambda frame: frame.assign(series_key=frame[SERIES_KEY].astype("object")), "string dtype"),
        (lambda frame: frame.assign(value=True), "native NumPy"),
        (lambda frame: frame.assign(note="not numeric"), "exogenous.*native NumPy"),
        (
            lambda frame: frame.assign(
                censor_status=pd.Series(["unknown"] * len(frame), dtype="string")
            ),
            "invalid values",
        ),
        (
            lambda frame: frame.assign(timestamp=frame[TIMESTAMP] + pd.Timedelta(days=1)),
            "does not lie on calendar",
        ),
    ],
)
def test_panel_rejects_invalid_schema_and_rows(
    mutation: Callable[[pd.DataFrame], pd.DataFrame], pattern: str
) -> None:
    with pytest.raises(PanelError, match=pattern):
        Panel.from_frame(mutation(_panel_frame()), calendar=Calendar("W-MON"))


@pytest.mark.parametrize("key", ["", None])
def test_panel_rejects_empty_or_missing_opaque_series_keys(key: object) -> None:
    frame = _panel_frame()
    frame.loc[0, SERIES_KEY] = cast(str, key)
    with pytest.raises(PanelError, match="series keys"):
        Panel.from_frame(frame, calendar=Calendar("W-MON"))


def test_panel_uses_utf8_byte_order_for_exact_opaque_keys() -> None:
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["é", "z"], dtype="string"),
            TIMESTAMP: pd.to_datetime(["2026-01-05", "2026-01-05"]),
            OBSERVED_VALUE: [1.0, 2.0],
        }
    )
    assert Panel.from_frame(frame, calendar=Calendar("W-MON")).series_keys == ("z", "é")


def test_panel_preserves_optional_censor_surface_and_records_undeclared_facts() -> None:
    no_facts = Panel.from_frame(
        _panel_frame().drop(columns=[CENSOR_STATUS, AVAILABILITY_BOUND]),
        calendar=Calendar("W-MON"),
    )
    bound_only = Panel.from_frame(
        _panel_frame().drop(columns=CENSOR_STATUS),
        calendar=Calendar("W-MON"),
    )

    assert not no_facts.has_censoring_facts
    assert CENSOR_STATUS not in no_facts.frame
    assert bound_only.has_censoring_facts
    assert bound_only.frame[CENSOR_STATUS].eq(UNDECLARED_CENSORING).all()
    assert set(CensoringAssertion) == {
        CensoringAssertion.CENSORED,
        CensoringAssertion.UNCENSORED,
    }
    reingested = Panel.from_frame(bound_only.frame, calendar=bound_only.calendar)
    pd.testing.assert_frame_equal(reingested.frame, bound_only.frame)


def test_task_partition_filters_all_history_at_or_after_origin_before_adapter_visibility() -> None:
    task = _tasks()[0]

    assert task.scope is Scope.GLOBAL
    assert task.series_keys == ("sku-a", "sku-b")
    assert task.history[TIMESTAMP].max() == pd.Timestamp("2026-01-12")
    assert task.history[TIMESTAMP].lt(task.origin).all()


def test_local_scope_yields_one_fixed_one_series_task_per_key() -> None:
    tasks = _tasks(scope=Scope.LOCAL)

    assert tuple(task.series_keys for task in tasks) == (("sku-a",), ("sku-b",))
    assert all(task.scope is Scope.LOCAL for task in tasks)
    assert all(set(task.history[SERIES_KEY]) == set(task.series_keys) for task in tasks)


def test_global_scope_yields_one_whole_panel_task() -> None:
    tasks = _tasks(scope=Scope.GLOBAL)
    assert len(tasks) == 1
    assert tasks[0].series_keys == _panel().series_keys


@pytest.mark.parametrize("scope", ["per-series", 7, None])
def test_partition_rejects_unknown_scope(scope: object) -> None:
    with pytest.raises(PanelError, match="scope"):
        _panel().forecast_tasks(
            origin=pd.Timestamp("2026-01-19"),
            horizon=2,
            scope=cast(Scope, scope),
            model_config={},
        )


@pytest.mark.parametrize(
    "origin", [pd.Timestamp("2026-01-20"), pd.Timestamp("2026-01-19", tz="UTC")]
)
def test_partition_requires_an_on_calendar_naive_origin(origin: pd.Timestamp) -> None:
    with pytest.raises(PanelError, match="calendar|timezone-naive"):
        _panel().forecast_tasks(
            origin=origin,
            horizon=1,
            scope=Scope.GLOBAL,
            model_config={},
        )


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_partition_requires_positive_integer_horizon(horizon: object) -> None:
    with pytest.raises(ForecastTaskError, match="positive integer"):
        _panel().forecast_tasks(
            origin=pd.Timestamp("2026-01-19"),
            horizon=cast(int, horizon),
            scope=Scope.GLOBAL,
            model_config={},
        )


def test_future_exogenous_is_canonical_and_partitioned_with_tasks() -> None:
    global_task = _tasks(future=_future())[0]
    local_tasks = _tasks(scope=Scope.LOCAL, future=_future())

    assert global_task.future_exogenous is not None
    assert global_task.future_exogenous[SERIES_KEY].tolist() == ["sku-a", "sku-b"]
    assert global_task.future_exogenous["promotion"].dtype == np.dtype("int64")
    local_keys = [
        task.future_exogenous[SERIES_KEY].tolist()
        for task in local_tasks
        if task.future_exogenous is not None
    ]
    assert local_keys == [["sku-a"], ["sku-b"]]


@pytest.mark.parametrize(
    ("future", "pattern"),
    [
        (_future(known_at="2026-01-20"), "known at or before origin"),
        (_future().assign(timestamp=pd.to_datetime(["2026-02-02", "2026-01-19"])), "horizon"),
        (
            _future().assign(series_key=pd.Series(["unknown", "sku-a"], dtype="string")),
            "unknown series",
        ),
        (_future().assign(promotion=pd.Series([True, False])), "native NumPy"),
        (_future().assign(promotion=pd.Series([np.nan, 1.0])), "must be known"),
        (_future().drop(columns=KNOWN_AT), "missing columns"),
    ],
)
def test_future_exogenous_rejects_unknown_or_temporally_illegitimate_facts(
    future: pd.DataFrame, pattern: str
) -> None:
    with pytest.raises(PanelError, match=pattern):
        _tasks(future=future)


@pytest.mark.parametrize(
    "config",
    [
        {"x": np.nan},
        {"x": np.inf},
        {"x": (1, 2)},
        {1: "non-string-key"},
        {"x": np.array([1])},
    ],
)
def test_task_model_configuration_must_be_finite_json(config: dict[object, object]) -> None:
    with pytest.raises(ForecastTaskError, match="finite|JSON|string object key"):
        _tasks(config=cast(dict[str, object], config))


def test_task_model_configuration_wraps_structural_json_errors() -> None:
    cycle: list[object] = []
    cycle.append(cycle)

    for config in ({"x": cycle}, {"x": "\ud800"}):
        with pytest.raises(ForecastTaskError, match="finite JSON values"):
            _tasks(config=config)


def test_task_model_configuration_wraps_excessive_json_nesting() -> None:
    nested: object = None
    for _ in range(sys.getrecursionlimit() + 10):
        nested = [nested]

    with pytest.raises(ForecastTaskError, match="finite JSON values"):
        _tasks(config={"nested": nested})


def test_task_defensively_copies_every_mutable_surface() -> None:
    future = _future()
    nested = [1]
    config: dict[str, object] = {"backend": "seasonal-naive", "m": 2, "nested": nested}
    task = _tasks(config=config, future=future)[0]

    future.loc[0, "promotion"] = 999
    nested.append(2)
    exposed_history = task.history
    exposed_history.loc[:, OBSERVED_VALUE] = 999
    exposed_future = task.future_exogenous
    assert exposed_future is not None
    exposed_future.loc[:, "promotion"] = 999
    exposed_config = task.model_config
    cast(list[int], exposed_config["nested"]).append(3)

    assert 999 not in task.history[OBSERVED_VALUE].dropna().tolist()
    assert task.future_exogenous is not None
    assert 999 not in task.future_exogenous["promotion"].tolist()
    assert task.model_config["nested"] == [1]


def test_task_round_trip_is_exact_and_preserves_datetime_resolution() -> None:
    task = _panel(resolution="ms").forecast_tasks(
        origin=pd.Timestamp(np.datetime64("2026-01-19", "ms")),
        horizon=2,
        scope=Scope.GLOBAL,
        model_config={"backend": "seasonal-naive", "m": 2},
        future_exogenous=_future(),
    )[0]

    restored = ForecastTask.from_bytes(task.to_bytes())

    assert restored.to_bytes() == task.to_bytes()
    assert restored.history[TIMESTAMP].dtype == np.dtype("datetime64[ms]")
    assert restored.origin.unit == "ms"
    pd.testing.assert_frame_equal(restored.history, task.history)
    restored_future = restored.future_exogenous
    task_future = task.future_exogenous
    assert restored_future is not None
    assert task_future is not None
    pd.testing.assert_frame_equal(restored_future, task_future)
    assert restored.calendar == task.calendar
    assert restored.scope is task.scope
    assert restored.series_keys == task.series_keys
    assert restored.model_config == task.model_config


def test_task_bytes_ignore_valid_interleaving_column_and_config_mapping_order() -> None:
    frame = _panel_frame()
    first = Panel.from_frame(frame, calendar=Calendar("W-MON")).forecast_tasks(
        origin=pd.Timestamp("2026-01-19"),
        horizon=2,
        scope=Scope.GLOBAL,
        model_config={"backend": "seasonal-naive", "m": 2},
    )[0]
    second = Panel.from_frame(
        frame.iloc[[0, 2, 1, 3, 5, 4]][list(reversed(frame.columns))],
        calendar=Calendar("W-MON"),
    ).forecast_tasks(
        origin=pd.Timestamp("2026-01-19"),
        horizon=2,
        scope=Scope.GLOBAL,
        model_config={"m": 2, "backend": "seasonal-naive"},
    )[0]
    assert first.to_bytes() == second.to_bytes()


def test_task_bytes_and_enumeration_are_deterministic_across_fresh_processes() -> None:
    script = """
import pandas as pd
from newcalibre.domain import Calendar, Panel, Scope
keys = list({'é', 'a', 'z'})
frame = pd.DataFrame({
    'series_key': pd.Series(keys, dtype='string'),
    'timestamp': pd.to_datetime(['2026-01-05'] * 3),
    'value': [{'a': 1.0, 'z': 2.0, 'é': 3.0}[key] for key in keys],
})
task = Panel.from_frame(frame, calendar=Calendar('W-MON')).forecast_tasks(
    origin=pd.Timestamp('2026-01-12'), horizon=2, scope=Scope.GLOBAL,
    model_config={'m': 1, 'backend': 'seasonal-naive'},
)[0]
print('|'.join(task.series_keys))
print(task.to_bytes().hex())
"""
    outputs: list[str] = []
    for seed in ("1", "777"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert outputs[0].splitlines()[0] == "a|z|é"


def test_task_deserialization_rejects_corruption_version_truncation_and_trailing_bytes() -> None:
    data = _tasks()[0].to_bytes()
    corrupt = bytearray(data)
    corrupt[-40] ^= 1
    wrong_version = bytearray(data)
    wrong_version[4] = 2

    with pytest.raises(ForecastTaskError, match="integrity"):
        ForecastTask.from_bytes(bytes(corrupt))
    with pytest.raises(ForecastTaskError, match="version"):
        ForecastTask.from_bytes(bytes(wrong_version))
    with pytest.raises(ForecastTaskError, match="truncated"):
        ForecastTask.from_bytes(data[:-40])
    with pytest.raises(ForecastTaskError, match="trailing"):
        ForecastTask.from_bytes(data + b"extra")


def test_task_deserialization_rejects_authenticated_schema_drift() -> None:
    data = _tasks()[0].to_bytes()
    payload = bytearray(data[:-32])
    marker = b'"dtype":"float64"'
    index = payload.find(marker)
    assert index >= 0
    payload[index : index + len(marker)] = b'"dtype":"float32"'
    drifted = bytes(payload) + hashlib.sha256(payload).digest()

    with pytest.raises(ForecastTaskError, match="schema drifted"):
        ForecastTask.from_bytes(drifted)
