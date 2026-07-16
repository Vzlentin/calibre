"""Exercise the closed U3a panel, calendar, and task transport contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from newcalibre.domain import (
    ACTUAL_VALUE,
    AVAILABILITY_BOUND,
    CENSOR_STATUS,
    FITTED_VALUE,
    HORIZON_STEP,
    KNOWN_AT,
    MODEL_NAME,
    OBSERVED_VALUE,
    ORIGIN,
    POINT_FORECAST,
    REQUIRED_FITTED_VALUE_COLUMNS,
    REQUIRED_FRAME_COLUMNS,
    REQUIRED_PANEL_COLUMNS,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    TIMESTAMP,
    UNDECLARED_CENSORING,
    Calendar,
    CalendarError,
    FittedValues,
    FittedValuesError,
    ForecastFrameError,
    ForecastTask,
    ForecastTaskError,
    Panel,
    PanelError,
    Scope,
    validate_forecast_frame,
)

pytestmark = pytest.mark.tier1
FRAME_CALENDAR = Calendar("W-MON").bind(pd.Timestamp("2026-01-19"))


def _panel_frame(*, string_storage: str = "pyarrow") -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(
                ["sku-a", "sku-b", "sku-a", "sku-b", "sku-b", "sku-a"],
                dtype=pd.StringDtype(storage=string_storage),
            ),
            TIMESTAMP: pd.Series(
                pd.to_datetime(
                    [
                        "2026-01-05",
                        "2026-01-05",
                        "2026-01-12",
                        "2026-01-12",
                        "2026-01-19",
                        "2026-01-19",
                    ]
                ).astype("datetime64[ms]")
            ),
            OBSERVED_VALUE: pd.Series([1, 1, 2, 2, 3, 3], dtype="int16"),
            CENSOR_STATUS: pd.Series(
                ["uncensored", "undeclared", "censored", None, None, "uncensored"],
                dtype=pd.StringDtype(storage=string_storage),
            ),
            AVAILABILITY_BOUND: pd.Series([1, 1, 2, 2, 3, 3], dtype="uint16"),
            "price": pd.Series([1, 1.5, 1.5, 2, 1, 2], dtype="float32"),
        }
    )


def _future(*, string_storage: str = "pyarrow") -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku-b", "sku-a"], dtype=pd.StringDtype(storage=string_storage)),
            TIMESTAMP: pd.Series(
                pd.to_datetime(["2026-01-26", "2026-01-19"]).astype("datetime64[us]")
            ),
            KNOWN_AT: pd.Series(
                pd.to_datetime(["2026-01-19", "2026-01-12"]).astype("datetime64[ms]")
            ),
            "promotion": pd.Series([0, 1], dtype="uint8"),
        }
    )


def _task(
    *,
    frame: pd.DataFrame | None = None,
    future: pd.DataFrame | None = None,
    scope: Scope = Scope.GLOBAL,
    calendar: Calendar | None = None,
    origin: pd.Timestamp | None = None,
    model_config: dict[str, object] | None = None,
) -> ForecastTask:
    panel = Panel.from_frame(
        frame if frame is not None else _panel_frame(), calendar=calendar or Calendar("W-MON")
    )
    return panel.forecast_tasks(
        origin=origin or pd.Timestamp(np.datetime64("2026-01-19", "ms")),
        horizon=2,
        scope=scope,
        model_config=model_config or {"backend": "seasonal-naive", "m": 2},
        future_exogenous=future,
    )[0]


def test_multiplied_weekly_calendar_binds_panel_phase_and_round_trips_it() -> None:
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku", "sku"], dtype="string"),
            TIMESTAMP: pd.Series(
                pd.to_datetime(["2026-01-05", "2026-01-19"]).astype("datetime64[ms]")
            ),
            OBSERVED_VALUE: pd.Series([1.0, 2.0], dtype="float64"),
        }
    )
    panel = Panel.from_frame(frame, calendar=Calendar("2W-MON"))

    assert panel.calendar.phase == pd.Timestamp(np.datetime64("2026-01-05", "ms"))
    assert panel.calendar.contains(pd.Timestamp("2026-01-05"))
    assert panel.calendar.contains(pd.Timestamp("2026-01-19"))
    assert not panel.calendar.contains(pd.Timestamp("2026-01-12"))

    task = panel.forecast_tasks(
        origin=pd.Timestamp(np.datetime64("2026-01-19", "ms")),
        horizon=2,
        scope=Scope.GLOBAL,
        model_config={"backend": "seasonal-naive", "m": 1},
    )[0]
    restored = ForecastTask.from_bytes(task.to_bytes())
    assert restored.calendar == task.calendar
    assert restored.calendar.phase is not None
    assert restored.calendar.phase.unit == "ms"


@pytest.mark.parametrize(
    ("frequency", "valid", "invalid"),
    [
        ("2W-MON", ["2026-01-05", "2026-01-19"], "2026-01-12"),
        ("2D", ["2026-01-05", "2026-01-07"], "2026-01-06"),
    ],
)
def test_multiplied_calendar_rejects_observation_and_origin_off_bound_phase(
    frequency: str, valid: list[str], invalid: str
) -> None:
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku", "sku"], dtype="string"),
            TIMESTAMP: pd.to_datetime(valid),
            OBSERVED_VALUE: pd.Series([1.0, 2.0], dtype="float64"),
        }
    )
    panel = Panel.from_frame(frame, calendar=Calendar(frequency))
    bad_observation = pd.concat(
        [
            frame.iloc[::-1],
            pd.DataFrame(
                {
                    SERIES_KEY: pd.Series(["sku"], dtype="string"),
                    TIMESTAMP: pd.to_datetime([invalid]),
                    OBSERVED_VALUE: pd.Series([3.0], dtype="float64"),
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(PanelError, match="does not lie on calendar"):
        Panel.from_frame(bad_observation, calendar=Calendar(frequency))
    with pytest.raises(PanelError, match="does not lie on calendar"):
        panel.forecast_tasks(
            origin=pd.Timestamp(invalid),
            horizon=1,
            scope=Scope.GLOBAL,
            model_config={},
        )


def test_weekly_anchor_refuses_a_monday_under_sunday_calendar() -> None:
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku"], dtype="string"),
            TIMESTAMP: pd.to_datetime(["2026-01-05"]),
            OBSERVED_VALUE: pd.Series([1.0], dtype="float64"),
        }
    )
    with pytest.raises(PanelError, match="does not lie on calendar"):
        Panel.from_frame(frame, calendar=Calendar("W-SUN"))


def test_unbound_calendar_cannot_answer_membership_without_dataset_phase() -> None:
    with pytest.raises(CalendarError, match="bound to a dataset phase"):
        Calendar("D").contains(pd.Timestamp("2026-01-05"))


@pytest.mark.parametrize(
    ("frequency", "valid", "invalid_observation", "invalid_origin"),
    [
        (
            "D",
            ["2026-01-05 00:00", "2026-01-06 00:00"],
            "2026-01-06 12:00",
            "2026-01-07 12:00",
        ),
        (
            "W-MON",
            ["2026-01-05 00:00", "2026-01-12 00:00"],
            "2026-01-12 12:00",
            "2026-01-19 12:00",
        ),
        (
            "h",
            ["2026-01-05 00:00", "2026-01-05 01:00"],
            "2026-01-05 01:30",
            "2026-01-05 02:30",
        ),
    ],
)
def test_every_calendar_frequency_enforces_observation_and_origin_clock_phase(
    frequency: str,
    valid: list[str],
    invalid_observation: str,
    invalid_origin: str,
) -> None:
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku", "sku"], dtype="string"),
            TIMESTAMP: pd.to_datetime(valid),
            OBSERVED_VALUE: pd.Series([1.0, 2.0], dtype="float64"),
        }
    )
    panel = Panel.from_frame(frame, calendar=Calendar(frequency))
    bad_observation = pd.concat(
        [
            frame,
            pd.DataFrame(
                {
                    SERIES_KEY: pd.Series(["sku"], dtype="string"),
                    TIMESTAMP: pd.to_datetime([invalid_observation]),
                    OBSERVED_VALUE: pd.Series([3.0], dtype="float64"),
                }
            ),
        ],
        ignore_index=True,
    )

    assert panel.calendar.phase == pd.Timestamp(valid[0])
    with pytest.raises(PanelError, match="does not lie on calendar"):
        Panel.from_frame(bad_observation, calendar=Calendar(frequency))
    with pytest.raises(PanelError, match="does not lie on calendar"):
        panel.forecast_tasks(
            origin=pd.Timestamp(invalid_origin),
            horizon=1,
            scope=Scope.GLOBAL,
            model_config={},
        )


def test_fixed_tick_membership_is_constant_space_for_far_timestamps() -> None:
    phase = pd.Timestamp(np.datetime64("1900-01-01T00:00:00", "s"))
    calendar = Calendar("2s").bind(phase)
    far_member = pd.Timestamp(np.datetime64("9999-12-31T23:59:58", "s"))

    assert calendar.contains(far_member)
    assert not calendar.contains(far_member + pd.Timedelta(seconds=1))


def test_nonfixed_advance_and_retreat_remain_on_the_phase_bound_grid() -> None:
    phase = pd.Timestamp("2026-01-05 09:00")
    calendar = Calendar("bh").bind(phase)

    next_hour = calendar.advance(phase, 1)

    assert next_hour == pd.Timestamp("2026-01-05 10:00")
    assert calendar.contains(next_hour)
    assert calendar.retreat(next_hour, 1) == phase

    closing_phase = pd.Timestamp("2026-01-05 17:00")
    closing_calendar = Calendar("bh").bind(closing_phase)
    previous_hour = closing_calendar.retreat(closing_phase, 1)
    assert closing_calendar.contains(previous_hour)
    assert closing_calendar.advance(previous_hour, 1) == closing_phase


def test_panel_refuses_a_prebound_noncanonical_phase() -> None:
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku", "sku"], dtype="string"),
            TIMESTAMP: pd.Series(
                pd.to_datetime(["2026-01-05", "2026-01-19"]).astype("datetime64[ms]")
            ),
            OBSERVED_VALUE: pd.Series([1.0, 2.0], dtype="float64"),
        }
    )
    calendar = Calendar("2W-MON", phase=pd.Timestamp(np.datetime64("2026-01-19", "ms")))

    with pytest.raises(PanelError, match="already bound"):
        Panel.from_frame(frame, calendar=calendar)


def test_rebinding_same_phase_in_another_resolution_retains_exact_representation() -> None:
    phase = pd.Timestamp(np.datetime64("2026-01-05", "ms"))
    calendar = Calendar("2W-MON", phase=phase)

    rebound = calendar.bind(pd.Timestamp(np.datetime64("2026-01-05", "us")))

    assert rebound is calendar
    assert rebound.phase is not None
    assert rebound.phase.unit == "ms"


def test_censor_null_and_explicit_undeclared_share_one_transport_sentinel() -> None:
    panel = Panel.from_frame(_panel_frame(), calendar=Calendar("W-MON"))
    statuses = panel.frame[CENSOR_STATUS]

    assert panel.has_censoring_facts
    assert statuses.isna().sum() == 0
    assert statuses.tolist().count(UNDECLARED_CENSORING) == 3
    assert statuses.dtype.storage == "pyarrow"
    assert (
        panel.frame.loc[panel.frame[CENSOR_STATUS] == UNDECLARED_CENSORING, AVAILABILITY_BOUND]
        .notna()
        .all()
    )

    reingested = Panel.from_frame(panel.frame, calendar=panel.calendar)
    pd.testing.assert_frame_equal(reingested.frame, panel.frame)


def test_absent_censor_fields_remain_absent_through_task_transport() -> None:
    frame = _panel_frame().drop(columns=[CENSOR_STATUS, AVAILABILITY_BOUND])
    panel = Panel.from_frame(frame, calendar=Calendar("W-MON"))
    task = panel.forecast_tasks(
        origin=pd.Timestamp(np.datetime64("2026-01-19", "ms")),
        horizon=1,
        scope=Scope.GLOBAL,
        model_config={},
    )[0]
    restored = ForecastTask.from_bytes(task.to_bytes())

    assert not panel.has_censoring_facts
    assert CENSOR_STATUS not in panel.frame.columns
    assert CENSOR_STATUS not in task.history.columns
    assert CENSOR_STATUS not in restored.history.columns
    pd.testing.assert_frame_equal(restored.history, task.history)


@pytest.mark.parametrize("column", REQUIRED_PANEL_COLUMNS)
def test_panel_rejects_each_missing_required_column(column: str) -> None:
    with pytest.raises(PanelError, match="missing required"):
        Panel.from_frame(_panel_frame().drop(columns=column), calendar=Calendar("W-MON"))


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        (SERIES_KEY, pd.Series(["sku"] * 6, dtype="object")),
        (TIMESTAMP, pd.Series(range(6), dtype="int64")),
        (OBSERVED_VALUE, pd.Series(["1"] * 6, dtype="string")),
    ],
)
def test_panel_rejects_each_mistyped_required_column(column: str, replacement: pd.Series) -> None:
    frame = _panel_frame()
    frame[column] = replacement
    with pytest.raises(PanelError, match=column):
        Panel.from_frame(frame, calendar=Calendar("W-MON"))


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        (
            lambda frame: frame.assign(
                censor_status=pd.Series(["invalid"] * len(frame), dtype="string")
            ),
            "invalid values",
        ),
        (
            lambda frame: frame.assign(censor_status=frame[CENSOR_STATUS].astype("object")),
            "string dtype",
        ),
        (lambda frame: frame.assign(availability_bound=True), "native NumPy"),
        (lambda frame: frame.assign(availability_bound="unknown"), "native NumPy"),
        (
            lambda frame: frame.assign(
                availability_bound=pd.Series([complex(1)] * len(frame), dtype="complex128")
            ),
            "native NumPy",
        ),
        (lambda frame: frame.assign(price=True), "native NumPy"),
        (
            lambda frame: frame.assign(
                price=pd.Series([complex(1)] * len(frame), dtype="complex128")
            ),
            "native NumPy",
        ),
    ],
)
def test_panel_rejects_invalid_status_bound_and_exogenous_values(
    mutation: Callable[[pd.DataFrame], pd.DataFrame], pattern: str
) -> None:
    with pytest.raises(PanelError, match=pattern):
        Panel.from_frame(mutation(_panel_frame()), calendar=Calendar("W-MON"))


def test_sparse_numeric_values_densify_losslessly_to_the_same_task_bytes() -> None:
    dense = _panel_frame().drop(columns=[CENSOR_STATUS, AVAILABILITY_BOUND])
    dense[OBSERVED_VALUE] = pd.Series([3, 0, 1, 2, 0, 3], dtype="float32")
    sparse = dense.copy(deep=True)
    sparse[OBSERVED_VALUE] = pd.Series(
        [3, 0, 1, 2, 0, 3], dtype=pd.SparseDtype("float32", fill_value=0)
    )

    sparse_task = _task(frame=sparse)
    dense_task = _task(frame=dense)

    assert sparse_task.history[OBSERVED_VALUE].dtype == np.dtype("float32")
    assert sparse_task.to_bytes() == dense_task.to_bytes()


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ],
)
def test_every_accepted_numeric_primitive_round_trips_exactly(dtype: str) -> None:
    frame = _panel_frame().drop(columns=[CENSOR_STATUS, AVAILABILITY_BOUND])
    frame[OBSERVED_VALUE] = pd.Series([3, 1, 1, 2, 2, 3], dtype=dtype)
    task = _task(frame=frame)
    restored = ForecastTask.from_bytes(task.to_bytes())

    assert restored.history[OBSERVED_VALUE].dtype == np.dtype(dtype)
    pd.testing.assert_frame_equal(restored.history, task.history)


@pytest.mark.parametrize(
    "dtype",
    [
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
    ],
)
def test_nullable_numeric_dtypes_round_trip_with_missing_values(dtype: str) -> None:
    frame = _panel_frame().drop(columns=[CENSOR_STATUS, AVAILABILITY_BOUND])
    frame[OBSERVED_VALUE] = pd.Series([3, pd.NA, 1, 2, 2, 3], dtype=dtype)
    task = _task(frame=frame)
    restored = ForecastTask.from_bytes(task.to_bytes())

    assert str(restored.history[OBSERVED_VALUE].dtype) == dtype
    pd.testing.assert_frame_equal(restored.history, task.history)


@pytest.mark.parametrize(
    "arrow_type",
    [pa.int64(), pa.uint64(), pa.float16(), pa.float32(), pa.float64()],
    ids=str,
)
def test_arrow_backed_numeric_dtypes_round_trip_with_missing_values(
    arrow_type: pa.DataType,
) -> None:
    frame = _panel_frame().drop(columns=[CENSOR_STATUS, AVAILABILITY_BOUND])
    values: list[object] = [3, None, 1, 2, 2, 3]
    if arrow_type == pa.uint64():
        values = [3, None, 2**63 + 1, 2, 2, 3]
    frame[OBSERVED_VALUE] = pd.Series(values, dtype=pd.ArrowDtype(arrow_type))
    task = _task(frame=frame)

    restored = ForecastTask.from_bytes(task.to_bytes())

    assert restored.history[OBSERVED_VALUE].dtype == pd.ArrowDtype(arrow_type)
    pd.testing.assert_frame_equal(restored.history, task.history)


def test_nullable_mask_payloads_canonicalize_to_identical_task_bytes() -> None:
    first = _panel_frame().drop(columns=[CENSOR_STATUS, AVAILABILITY_BOUND])
    second = first.copy(deep=True)
    mask = np.array([False, True, False, False, False, False])
    first[OBSERVED_VALUE] = pd.Series(
        pd.arrays.IntegerArray(np.array([3, 99, 1, 2, 2, 3], dtype="int16"), mask)
    )
    second[OBSERVED_VALUE] = pd.Series(
        pd.arrays.IntegerArray(np.array([3, 0, 1, 2, 2, 3], dtype="int16"), mask)
    )

    assert _task(frame=first).to_bytes() == _task(frame=second).to_bytes()


def test_nullable_uint64_round_trip_preserves_values_above_float_precision() -> None:
    frame = _panel_frame().drop(columns=[CENSOR_STATUS, AVAILABILITY_BOUND])
    frame[OBSERVED_VALUE] = pd.Series(
        [2**64 - 1, 2**63 + 1, 2**63, 2, pd.NA, 3],
        dtype="UInt64",
    )
    task = _task(frame=frame)

    restored = ForecastTask.from_bytes(task.to_bytes())

    pd.testing.assert_frame_equal(restored.history, task.history)
    assert 2**64 - 1 in restored.history[OBSERVED_VALUE].tolist()
    assert 2**63 + 1 in restored.history[OBSERVED_VALUE].tolist()


def test_nullable_float_canonicalizes_valid_nan_to_the_missing_representation() -> None:
    valid_nan = _panel_frame().drop(columns=[CENSOR_STATUS, AVAILABILITY_BOUND])
    missing = valid_nan.copy(deep=True)
    values = np.array([3.0, np.nan, 0.0, 2.0, 2.0, 3.0], dtype="float32")
    mask = np.array([False, False, True, False, False, False])
    valid_nan[OBSERVED_VALUE] = pd.Series(pd.arrays.FloatingArray(values, mask))
    missing[OBSERVED_VALUE] = pd.Series(pd.arrays.FloatingArray(values, mask | np.isnan(values)))

    valid_nan_task = _task(frame=valid_nan)
    missing_task = _task(frame=missing)
    restored = ForecastTask.from_bytes(valid_nan_task.to_bytes())

    assert valid_nan_task.to_bytes() == missing_task.to_bytes()
    pd.testing.assert_frame_equal(restored.history, valid_nan_task.history)


def test_python_string_storage_canonicalizes_to_arrow_and_round_trips() -> None:
    task = _task(
        frame=_panel_frame(string_storage="python"),
        future=_future(string_storage="python"),
    )
    restored = ForecastTask.from_bytes(task.to_bytes())

    assert task.history[SERIES_KEY].dtype.storage == "pyarrow"
    assert task.history[CENSOR_STATUS].dtype.storage == "pyarrow"
    assert task.future_exogenous is not None
    assert task.future_exogenous[SERIES_KEY].dtype.storage == "pyarrow"
    pd.testing.assert_frame_equal(restored.history, task.history)
    pd.testing.assert_frame_equal(restored.future_exogenous, task.future_exogenous)


def test_task_bytes_discard_pandas_metadata_and_depend_only_on_declared_inputs() -> None:
    marked = _panel_frame()
    marked.attrs["transport-secret-marker"] = "must-not-ship"
    marked.flags.allows_duplicate_labels = False
    marked.index.name = "transport-secret-index"
    marked.columns.name = "transport-secret-columns"
    marked_future = _future()
    marked_future.attrs["transport-secret-marker"] = "must-not-ship"
    marked_future.flags.allows_duplicate_labels = False
    marked_future.index.name = "transport-secret-future-index"
    marked_future.columns.name = "transport-secret-future-columns"

    marked_task = _task(frame=marked, future=marked_future)
    clean_task = _task(frame=_panel_frame(), future=_future())
    payload = marked_task.to_bytes()

    assert marked_task.history.attrs == {}
    assert marked_task.history.flags.allows_duplicate_labels
    assert marked_task.history.index.name is None
    assert marked_task.history.columns.name is None
    assert marked_task.future_exogenous is not None
    assert marked_task.future_exogenous.attrs == {}
    assert marked_task.future_exogenous.flags.allows_duplicate_labels
    assert b"transport-secret" not in payload
    assert b"pandas" not in payload
    assert payload == clean_task.to_bytes()

    restored = ForecastTask.from_bytes(payload)
    pd.testing.assert_frame_equal(restored.history, marked_task.history)
    pd.testing.assert_frame_equal(restored.future_exogenous, marked_task.future_exogenous)


def test_task_bytes_ignore_arrow_string_chunking_and_slice_offsets() -> None:
    direct = _panel_frame()
    chunked = _panel_frame()
    chunked[SERIES_KEY] = pd.concat(
        [chunked[SERIES_KEY].iloc[:2], chunked[SERIES_KEY].iloc[2:]],
        ignore_index=True,
    )
    sliced = _panel_frame()
    padded = pd.Series(
        ["discarded", *sliced[SERIES_KEY].tolist()],
        dtype=pd.StringDtype(storage="pyarrow"),
    )
    sliced[SERIES_KEY] = padded.iloc[1:].reset_index(drop=True)

    assert _task(frame=chunked).to_bytes() == _task(frame=direct).to_bytes()
    assert _task(frame=sliced).to_bytes() == _task(frame=direct).to_bytes()


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_task_bytes_normalize_equivalent_nan_payloads(dtype: str) -> None:
    standard = _panel_frame().drop(columns=[CENSOR_STATUS, AVAILABILITY_BOUND])
    custom = standard.copy(deep=True)
    standard_values = np.array([3, np.nan, 1, 2, 2, 3], dtype=dtype)
    if dtype == "float16":
        custom_values = np.array(
            [0x4200, 0x7E01, 0x3C00, 0x4000, 0x4000, 0x4200],
            dtype="uint16",
        ).view("float16")
    elif dtype == "float32":
        custom_values = np.array(
            [0x40400000, 0x7FC00001, 0x3F800000, 0x40000000, 0x40000000, 0x40400000],
            dtype="uint32",
        ).view("float32")
    else:
        custom_values = np.array(
            [
                0x4008000000000000,
                0x7FF8000000000001,
                0x3FF0000000000000,
                0x4000000000000000,
                0x4000000000000000,
                0x4008000000000000,
            ],
            dtype="uint64",
        ).view("float64")
    standard[OBSERVED_VALUE] = standard_values
    custom[OBSERVED_VALUE] = custom_values

    assert _task(frame=custom).to_bytes() == _task(frame=standard).to_bytes()


@pytest.mark.parametrize("dtype", ["longdouble"])
def test_panel_rejects_numeric_dtypes_without_exact_arrow_round_trip(dtype: str) -> None:
    frame = _panel_frame()
    frame[OBSERVED_VALUE] = np.array([3, 1, 1, 2, 2, 3], dtype=dtype)
    with pytest.raises(PanelError, match="supported by Arrow"):
        Panel.from_frame(frame, calendar=Calendar("W-MON"))


def test_scope_is_not_representable_inside_adapter_configuration() -> None:
    for declared in (Scope.LOCAL.value, Scope.GLOBAL.value):
        with pytest.raises(ForecastTaskError, match="scope is engine configuration"):
            _task(model_config={"backend": "seasonal-naive", "m": 2, "scope": declared})


def test_task_rejects_configuration_and_column_labels_that_are_not_utf8_transportable() -> None:
    with pytest.raises(ForecastTaskError, match="finite|JSON"):
        _task(model_config={"label": "\ud800"})

    frame = _panel_frame()
    frame.columns = pd.Index(
        ["\ud800" if column == "price" else column for column in frame.columns],
        dtype="object",
    )
    with pytest.raises(PanelError, match="UTF-8"):
        Panel.from_frame(frame, calendar=Calendar("W-MON"))


@pytest.mark.parametrize(
    ("mutation", "pattern"),
    [
        (
            lambda frame: frame.assign(timestamp=pd.to_datetime(["2026-01-12", "2026-01-19"])),
            "task horizon",
        ),
        (
            lambda frame: frame.assign(timestamp=pd.to_datetime(["2026-01-20", "2026-01-19"])),
            "does not lie on calendar",
        ),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate"),
        (
            lambda frame: frame.assign(
                known_at=pd.Series([pd.NaT, pd.Timestamp("2026-01-12")], dtype="datetime64[us]")
            ),
            "cannot be missing",
        ),
    ],
)
def test_future_exogenous_rejects_before_origin_off_grid_duplicate_and_unknown_known_at(
    mutation: Callable[[pd.DataFrame], pd.DataFrame], pattern: str
) -> None:
    with pytest.raises(PanelError, match=pattern):
        _task(future=mutation(_future()))


@st.composite
def _valid_multiseries_panels(draw: st.DrawFn) -> pd.DataFrame:
    keys = draw(
        st.lists(
            st.text(alphabet=st.sampled_from(list("abcxyz012")), min_size=1, max_size=5),
            min_size=2,
            max_size=4,
            unique=True,
        )
    )
    rows: list[tuple[str, pd.Timestamp, float]] = []
    for key in keys:
        values = draw(
            st.lists(
                st.floats(
                    min_value=-1_000,
                    max_value=1_000,
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                ),
                min_size=4,
                max_size=4,
            )
        )
        rows.extend(
            (key, pd.Timestamp(timestamp), value)
            for timestamp, value in zip(
                ("2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"),
                values,
                strict=True,
            )
        )
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series([row[0] for row in rows], dtype="string"),
            TIMESTAMP: pd.Series([row[1] for row in rows], dtype="datetime64[us]"),
            OBSERVED_VALUE: pd.Series([row[2] for row in rows], dtype="float32"),
        }
    )


class _HistorySpy:
    def __init__(self) -> None:
        self.seen: list[tuple[ForecastTask, pd.DataFrame]] = []

    def fit(self, task: ForecastTask) -> None:
        self.seen.append((task, task.history))


@given(frame=_valid_multiseries_panels(), scope=st.sampled_from(list(Scope)))
@settings(max_examples=100, deadline=None)
def test_no_at_or_after_origin_history_reaches_adapter(frame: pd.DataFrame, scope: Scope) -> None:
    origin = pd.Timestamp("2026-01-19")
    panel = Panel.from_frame(frame, calendar=Calendar("W-MON"))
    tasks = panel.forecast_tasks(
        origin=origin,
        horizon=2,
        scope=scope,
        model_config={},
    )
    spy = _HistorySpy()

    assert panel.frame[TIMESTAMP].ge(origin).any()
    for task in tasks:
        spy.fit(task)

    for task, seen in spy.seen:
        expected = panel.frame[
            panel.frame[TIMESTAMP].lt(origin) & panel.frame[SERIES_KEY].isin(task.series_keys)
        ].reset_index(drop=True)
        assert seen[TIMESTAMP].lt(origin).all()
        pd.testing.assert_frame_equal(seen, expected)


def _forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku"], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime(["2026-01-19"]),
            ACTUAL_VALUE: pd.Series([np.nan], dtype="float64"),
            POINT_FORECAST: pd.Series([2.0], dtype="float64"),
            HORIZON_STEP: pd.Series([1], dtype="int64"),
            ORIGIN: pd.to_datetime(["2026-01-19"]),
            MODEL_NAME: pd.Series(["model"], dtype="string"),
        }
    )


def _fitted_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku"], dtype="string"),
            TIMESTAMP: pd.to_datetime(["2026-01-12"]),
            ACTUAL_VALUE: pd.Series([1.0], dtype="float64"),
            FITTED_VALUE: pd.Series([1.1], dtype="float64"),
            MODEL_NAME: pd.Series(["model"], dtype="string"),
        }
    )


@pytest.mark.parametrize("column", REQUIRED_FITTED_VALUE_COLUMNS)
def test_fitted_values_rejects_each_missing_required_column(column: str) -> None:
    with pytest.raises(FittedValuesError, match="exact schema"):
        FittedValues.from_frame(_fitted_frame().drop(columns=column))


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        (SERIES_KEY, pd.Series(["sku"], dtype="object")),
        (TIMESTAMP, pd.Series([1], dtype="int64")),
        (ACTUAL_VALUE, pd.Series([True], dtype="bool")),
        (FITTED_VALUE, pd.Series([1 + 2j], dtype="complex128")),
        (MODEL_NAME, pd.Series(["model"], dtype="object")),
    ],
)
def test_fitted_values_rejects_each_mistyped_required_column(
    column: str, replacement: pd.Series
) -> None:
    frame = _fitted_frame()
    frame[column] = replacement
    with pytest.raises(FittedValuesError, match=column):
        FittedValues.from_frame(frame)


@pytest.mark.parametrize("column", ["lower_0.9", "upper_0.9", "quantile_0.5"])
def test_forecast_frame_rejects_mistyped_optional_forecast_values(column: str) -> None:
    frame = _forecast_frame()
    if column.startswith(("lower", "upper")):
        frame["lower_0.9"] = pd.Series([1.0], dtype="float64")
        frame["upper_0.9"] = pd.Series([3.0], dtype="float64")
    else:
        frame[column] = pd.Series([2.0], dtype="float64")
    frame[column] = pd.Series(["bad"], dtype="string")

    with pytest.raises(ForecastFrameError, match=column):
        validate_forecast_frame(frame, calendar=FRAME_CALENDAR)


def test_forecast_and_fitted_surfaces_reject_windows_longdouble_alias() -> None:
    forecast = _forecast_frame()
    forecast[POINT_FORECAST] = np.array([2.0], dtype=np.longdouble)
    with pytest.raises(ForecastFrameError, match="exact float64"):
        validate_forecast_frame(forecast, calendar=FRAME_CALENDAR)

    fitted = _fitted_frame()
    fitted[FITTED_VALUE] = np.array([1.1], dtype=np.longdouble)
    with pytest.raises(FittedValuesError, match="numeric"):
        FittedValues.from_frame(fitted)


def test_forecast_frame_and_fitted_values_reject_each_others_surfaces() -> None:
    forecast = _forecast_frame()
    forecast[FITTED_VALUE] = pd.Series([1.5], dtype="float64")
    with pytest.raises(ForecastFrameError, match="separate fitted-values sidecar"):
        validate_forecast_frame(forecast, calendar=FRAME_CALENDAR)

    with pytest.raises(FittedValuesError, match="exact schema"):
        FittedValues.from_frame(_forecast_frame())
    with pytest.raises(ForecastFrameError, match="missing required columns"):
        validate_forecast_frame(_fitted_frame(), calendar=FRAME_CALENDAR)


def test_public_frame_strings_are_arrow_backed_and_sidecar_metadata_is_stripped() -> None:
    forecast = _forecast_frame()
    forecast[SERIES_KEY] = forecast[SERIES_KEY].astype(pd.StringDtype(storage="python"))
    forecast.flags.allows_duplicate_labels = False
    validated = validate_forecast_frame(forecast, calendar=FRAME_CALENDAR)
    assert validated[SERIES_KEY].dtype.storage == "pyarrow"
    assert validated.flags.allows_duplicate_labels

    fitted = _fitted_frame()
    fitted.attrs["ignored"] = "metadata"
    fitted.flags.allows_duplicate_labels = False
    fitted.index.name = "ignored-index"
    fitted.columns.name = "ignored-columns"
    sidecar = FittedValues.from_frame(fitted)
    assert sidecar.frame[SERIES_KEY].dtype.storage == "pyarrow"
    assert sidecar.frame.attrs == {}
    assert sidecar.frame.flags.allows_duplicate_labels
    assert sidecar.frame.index.name is None
    assert sidecar.frame.columns.name is None


@pytest.mark.parametrize("column", REQUIRED_FRAME_COLUMNS)
def test_cross_surface_fixture_still_covers_every_forecast_required_column(column: str) -> None:
    frame = _forecast_frame().drop(columns=column)
    with pytest.raises(ForecastFrameError, match="missing required columns"):
        validate_forecast_frame(frame, calendar=FRAME_CALENDAR)


def test_invalid_scope_objects_still_fail_before_adapter_configuration() -> None:
    panel = Panel.from_frame(_panel_frame(), calendar=Calendar("W-MON"))
    with pytest.raises(PanelError, match="scope"):
        panel.forecast_tasks(
            origin=pd.Timestamp("2026-01-19"),
            horizon=1,
            scope=cast(Scope, "global"),
            model_config={},
        )
