"""Load VN2 wide tables with progressive reveal and temporal hygiene."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Integral
from pathlib import Path
from typing import Self, cast

import numpy as np
import pandas as pd

from newcalibre.domain import CalendarError
from newcalibre.protocols.vn2.config import VN2ProtocolConfig
from newcalibre.protocols.vn2.inventory import (
    VN2InputError,
    VN2InputInventory,
    read_verified_vn2_input,
    verify_vn2_inputs,
)

_DATE_COLUMN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_INTEGER_KEY = re.compile(r"[0-9]+")
_SIGNED_INT64_MAX = 2**63 - 1


class VN2DataError(VN2InputError):
    """Report challenge bytes that violate the configured VN2 data contract."""


@dataclass(frozen=True, slots=True)
class VN2RoundInput:
    """Expose only facts admissible at one decision origin."""

    round_number: int
    reveal_number: int
    origin: pd.Timestamp
    sales: pd.DataFrame = field(repr=False)
    master: pd.DataFrame = field(repr=False)
    in_stock: pd.DataFrame = field(repr=False)
    initial_state: pd.DataFrame = field(repr=False)


@dataclass(frozen=True, slots=True)
class VN2WeeklyActuals:
    """Expose one realized protocol week without carrying later reveals."""

    week_number: int
    period: pd.Timestamp
    sales: pd.DataFrame = field(repr=False)


class VN2Dataset:
    """Own validated reveals while exposing only origin-safe round snapshots."""

    _config: VN2ProtocolConfig
    _input_inventory_sha256: str
    _sales: pd.DataFrame
    _master: pd.DataFrame
    _in_stock: pd.DataFrame
    _initial_state: pd.DataFrame

    def __init__(self) -> None:
        raise TypeError("VN2Dataset must be created with load_vn2_dataset()")

    @classmethod
    def _from_validated(
        cls,
        *,
        config: VN2ProtocolConfig,
        input_inventory_sha256: str,
        sales: pd.DataFrame,
        master: pd.DataFrame,
        in_stock: pd.DataFrame,
        initial_state: pd.DataFrame,
    ) -> Self:
        instance = object.__new__(cls)
        instance._config = config
        instance._input_inventory_sha256 = input_inventory_sha256
        instance._sales = sales
        instance._master = master
        instance._in_stock = in_stock
        instance._initial_state = initial_state
        return instance

    @property
    def config(self) -> VN2ProtocolConfig:
        """Return the immutable protocol configuration used for validation."""
        return self._config

    @property
    def input_inventory_sha256(self) -> str:
        """Return the exact input-inventory identity verified by the loader."""
        return self._input_inventory_sha256

    def round_input(self, round_number: int) -> VN2RoundInput:
        """Return all reveals before the configured origin and its static facts."""
        if (
            isinstance(round_number, bool)
            or not isinstance(round_number, Integral)
            or not 1 <= int(round_number) <= self._config.round_count
        ):
            raise VN2DataError(f"round must be an integer in 1..{self._config.round_count}")
        normalized = int(round_number)
        key_count = len(self._config.columns.series_keys)
        origin = self._config.decision_origins[normalized - 1]
        visible_dates = tuple(
            column for column in self._sales.columns[key_count:] if pd.Timestamp(column) < origin
        )
        reveal_number = len(visible_dates) - self._config.history.initial_periods
        sales = self._sales[[*self._config.columns.series_keys, *visible_dates]].copy(deep=True)
        return VN2RoundInput(
            round_number=normalized,
            reveal_number=reveal_number,
            origin=origin,
            sales=sales,
            master=self._master.copy(deep=True),
            in_stock=self._in_stock.copy(deep=True),
            initial_state=self._initial_state.copy(deep=True),
        )

    def weekly_actuals(self, week_number: int) -> VN2WeeklyActuals:
        """Return one realized decision or drain week as an isolated long slice."""
        final_week = len(self._config.realized_periods)
        if (
            isinstance(week_number, bool)
            or not isinstance(week_number, Integral)
            or not 1 <= int(week_number) <= final_week
        ):
            raise VN2DataError(f"week must be an integer in 1..{final_week}")
        normalized = int(week_number)
        period = self._config.realized_periods[normalized - 1]
        period_column = period.strftime("%Y-%m-%d")
        sales = self._sales[[*self._config.columns.series_keys, period_column]].copy(deep=True)
        sales.rename(columns={period_column: "sales"}, inplace=True)
        return VN2WeeklyActuals(
            week_number=normalized,
            period=period,
            sales=sales,
        )


def load_vn2_dataset(
    data_directory: Path,
    inventory_path: Path,
    config: VN2ProtocolConfig,
) -> VN2Dataset:
    """Verify the configured inventory, then validate and retain challenge tables."""
    if not isinstance(config, VN2ProtocolConfig):
        raise VN2DataError("config must be a VN2ProtocolConfig")
    try:
        inventory = verify_vn2_inputs(data_directory, inventory_path)
    except VN2InputError as error:
        raise VN2DataError(str(error)) from error
    inventory_names = frozenset(inventory.by_name)
    configured_names = config.files.all_names
    if inventory_names != configured_names:
        missing = sorted(configured_names - inventory_names)
        extra = sorted(inventory_names - configured_names)
        raise VN2DataError(
            f"configured files do not match approved inventory: missing={missing} extra={extra}"
        )

    previous: pd.DataFrame | None = None
    expected_base_dates = tuple(
        config.calendar.advance(config.history.first_week, offset).strftime("%Y-%m-%d")
        for offset in range(config.history.initial_periods)
    )
    for reveal_number, name in enumerate(config.files.sales_reveals):
        frame = _read_csv(
            data_directory,
            inventory,
            name,
            key_columns=config.columns.series_keys,
        )
        normalized = _normalize_sales(
            frame,
            config=config,
            reveal_number=reveal_number,
            expected_base_dates=expected_base_dates,
            previous=previous,
        )
        previous = normalized

    assert previous is not None
    sales = previous
    reference_keys = _key_rows(sales, config=config)
    master = _normalize_master(
        _read_csv(
            data_directory,
            inventory,
            config.files.master,
            key_columns=config.columns.series_keys,
        ),
        config=config,
        reference_keys=reference_keys,
    )
    in_stock = _normalize_in_stock(
        _read_csv(
            data_directory,
            inventory,
            config.files.in_stock,
            key_columns=config.columns.series_keys,
        ),
        config=config,
        reference_keys=reference_keys,
        expected_base_dates=expected_base_dates,
        allowed_dates=tuple(sales.columns[len(config.columns.series_keys) :]),
    )
    initial_state = _normalize_initial_state(
        _read_csv(
            data_directory,
            inventory,
            config.files.initial_state,
            key_columns=config.columns.series_keys,
        ),
        config=config,
        reference_keys=reference_keys,
    )
    return VN2Dataset._from_validated(
        config=config,
        input_inventory_sha256=inventory.content_sha256,
        sales=sales,
        master=master,
        in_stock=in_stock,
        initial_state=initial_state,
    )


def _read_csv(
    data_directory: Path,
    inventory: VN2InputInventory,
    name: str,
    *,
    key_columns: tuple[str, str],
) -> pd.DataFrame:
    try:
        payload = read_verified_vn2_input(data_directory, name, inventory)
        return pd.read_csv(
            io.BytesIO(payload),
            dtype={column: "string" for column in key_columns},
        )
    except VN2InputError as error:
        raise VN2DataError(str(error)) from error
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        raise VN2DataError(f"{name}: CSV parse failed") from error


def _normalize_sales(
    frame: pd.DataFrame,
    *,
    config: VN2ProtocolConfig,
    reveal_number: int,
    expected_base_dates: tuple[str, ...],
    previous: pd.DataFrame | None,
) -> pd.DataFrame:
    surface = f"sales reveal {reveal_number}"
    _require_unique_columns(frame, surface=surface)
    key_columns = config.columns.series_keys
    if tuple(frame.columns[: len(key_columns)]) != key_columns:
        raise VN2DataError(f"{surface} must begin with exact Store/Product key columns")
    date_columns = tuple(frame.columns[len(key_columns) :])
    expected_count = config.history.initial_periods + reveal_number
    if len(date_columns) != expected_count:
        raise VN2DataError(
            f"{surface} must append exactly one date column; found {len(date_columns)}"
        )
    expected_new: str | None = None
    if reveal_number == 0:
        if date_columns != expected_base_dates:
            raise VN2DataError("round-zero sales columns do not match configured history shape")
    else:
        assert previous is not None
        previous_dates = tuple(previous.columns[len(key_columns) :])
        if date_columns[:-1] != previous_dates or len(date_columns) != len(previous_dates) + 1:
            raise VN2DataError(f"{surface} must append exactly one date column")
        expected_new = config.calendar.advance(pd.Timestamp(previous_dates[-1]), 1).strftime(
            "%Y-%m-%d"
        )
    _validate_date_columns(date_columns, config=config, surface=surface)
    if expected_new is not None and date_columns[-1] != expected_new:
        raise VN2DataError(f"{surface} new date must follow exact weekly cadence")

    normalized = frame.copy(deep=True)
    keys = _normalize_keys(normalized, config=config, surface=surface)
    if len(keys) != config.series_count:
        raise VN2DataError(
            f"{surface} must contain exactly {config.series_count} Store/Product rows"
        )
    for column in date_columns:
        normalized[column] = _nonnegative_float_series(
            normalized[column],
            name=f"{surface} column {column}",
            missing_as_zero=True,
        )
    if previous is not None:
        previous_keys = _key_rows(previous, config=config)
        if keys != previous_keys:
            raise VN2DataError(f"{surface} Store/Product key order changed")
        prior_columns = [*key_columns, *date_columns[:-1]]
        if not normalized[prior_columns].equals(previous[prior_columns]):
            raise VN2DataError(f"{surface} changed previously revealed values")
    return normalized


def _normalize_master(
    frame: pd.DataFrame,
    *,
    config: VN2ProtocolConfig,
    reference_keys: tuple[tuple[int, int], ...],
) -> pd.DataFrame:
    surface = "master table"
    expected = (*config.columns.series_keys, *config.columns.master_attributes)
    _require_exact_columns(frame, expected=expected, surface=surface)
    normalized = frame.copy(deep=True)
    if _normalize_keys(normalized, config=config, surface=surface) != reference_keys:
        raise VN2DataError("master table Store/Product key order differs from sales")
    if normalized[list(config.columns.master_attributes)].isna().any(axis=None):
        raise VN2DataError("master table attributes must be present for every series")
    return normalized


def _normalize_in_stock(
    frame: pd.DataFrame,
    *,
    config: VN2ProtocolConfig,
    reference_keys: tuple[tuple[int, int], ...],
    expected_base_dates: tuple[str, ...],
    allowed_dates: tuple[str, ...],
) -> pd.DataFrame:
    surface = "in-stock table"
    _require_unique_columns(frame, surface=surface)
    key_columns = config.columns.series_keys
    if tuple(frame.columns[: len(key_columns)]) != key_columns:
        raise VN2DataError("in-stock table must begin with exact Store/Product key columns")
    supplied_dates = tuple(frame.columns[len(key_columns) :])
    _validate_date_columns(supplied_dates, config=config, surface=surface)
    if (
        len(supplied_dates) < len(expected_base_dates)
        or supplied_dates[: len(expected_base_dates)] != expected_base_dates
        or len(supplied_dates) > len(allowed_dates)
        or supplied_dates != allowed_dates[: len(supplied_dates)]
    ):
        raise VN2DataError(
            "in-stock date columns must be a prefix of sales containing the complete base history"
        )
    normalized = frame.copy(deep=True)
    if _normalize_keys(normalized, config=config, surface=surface) != reference_keys:
        raise VN2DataError("in-stock table Store/Product key order differs from sales")
    for column in supplied_dates:
        values = normalized[column].tolist()
        if any(not isinstance(value, (bool, np.bool_)) for value in values):
            raise VN2DataError(f"in-stock column {column} must contain exact boolean values")
        normalized[column] = pd.Series(values, dtype="bool")
    return normalized[[*key_columns, *expected_base_dates]].copy(deep=True)


def _normalize_initial_state(
    frame: pd.DataFrame,
    *,
    config: VN2ProtocolConfig,
    reference_keys: tuple[tuple[int, int], ...],
) -> pd.DataFrame:
    surface = "initial-state table"
    _require_exact_columns(
        frame,
        expected=config.columns.initial_state_columns,
        surface=surface,
    )
    normalized = frame.copy(deep=True)
    if _normalize_keys(normalized, config=config, surface=surface) != reference_keys:
        raise VN2DataError("initial-state table Store/Product key order differs from sales")
    for column in config.columns.initial_state_columns[len(config.columns.series_keys) :]:
        normalized[column] = _nonnegative_float_series(
            normalized[column],
            name=f"initial-state column {column}",
            missing_as_zero=False,
        )
    return normalized


def _normalize_keys(
    frame: pd.DataFrame,
    *,
    config: VN2ProtocolConfig,
    surface: str,
) -> tuple[tuple[int, int], ...]:
    key_columns = config.columns.series_keys
    if frame[list(key_columns)].isna().any(axis=None):
        raise VN2DataError(f"{surface} Store/Product key cannot contain missing values")
    for column in key_columns:
        values = [
            _exact_integer_key(value, surface=surface, column=column) for value in frame[column]
        ]
        frame[column] = pd.Series(values, index=frame.index, dtype="int64")
    if frame.duplicated(subset=list(key_columns)).any():
        raise VN2DataError(f"{surface} contains duplicate Store/Product keys")
    return _key_rows(frame, config=config)


def _exact_integer_key(value: object, *, surface: str, column: str) -> int:
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        normalized = int(value)
    elif isinstance(value, str) and _INTEGER_KEY.fullmatch(value) is not None:
        significant_digits = value.lstrip("0") or "0"
        if len(significant_digits) > 19:
            raise VN2DataError(
                f"{surface} {column} key must be a non-negative signed 64-bit integer"
            )
        normalized = int(significant_digits, 10)
    else:
        raise VN2DataError(f"{surface} {column} key must be a non-negative signed 64-bit integer")
    if not 0 <= normalized <= _SIGNED_INT64_MAX:
        raise VN2DataError(f"{surface} {column} key must be a non-negative signed 64-bit integer")
    return normalized


def _key_rows(
    frame: pd.DataFrame,
    *,
    config: VN2ProtocolConfig,
) -> tuple[tuple[int, int], ...]:
    first, second = config.columns.series_keys
    return tuple(
        (int(left), int(right)) for left, right in zip(frame[first], frame[second], strict=True)
    )


def _nonnegative_float_series(
    series: pd.Series,
    *,
    name: str,
    missing_as_zero: bool,
) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
        if missing_as_zero:
            numeric = numeric.fillna(0.0)
        values = numeric.to_numpy(dtype="float64")
    except (TypeError, ValueError) as error:
        raise VN2DataError(f"{name} must contain non-negative floats") from error
    if np.isnan(values).any() and not missing_as_zero:
        raise VN2DataError(f"{name} must not contain missing values")
    if not np.isfinite(values).all():
        raise VN2DataError(f"{name} must contain finite values")
    if (values < 0.0).any():
        raise VN2DataError(f"{name} must contain non-negative values")
    return pd.Series(values, index=series.index, name=series.name, dtype="float64")


def _validate_date_columns(
    columns: tuple[object, ...],
    *,
    config: VN2ProtocolConfig,
    surface: str,
) -> None:
    if any(
        not isinstance(column, str) or _DATE_COLUMN.fullmatch(column) is None for column in columns
    ):
        raise VN2DataError(f"{surface} non-key columns must be exact YYYY-MM-DD dates")
    try:
        timestamps = tuple(pd.Timestamp(cast(str, column)) for column in columns)
    except (OverflowError, TypeError, ValueError) as error:
        raise VN2DataError(f"{surface} non-key columns must be valid YYYY-MM-DD dates") from error
    for timestamp in timestamps:
        try:
            config.calendar.require_member(timestamp, name="week column")
        except CalendarError as error:
            raise VN2DataError(f"{surface} week columns must use the Monday anchor") from error
    for previous, current in pairwise(timestamps):
        if config.calendar.advance(previous, 1) != current:
            raise VN2DataError(f"{surface} week columns must follow exact weekly cadence")


def _require_exact_columns(
    frame: pd.DataFrame,
    *,
    expected: tuple[str, ...],
    surface: str,
) -> None:
    _require_unique_columns(frame, surface=surface)
    if tuple(frame.columns) != expected:
        raise VN2DataError(f"{surface} must contain exact columns in protocol order")


def _require_unique_columns(frame: pd.DataFrame, *, surface: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise VN2DataError(f"{surface} must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise VN2DataError(f"{surface} contains duplicate column labels")
    if any(not isinstance(column, str) for column in frame.columns):
        raise VN2DataError(f"{surface} column labels must be strings")
