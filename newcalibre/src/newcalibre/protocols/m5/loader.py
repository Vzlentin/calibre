"""Load and validate one verified M5 evaluation release."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.protocols.m5.config import M5ProtocolConfig
from newcalibre.protocols.m5.inventory import (
    M5InputError,
    M5InputInventory,
    read_verified_m5_input,
    verify_m5_inputs,
)

_SALES_NAME = "sales_train_evaluation.csv"
_CALENDAR_NAME = "calendar.csv"
_SOURCE_FACTS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
_EVALUATION_DAY_COUNT = 1941
_DAY_COLUMNS = tuple(f"d_{index}" for index in range(1, _EVALUATION_DAY_COUNT + 1))
_SIGNED_INT64_MAX = 2**63 - 1
_SIGNED_INT64_BOUND = float(2**63)
_FLOAT_INTEGER_AMBIGUITY_BOUND = float(2**53)


class M5DataError(M5InputError):
    """Report verified M5 bytes that violate the evaluation data contract."""


@dataclass(frozen=True, slots=True, init=False)
class M5Dataset:
    """Own one validated and population-selected M5 evaluation dataset."""

    _config: M5ProtocolConfig
    _input_inventory_sha256: str
    _sales: pd.DataFrame = field(repr=False)
    _dates: tuple[pd.Timestamp, ...]
    _bottom_series: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError("M5Dataset must be created with load_m5_dataset()")

    @classmethod
    def _from_validated(
        cls,
        *,
        config: M5ProtocolConfig,
        input_inventory_sha256: str,
        sales: pd.DataFrame,
        dates: tuple[pd.Timestamp, ...],
    ) -> Self:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_config", config)
        object.__setattr__(instance, "_input_inventory_sha256", input_inventory_sha256)
        object.__setattr__(instance, "_sales", sales.copy(deep=True))
        object.__setattr__(instance, "_dates", dates)
        object.__setattr__(instance, "_bottom_series", tuple(sales["series_key"]))
        return instance

    @property
    def config(self) -> M5ProtocolConfig:
        """Return the immutable configuration that selected this dataset."""
        return self._config

    @property
    def input_inventory_sha256(self) -> str:
        """Return the exact input-inventory identity verified by the loader."""
        return self._input_inventory_sha256

    @property
    def sales(self) -> pd.DataFrame:
        """Return an isolated selected wide-sales snapshot."""
        return self._sales.copy(deep=True)

    @property
    def dates(self) -> tuple[pd.Timestamp, ...]:
        """Return dates mapped positionally to the consumed evaluation days."""
        return self._dates

    @property
    def history_end(self) -> pd.Timestamp:
        """Return the date mapped to the final consumed evaluation sales day."""
        return self._dates[-1]

    @property
    def bottom_series(self) -> tuple[str, ...]:
        """Return selected bottom labels in canonical UTF-8 byte order."""
        return self._bottom_series


def load_m5_dataset(
    project_root: Path,
    config: M5ProtocolConfig,
) -> M5Dataset:
    """Verify configured inputs, validate the full release, then select population."""
    if not isinstance(project_root, Path):
        raise M5DataError("project root must be a pathlib.Path")
    if not isinstance(config, M5ProtocolConfig):
        raise M5DataError("config must be an M5ProtocolConfig")
    if config.phase != "evaluation":
        raise M5DataError("M5 loader requires the declared evaluation phase")
    data_directory = project_root / config.data_dir
    inventory_path = project_root / config.inventory_path
    try:
        inventory = verify_m5_inputs(data_directory, inventory_path)
        calendar = _read_csv(data_directory, inventory, _CALENDAR_NAME)
        dates = _normalize_calendar(calendar)
        sales = _read_csv(data_directory, inventory, _SALES_NAME)
    except M5InputError as error:
        raise M5DataError(str(error)) from error

    normalized_sales = _normalize_sales(sales)
    if len(dates) < _EVALUATION_DAY_COUNT:
        raise M5DataError("calendar does not map every consumed evaluation day")
    consumed_dates = dates[:_EVALUATION_DAY_COUNT]
    selected = _select_population(normalized_sales, config=config)
    return M5Dataset._from_validated(
        config=config,
        input_inventory_sha256=inventory.content_sha256,
        sales=selected,
        dates=consumed_dates,
    )


def _read_csv(
    data_directory: Path,
    inventory: M5InputInventory,
    name: str,
) -> pd.DataFrame:
    try:
        payload = read_verified_m5_input(data_directory, name, inventory)
        return pd.read_csv(
            io.BytesIO(payload),
            dtype={column: "string" for column in _SOURCE_FACTS},
            low_memory=False,
        )
    except M5InputError:
        raise
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        raise M5DataError(f"{name}: CSV parse failed") from error


def _normalize_sales(frame: pd.DataFrame) -> pd.DataFrame:
    _require_frame(frame, surface="evaluation sales")
    expected = (*_SOURCE_FACTS, *_DAY_COLUMNS)
    if tuple(frame.columns) != expected:
        raise M5DataError(
            "evaluation sales columns must contain exact metadata followed by day labels "
            "d_1..d_1941 in positional order"
        )
    normalized = frame.copy(deep=True)
    for attribute in _SOURCE_FACTS:
        normalized[attribute] = _hierarchy_strings(
            normalized[attribute],
            attribute=attribute,
        )
    pairs = list(zip(normalized["item_id"], normalized["store_id"], strict=True))
    if len(set(pairs)) != len(pairs):
        raise M5DataError("evaluation sales contain a duplicate (item_id, store_id) identity")

    bottom = [f"{item}_{store}" for item, store in pairs]
    if len(set(bottom)) != len(bottom):
        raise M5DataError("evaluation sales bottom label collision after item/store rendering")
    normalized["series_key"] = pd.Series(bottom, index=normalized.index, dtype="string")

    for day in _DAY_COLUMNS:
        normalized[day] = _integral_sales(normalized[day], day=day)
    order = sorted(range(len(normalized)), key=lambda index: bottom[index].encode("utf-8"))
    columns = ["series_key", *_SOURCE_FACTS, *_DAY_COLUMNS]
    return normalized.iloc[order].reset_index(drop=True).loc[:, columns]


def _normalize_calendar(frame: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
    _require_frame(frame, surface="calendar")
    if "date" not in frame:
        raise M5DataError("calendar must contain a date column")
    dates = tuple(_calendar_date(value) for value in frame["date"])
    if len(set(dates)) != len(dates):
        raise M5DataError("calendar dates must be unique")

    if "d" in frame:
        raw_labels = tuple(frame["d"])
        if any(not isinstance(value, str) or not value for value in raw_labels):
            raise M5DataError("calendar day labels must be non-missing strings")
        if len(set(raw_labels)) != len(raw_labels):
            raise M5DataError("calendar day labels must be unique")
        expected = tuple(f"d_{index}" for index in range(1, len(raw_labels) + 1))
        if raw_labels != expected:
            raise M5DataError("calendar day labels must be positional d_1..d_N")
        _require_contiguous_dates(dates, surface="calendar")
        return dates

    sorted_dates = tuple(sorted(dates))
    _require_contiguous_dates(sorted_dates, surface="label-less calendar")
    return sorted_dates


def _select_population(frame: pd.DataFrame, *, config: M5ProtocolConfig) -> pd.DataFrame:
    population = config.population
    if population.kind == "full":
        return frame
    count = population.bottom_count
    salt = population.salt
    if count is None or salt is None:
        raise M5DataError("digest_rank population is incomplete")
    if count > len(frame):
        raise M5DataError(
            f"population bottom_count {count} exceeds validated release size {len(frame)}"
        )
    ranked = sorted(
        tuple(frame["series_key"]),
        key=lambda series_key: (_digest_rank(salt, series_key), series_key.encode("utf-8")),
    )
    selected = set(ranked[:count])
    return frame[frame["series_key"].isin(selected)].reset_index(drop=True)


def _digest_rank(salt: str, series_key: str) -> bytes:
    try:
        preimage = canonical_json_bytes(
            {"salt": salt, "series_key": series_key},
            path="M5 population digest rank",
        )
    except CanonicalJsonError as error:
        raise M5DataError(str(error)) from error
    return hashlib.sha256(preimage).digest()


def _integral_sales(series: pd.Series, *, day: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as error:
        raise M5DataError(
            f"evaluation sales {day} values must be finite integral counts"
        ) from error
    if numeric.isna().any():
        raise M5DataError(f"evaluation sales {day} values must be finite integral counts")
    if is_integer_dtype(numeric.dtype):
        if (numeric < 0).any():
            raise M5DataError(f"evaluation sales {day} values must be non-negative")
        if (numeric > _SIGNED_INT64_MAX).any():
            raise M5DataError(f"evaluation sales {day} values exceed signed 64-bit counts")
        return numeric.astype("int64")

    values = numeric.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise M5DataError(f"evaluation sales {day} values must be finite integral counts")
    if (values < 0.0).any():
        raise M5DataError(f"evaluation sales {day} values must be non-negative")
    if (values >= _SIGNED_INT64_BOUND).any():
        raise M5DataError(f"evaluation sales {day} values exceed signed 64-bit counts")
    if (values >= _FLOAT_INTEGER_AMBIGUITY_BOUND).any():
        raise M5DataError(f"evaluation sales {day} values cannot be validated exactly")
    return pd.Series(values, index=series.index, name=series.name, dtype="int64")


def _hierarchy_strings(series: pd.Series, *, attribute: str) -> pd.Series:
    if not isinstance(series.dtype, pd.StringDtype) or series.isna().any():
        raise M5DataError(f"hierarchy attribute {attribute!r} must be present for every bottom")
    values = series.tolist()
    if any(not isinstance(value, str) or not value for value in values):
        raise M5DataError(f"hierarchy attribute {attribute!r} must be a non-empty string")
    try:
        for value in values:
            value.encode("utf-8")
    except UnicodeError as error:
        raise M5DataError(f"hierarchy attribute {attribute!r} must be valid UTF-8") from error
    return pd.Series(values, index=series.index, name=series.name, dtype="string")


def _calendar_date(value: object) -> pd.Timestamp:
    if not isinstance(value, str):
        raise M5DataError("calendar dates must be valid timezone-naive dates")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise M5DataError("calendar dates must be valid timezone-naive dates") from error
    if pd.isna(timestamp) or timestamp.tz is not None or timestamp != timestamp.normalize():
        raise M5DataError("calendar dates must be valid timezone-naive dates")
    return timestamp


def _require_contiguous_dates(dates: tuple[pd.Timestamp, ...], *, surface: str) -> None:
    if not dates:
        raise M5DataError(f"{surface} must contain at least one date")
    for previous, current in zip(dates, dates[1:], strict=False):
        if current != previous + pd.Timedelta(days=1):
            raise M5DataError(f"{surface} dates must form one contiguous daily sequence")


def _require_frame(frame: pd.DataFrame, *, surface: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise M5DataError(f"{surface} must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise M5DataError(f"{surface} columns must be unique")
    if any(not isinstance(column, str) for column in frame.columns):
        raise M5DataError(f"{surface} column labels must be strings")
    if frame.empty:
        raise M5DataError(f"{surface} must not be empty")
    if any(is_bool_dtype(frame[column].dtype) for column in frame.columns):
        raise M5DataError(f"{surface} cannot contain boolean columns")
