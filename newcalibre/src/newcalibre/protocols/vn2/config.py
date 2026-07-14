"""Parse VN2 protocol constants as strict, immutable configuration data."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from newcalibre.domain import (
    ActualsSemantics,
    Calendar,
    CalendarError,
    CostStructure,
    DecisionTiming,
    StockoutRule,
)
from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.protocols.vn2._constants import VN2_SEASONAL_NAIVE_BACKEND

_TOP_LEVEL_KEYS = frozenset(
    {
        "actuals_semantics",
        "calendar_frequency",
        "columns",
        "conformal_config",
        "cost",
        "dataset",
        "decision",
        "files",
        "history",
        "model_config",
        "ordering_policy",
        "schema",
        "series_count",
        "stockout_rule",
    }
)
_HISTORY_KEYS = frozenset({"first_week", "initial_last_week", "initial_periods"})
_DECISION_KEYS = frozenset(
    {
        "drain_periods",
        "lead_time",
        "origins",
        "protection_period",
        "review_period",
        "round_count",
        "task_horizon",
    }
)
_COST_KEYS = frozenset(
    {
        "currency",
        "holding_rate",
        "overage_rate",
        "shortage_rate",
        "underage_rate",
    }
)
_FILE_KEYS = frozenset({"in_stock", "initial_state", "master", "sales_reveals"})
_COLUMN_KEYS = frozenset(
    {
        "initial_on_hand",
        "initial_pipeline",
        "initial_state_columns",
        "master_attributes",
        "series_keys",
    }
)
_MODEL_KEYS = frozenset({"backend", "censoring_aware", "m", "model_name", "quantile_levels"})
_ORDERING_KEYS = frozenset(
    {
        "coverage",
        "explicit_decision_fractile",
        "name",
        "quantile",
        "reorder_point",
        "reorder_point_scale",
        "target_cap",
        "target_floor",
        "target_scale",
    }
)


class VN2ConfigError(ValueError):
    """Report an incomplete, ambiguous, or inconsistent VN2 configuration."""


class _DuplicateYAMLKey(yaml.YAMLError):
    """Retain one duplicate mapping key refused during YAML construction."""

    def __init__(self, key: object) -> None:
        super().__init__(key)
        self.key = key


class _UniqueKeyLoader(yaml.SafeLoader):
    """Construct safe YAML while refusing duplicate mappings recursively."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise _DuplicateYAMLKey(key)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class VN2HistoryConfig:
    """Describe the round-zero weekly history shape."""

    first_week: pd.Timestamp
    initial_last_week: pd.Timestamp
    initial_periods: int


@dataclass(frozen=True, slots=True)
class VN2FileConfig:
    """Name every challenge-distribution input without owning its bytes."""

    sales_reveals: tuple[str, ...]
    master: str
    in_stock: str
    initial_state: str

    @property
    def all_names(self) -> frozenset[str]:
        """Return the exact configured input file set."""
        return frozenset((*self.sales_reveals, self.master, self.in_stock, self.initial_state))


@dataclass(frozen=True, slots=True)
class VN2ColumnConfig:
    """Map challenge-distribution columns to chapter-20 facts."""

    series_keys: tuple[str, str]
    master_attributes: tuple[str, ...]
    initial_state_columns: tuple[str, ...]
    initial_on_hand: str
    initial_pipeline: tuple[str, ...]


@dataclass(frozen=True, slots=True, init=False)
class VN2ProtocolConfig:
    """Carry one fully validated VN2 protocol configuration."""

    dataset: str
    series_count: int
    calendar: Calendar
    history: VN2HistoryConfig
    round_count: int
    timing: DecisionTiming
    task_horizon: int
    drain_periods: int
    decision_origins: tuple[pd.Timestamp, ...]
    realized_periods: tuple[pd.Timestamp, ...]
    currency: str
    cost_structure: CostStructure
    actuals_semantics: ActualsSemantics
    stockout_rule: StockoutRule
    files: VN2FileConfig
    columns: VN2ColumnConfig
    conformal_config: None
    _model_config_json: bytes = field(repr=False)
    _ordering_policy_json: bytes = field(repr=False)

    def __init__(self) -> None:
        raise TypeError("VN2ProtocolConfig must be created with load_vn2_config()")

    @property
    def holding_rate(self) -> float:
        """Return the protocol holding rate from the authoritative cost structure."""
        return self.cost_structure.holding

    @property
    def shortage_rate(self) -> float:
        """Return the protocol shortage rate from the authoritative cost structure."""
        return self.cost_structure.shortage

    @property
    def model_config(self) -> dict[str, object]:
        """Return an isolated adapter configuration snapshot."""
        return cast(dict[str, object], json.loads(self._model_config_json))

    @property
    def ordering_policy(self) -> dict[str, object]:
        """Return an isolated explicit-quantile (R,S) configuration snapshot."""
        return cast(dict[str, object], json.loads(self._ordering_policy_json))


def load_vn2_config(path: Path) -> VN2ProtocolConfig:
    """Load a strict YAML protocol instance without executing engine behavior."""
    if not isinstance(path, Path):
        raise VN2ConfigError("configuration path must be a pathlib.Path")
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except _DuplicateYAMLKey as error:
        raise VN2ConfigError(
            f"VN2 configuration contains duplicate YAML key {error.key!r}"
        ) from error
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise VN2ConfigError(f"VN2 configuration is unreadable: {path}") from error
    top = _exact_mapping(raw, keys=_TOP_LEVEL_KEYS, surface="top-level configuration")
    if top["schema"] != 1:
        raise VN2ConfigError("configuration schema must equal 1")
    if top["dataset"] != "vn2":
        raise VN2ConfigError("configuration dataset must equal 'vn2'")
    series_count = _positive_integer(top["series_count"], name="series_count")

    frequency = _string(top["calendar_frequency"], name="calendar_frequency")
    if frequency != "W-MON":
        raise VN2ConfigError("VN2 calendar_frequency must be the Monday anchor 'W-MON'")
    calendar, history = _parse_history(top["history"], frequency=frequency)
    decision = _parse_decision(top["decision"], calendar=calendar, history=history)
    round_count, timing, horizon, drains, origins, realized_periods, expected_reveals = decision
    currency, cost_structure = _parse_cost(top["cost"])

    try:
        actuals_semantics = ActualsSemantics(top["actuals_semantics"])
    except (TypeError, ValueError) as error:
        raise VN2ConfigError("actuals_semantics must equal 'censored_sales_surrogate'") from error
    if actuals_semantics is not ActualsSemantics.CENSORED_SALES_SURROGATE:
        raise VN2ConfigError("actuals_semantics must equal 'censored_sales_surrogate'")
    try:
        stockout_rule = StockoutRule(top["stockout_rule"])
    except (TypeError, ValueError) as error:
        raise VN2ConfigError("VN2 stockout_rule must equal 'lost-sales'") from error
    if stockout_rule is not StockoutRule.LOST_SALES:
        raise VN2ConfigError("VN2 stockout_rule must equal 'lost-sales'")

    files = _parse_files(
        top["files"],
        expected_reveals=expected_reveals,
    )
    columns = _parse_columns(top["columns"], lead_time=timing.lead_time)
    model = _parse_model_config(top["model_config"])
    if top["conformal_config"] is not None:
        raise VN2ConfigError("Gate-A conformal_config must be explicit null")
    ordering = _parse_ordering_policy(top["ordering_policy"])

    config = object.__new__(VN2ProtocolConfig)
    object.__setattr__(config, "dataset", "vn2")
    object.__setattr__(config, "series_count", series_count)
    object.__setattr__(config, "calendar", calendar)
    object.__setattr__(config, "history", history)
    object.__setattr__(config, "round_count", round_count)
    object.__setattr__(config, "timing", timing)
    object.__setattr__(config, "task_horizon", horizon)
    object.__setattr__(config, "drain_periods", drains)
    object.__setattr__(config, "decision_origins", origins)
    object.__setattr__(config, "realized_periods", realized_periods)
    object.__setattr__(config, "currency", currency)
    object.__setattr__(config, "cost_structure", cost_structure)
    object.__setattr__(config, "actuals_semantics", actuals_semantics)
    object.__setattr__(config, "stockout_rule", stockout_rule)
    object.__setattr__(config, "files", files)
    object.__setattr__(config, "columns", columns)
    object.__setattr__(config, "conformal_config", None)
    object.__setattr__(config, "_model_config_json", _canonical_json(model))
    object.__setattr__(config, "_ordering_policy_json", _canonical_json(ordering))
    return config


def _parse_history(value: object, *, frequency: str) -> tuple[Calendar, VN2HistoryConfig]:
    payload = _exact_mapping(value, keys=_HISTORY_KEYS, surface="history")
    first = _iso_timestamp(payload["first_week"], name="history first_week")
    try:
        calendar = Calendar(frequency).bind(first)
    except CalendarError as error:
        raise VN2ConfigError(
            "history first_week must lie on the configured Monday calendar"
        ) from error
    last = _timestamp(
        payload["initial_last_week"],
        name="history initial_last_week",
        calendar=calendar,
    )
    periods = _positive_integer(payload["initial_periods"], name="history initial_periods")
    expected_last = _advance_calendar(
        calendar,
        first,
        periods - 1,
        field="history initial_periods",
    )
    if expected_last != last:
        raise VN2ConfigError(
            "history first_week, initial_periods, and initial_last_week must describe one cadence"
        )
    return calendar, VN2HistoryConfig(
        first_week=first,
        initial_last_week=last,
        initial_periods=periods,
    )


def _parse_decision(
    value: object,
    *,
    calendar: Calendar,
    history: VN2HistoryConfig,
) -> tuple[
    int,
    DecisionTiming,
    int,
    int,
    tuple[pd.Timestamp, ...],
    tuple[pd.Timestamp, ...],
    int,
]:
    payload = _exact_mapping(value, keys=_DECISION_KEYS, surface="decision")
    rounds = _positive_integer(payload["round_count"], name="decision round_count")
    lead_time = _positive_integer(payload["lead_time"], name="decision lead_time")
    review_period = _positive_integer(payload["review_period"], name="decision review_period")
    timing = DecisionTiming(lead_time=lead_time, review_period=review_period)
    protection = _positive_integer(payload["protection_period"], name="decision protection_period")
    if protection != timing.protection_period:
        raise VN2ConfigError("decision protection_period must equal lead_time + review_period")
    horizon = _positive_integer(payload["task_horizon"], name="decision task_horizon")
    if horizon < protection:
        raise VN2ConfigError("decision task_horizon must cover the protection period")
    drains = _nonnegative_integer(payload["drain_periods"], name="decision drain_periods")
    if drains < lead_time:
        raise VN2ConfigError("decision drain_periods must expose the complete lead-time pipeline")

    raw_origins = payload["origins"]
    if not isinstance(raw_origins, list) or len(raw_origins) != rounds:
        count = len(raw_origins) if isinstance(raw_origins, list) else "non-list"
        raise VN2ConfigError(
            f"decision origins must contain round_count={rounds} values, found {count}"
        )
    origins = tuple(
        _timestamp(raw, name=f"decision origin {index + 1}", calendar=calendar)
        for index, raw in enumerate(raw_origins)
    )
    expected_first = _advance_calendar(
        calendar,
        history.initial_last_week,
        review_period,
        field="decision review_period",
    )
    if origins[0] != expected_first:
        raise VN2ConfigError("first decision origin must follow the round-zero reveal cadence")
    try:
        expected = tuple(
            calendar.advance(expected_first, index * review_period) for index in range(rounds)
        )
    except CalendarError as error:
        raise VN2ConfigError("decision origins exceed configured calendar bounds") from error
    if origins != expected:
        raise VN2ConfigError("decision origins must follow the configured review cadence")

    realized_end = _advance_calendar(
        calendar,
        origins[-1],
        review_period + drains - 1,
        field="decision review_period and drain_periods",
    )
    realized_periods = _weekly_periods(
        calendar,
        start=origins[0],
        end=realized_end,
        field="decision review_period and drain_periods",
    )
    required_end = _advance_calendar(
        calendar,
        origins[-1],
        max(horizon, review_period + drains) - 1,
        field="decision task_horizon and drain_periods",
    )
    first_post_history = _advance_calendar(
        calendar,
        history.initial_last_week,
        1,
        field="history initial_last_week",
    )
    required_sales_periods = _weekly_periods(
        calendar,
        start=first_post_history,
        end=required_end,
        field="decision task_horizon and origins",
    )
    expected_reveals = len(required_sales_periods) + 1
    return rounds, timing, horizon, drains, origins, realized_periods, expected_reveals


def _advance_calendar(
    calendar: Calendar,
    timestamp: pd.Timestamp,
    periods: int,
    *,
    field: str,
) -> pd.Timestamp:
    try:
        return calendar.advance(timestamp, periods)
    except CalendarError as error:
        raise VN2ConfigError(f"{field} exceeds configured calendar bounds") from error


def _weekly_periods(
    calendar: Calendar,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    field: str,
) -> tuple[pd.Timestamp, ...]:
    if end < start:
        raise VN2ConfigError(f"{field} must describe a non-empty weekly horizon")
    periods = [start]
    while periods[-1] != end:
        periods.append(_advance_calendar(calendar, periods[-1], 1, field=field))
    return tuple(periods)


def _parse_cost(value: object) -> tuple[str, CostStructure]:
    payload = _exact_mapping(value, keys=_COST_KEYS, surface="cost")
    currency = _string(payload["currency"], name="cost currency")
    underage = _nonnegative_real(payload["underage_rate"], name="underage_rate")
    overage = _nonnegative_real(payload["overage_rate"], name="overage_rate")
    holding = _nonnegative_real(payload["holding_rate"], name="holding_rate")
    shortage = _nonnegative_real(payload["shortage_rate"], name="shortage_rate")
    if underage != shortage or overage != holding:
        raise VN2ConfigError(
            "underage/overage rates must explicitly match shortage/holding protocol rates"
        )
    return currency, CostStructure(underage, overage, holding, shortage)


def _parse_files(value: object, *, expected_reveals: int) -> VN2FileConfig:
    payload = _exact_mapping(value, keys=_FILE_KEYS, surface="files")
    raw_sales = payload["sales_reveals"]
    if not isinstance(raw_sales, list) or len(raw_sales) != expected_reveals:
        raise VN2ConfigError(
            f"files sales_reveals must contain exactly {expected_reveals} reveal names"
        )
    sales = tuple(_safe_file_name(name, surface="sales reveal") for name in raw_sales)
    master = _safe_file_name(payload["master"], surface="master")
    in_stock = _safe_file_name(payload["in_stock"], surface="in_stock")
    initial = _safe_file_name(payload["initial_state"], surface="initial_state")
    names = (*sales, master, in_stock, initial)
    if len(set(names)) != len(names):
        raise VN2ConfigError("files names must be unique")
    return VN2FileConfig(sales, master, in_stock, initial)


def _parse_columns(value: object, *, lead_time: int) -> VN2ColumnConfig:
    payload = _exact_mapping(value, keys=_COLUMN_KEYS, surface="columns")
    keys = _string_sequence(payload["series_keys"], name="columns series_keys")
    if keys != ("Store", "Product"):
        raise VN2ConfigError("columns series_keys must equal ['Store', 'Product']")
    attributes = _string_sequence(payload["master_attributes"], name="columns master_attributes")
    if len(attributes) != 6:
        raise VN2ConfigError("columns master_attributes must contain exactly six names")
    state_columns = _string_sequence(
        payload["initial_state_columns"], name="columns initial_state_columns"
    )
    on_hand = _string(payload["initial_on_hand"], name="columns initial_on_hand")
    pipeline = _string_sequence(payload["initial_pipeline"], name="columns initial_pipeline")
    if len(pipeline) != lead_time:
        raise VN2ConfigError("initial pipeline column count must equal decision lead_time")
    required = {*keys, on_hand, *pipeline}
    if not required <= set(state_columns):
        raise VN2ConfigError("initial state columns omit configured on-hand or pipeline facts")
    return VN2ColumnConfig(
        series_keys=cast(tuple[str, str], keys),
        master_attributes=attributes,
        initial_state_columns=state_columns,
        initial_on_hand=on_hand,
        initial_pipeline=pipeline,
    )


def _parse_model_config(value: object) -> dict[str, object]:
    payload = _exact_mapping(value, keys=_MODEL_KEYS, surface="model_config")
    if payload["backend"] != VN2_SEASONAL_NAIVE_BACKEND:
        raise VN2ConfigError(f"model_config backend must equal {VN2_SEASONAL_NAIVE_BACKEND!r}")
    _positive_integer(payload["m"], name="model_config m")
    _string(payload["model_name"], name="model_config model_name")
    levels = payload["quantile_levels"]
    if (
        not isinstance(levels, list)
        or len(levels) != 1
        or isinstance(levels[0], bool)
        or not isinstance(levels[0], Real)
        or not math.isfinite(float(levels[0]))
        or float(levels[0]) != 0.5
    ):
        raise VN2ConfigError("model_config requires the single 0.5 quantile level")
    if payload["censoring_aware"] is not False:
        raise VN2ConfigError("model_config censoring_aware must be false for this adapter")
    return dict(payload)


def _parse_ordering_policy(value: object) -> dict[str, object]:
    payload = _exact_mapping(value, keys=_ORDERING_KEYS, surface="ordering_policy")
    if payload["name"] != "rs":
        raise VN2ConfigError("ordering_policy name must equal 'rs'")
    quantile = payload["quantile"]
    if isinstance(quantile, bool) or not isinstance(quantile, Real) or float(quantile) != 0.5:
        raise VN2ConfigError("ordering_policy quantile must equal 0.5")
    nullable = _ORDERING_KEYS - {"name", "quantile"}
    non_null = sorted(key for key in nullable if payload[key] is not None)
    if non_null:
        raise VN2ConfigError(
            "ordering_policy unused fields must be explicit null: " + ", ".join(non_null)
        )
    return dict(payload)


def _exact_mapping(
    value: object,
    *,
    keys: frozenset[str],
    surface: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise VN2ConfigError(f"{surface} must be a string-keyed mapping")
    payload = dict(cast(Mapping[str, object], value))
    actual_keys = set(payload)
    if actual_keys != keys:
        missing = sorted(keys - actual_keys)
        extra = sorted(actual_keys - keys)
        raise VN2ConfigError(f"{surface} must contain exact keys: missing={missing} extra={extra}")
    return payload


def _timestamp(value: object, *, name: str, calendar: Calendar) -> pd.Timestamp:
    timestamp = _iso_timestamp(value, name=name)
    try:
        calendar.require_member(timestamp, name=name)
    except CalendarError as error:
        raise VN2ConfigError(f"{name} must lie on the configured Monday calendar") from error
    return timestamp


def _iso_timestamp(value: object, *, name: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise VN2ConfigError(f"{name} must be an ISO date string")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise VN2ConfigError(f"{name} must be an ISO date string") from error
    if timestamp.strftime("%Y-%m-%d") != value or timestamp.tz is not None:
        raise VN2ConfigError(f"{name} must use exact timezone-naive YYYY-MM-DD form")
    return timestamp


def _safe_file_name(value: object, *, surface: str) -> str:
    name = _string(value, name=f"files {surface}")
    if Path(name).name != name or not name.endswith(".csv"):
        raise VN2ConfigError(f"files {surface} must be a safe CSV basename")
    return name


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise VN2ConfigError(f"{name} must be a non-empty list of strings")
    values = tuple(_string(item, name=name) for item in value)
    if len(set(values)) != len(values):
        raise VN2ConfigError(f"{name} must not contain duplicates")
    return values


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VN2ConfigError(f"{name} must be a non-empty trimmed string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise VN2ConfigError(f"{name} must be valid UTF-8") from error
    return value


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result < 1:
        raise VN2ConfigError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise VN2ConfigError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise VN2ConfigError(f"{name} must be a non-negative integer")
    return result


def _nonnegative_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise VN2ConfigError(f"{name} must be a finite non-negative real")
    result = float(value)
    if not math.isfinite(result):
        raise VN2ConfigError(f"{name} must be finite")
    if result < 0.0:
        raise VN2ConfigError(f"{name} must be non-negative")
    return 0.0 if result == 0.0 else result


def _canonical_json(value: Mapping[str, object]) -> bytes:
    try:
        return canonical_json_bytes(dict(value), path="VN2 configuration snapshot")
    except CanonicalJsonError as error:
        raise VN2ConfigError(str(error)) from error
