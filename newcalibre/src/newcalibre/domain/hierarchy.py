"""Compile hierarchy facts and construct coherent aggregate history."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral, Real
from typing import Any, Final, cast
from urllib.parse import quote

import numpy as np
import pandas as pd

from newcalibre.domain.calendar import Calendar
from newcalibre.domain.forecast_frame import SERIES_KEY
from newcalibre.domain.panel import (
    CENSOR_STATUS,
    OBSERVED_VALUE,
    TIMESTAMP,
    UNDECLARED_CENSORING,
    Panel,
)

TOTAL_NODE_LABEL: Final = "__total__"
AGGREGATE_NODE_PREFIX: Final = "__aggregate__"


class HierarchyError(ValueError):
    """Report invalid hierarchy facts or an incompatible history panel."""


class HierarchyNodeKind(StrEnum):
    """Name the three node families in an aggregation lattice."""

    BOTTOM = "bottom"
    AGGREGATE = "aggregate"
    TOTAL = "total"


@dataclass(frozen=True, slots=True)
class HierarchyNode:
    """Describe one immutable lattice node and its expected bottom members."""

    label: str
    kind: HierarchyNodeKind
    members: tuple[str, ...]
    expected_member_count: int


@dataclass(frozen=True, slots=True)
class _AttributeValue:
    tag: str
    payload: str
    order_key: tuple[int, object]

    @property
    def label_token(self) -> str:
        return f"{self.tag}:{quote(self.payload, safe='')}"


@dataclass(frozen=True, slots=True, init=False)
class HierarchyIndex:
    """Own the validated, canonical aggregation lattice for one run."""

    _bottom_series: tuple[str, ...]
    _attribute_names: tuple[str, ...]
    _nodes: tuple[HierarchyNode, ...]
    _calendar: Calendar = field(repr=False)

    @classmethod
    def from_facts(
        cls,
        facts: pd.DataFrame,
        *,
        bottom_series: Iterable[str],
        calendar: Calendar,
    ) -> HierarchyIndex:
        """Compile per-bottom hierarchy facts into a deterministic lattice."""
        bottom = _canonical_bottom_series(bottom_series)
        if not isinstance(calendar, Calendar) or calendar.phase is None:
            raise HierarchyError("hierarchy calendar must be a bound Calendar")
        normalized, attributes = _normalize_facts(facts, bottom=bottom)

        nodes: list[HierarchyNode] = [
            HierarchyNode(
                label=series_key,
                kind=HierarchyNodeKind.BOTTOM,
                members=(series_key,),
                expected_member_count=1,
            )
            for series_key in bottom
        ]
        generated_labels: set[str] = set()
        for attribute in attributes:
            groups: dict[_AttributeValue, list[str]] = {}
            for series_key, value in zip(
                normalized[SERIES_KEY], normalized[attribute], strict=True
            ):
                groups.setdefault(value, []).append(series_key)
            ordered_groups = sorted(groups.items(), key=lambda item: item[0].order_key)
            for value, member_list in ordered_groups:
                members = tuple(sorted(member_list, key=str.encode))
                label = _aggregate_label(attribute, value)
                if label in generated_labels:
                    raise HierarchyError(f"generated hierarchy label collision: {label!r}")
                generated_labels.add(label)
                nodes.append(
                    HierarchyNode(
                        label=label,
                        kind=HierarchyNodeKind.AGGREGATE,
                        members=members,
                        expected_member_count=len(members),
                    )
                )

        generated_labels.add(TOTAL_NODE_LABEL)
        bottom_collisions = sorted(set(bottom) & generated_labels, key=str.encode)
        if bottom_collisions:
            raise HierarchyError(
                f"hierarchy node labels collide with bottom series keys: {bottom_collisions}"
            )
        nodes.append(
            HierarchyNode(
                label=TOTAL_NODE_LABEL,
                kind=HierarchyNodeKind.TOTAL,
                members=bottom,
                expected_member_count=len(bottom),
            )
        )

        instance = object.__new__(cls)
        object.__setattr__(instance, "_bottom_series", bottom)
        object.__setattr__(instance, "_attribute_names", attributes)
        object.__setattr__(instance, "_nodes", tuple(nodes))
        object.__setattr__(instance, "_calendar", calendar)
        return instance

    @property
    def bottom_series(self) -> tuple[str, ...]:
        """Return bottom labels in canonical UTF-8 byte order."""
        return self._bottom_series

    @property
    def attribute_names(self) -> tuple[str, ...]:
        """Return hierarchy attribute names in canonical UTF-8 byte order."""
        return self._attribute_names

    @property
    def nodes(self) -> tuple[HierarchyNode, ...]:
        """Return bottom, aggregate, and total nodes in lattice order."""
        return self._nodes

    @property
    def node_labels(self) -> tuple[str, ...]:
        """Return every collision-free node label in lattice order."""
        return tuple(node.label for node in self._nodes)

    def expand_history(self, panel: Panel) -> Panel:
        """Return bottom history plus coherent aggregate and total rows.

        Aggregate values are defined only where every expected bottom member
        has a non-missing observation. Censoring, availability, and exogenous
        facts are deliberately not aggregated: aggregate rows record
        undeclared censoring and missing numeric metadata.
        """
        if not isinstance(panel, Panel):
            raise HierarchyError("history must be a Panel")
        if not self._calendar.shares_grid_with(panel.calendar):
            raise HierarchyError("history calendar is incompatible with the hierarchy calendar")
        present = set(panel.series_keys)
        expected = set(self._bottom_series)
        if present != expected:
            missing = sorted(expected - present, key=str.encode)
            unexpected = sorted(present - expected, key=str.encode)
            raise HierarchyError(
                "history must contain exactly the hierarchy bottom series; "
                f"missing={missing}, unexpected={unexpected}"
            )

        source = panel.frame
        timestamps = tuple(sorted(pd.Timestamp(value) for value in source[TIMESTAMP].unique()))
        timestamp_positions = {timestamp: index for index, timestamp in enumerate(timestamps)}
        aggregate_nodes = self._nodes[len(self._bottom_series) :]
        aggregate_values = np.empty((len(aggregate_nodes), len(timestamps)), dtype=np.float64)
        for timestamp, rows in source.groupby(TIMESTAMP, sort=False, observed=True):
            normalized_timestamp = cast(pd.Timestamp, timestamp)
            values = dict(zip(rows[SERIES_KEY], rows[OBSERVED_VALUE], strict=True))
            timestamp_position = timestamp_positions[normalized_timestamp]
            for node_position, node in enumerate(aggregate_nodes):
                aggregate_values[node_position, timestamp_position] = _coherent_sum(
                    values,
                    node=node,
                    timestamp=normalized_timestamp,
                )

        aggregate_labels = (
            pd.Series(
                [node.label for node in aggregate_nodes],
                dtype=source[SERIES_KEY].dtype,
            )
            .repeat(len(timestamps))
            .reset_index(drop=True)
        )
        timestamp_values = pd.Series(timestamps, dtype=source[TIMESTAMP].dtype)
        aggregate_timestamps = pd.concat(
            [timestamp_values] * len(aggregate_nodes),
            ignore_index=True,
        )
        aggregate_value_series = pd.Series(
            aggregate_values.reshape(-1),
            dtype="float64",
        )

        expanded = _append_aggregate_rows(
            source,
            labels=aggregate_labels,
            timestamps=aggregate_timestamps,
            values=aggregate_value_series,
        )
        return Panel.from_frame(expanded, calendar=panel.calendar)


def _canonical_bottom_series(bottom_series: Iterable[str]) -> tuple[str, ...]:
    if isinstance(bottom_series, (str, bytes)):
        raise HierarchyError("bottom_series must be an iterable of series keys")
    try:
        values = tuple(bottom_series)
    except TypeError as error:
        raise HierarchyError("bottom_series must be an iterable of series keys") from error
    if not values:
        raise HierarchyError("hierarchy requires at least one bottom series")
    for value in values:
        if not isinstance(value, str) or not value:
            raise HierarchyError("bottom series keys must be non-empty strings")
        _require_utf8(value, name="bottom series key")
    if len(set(values)) != len(values):
        raise HierarchyError("bottom series keys must be unique")
    return tuple(sorted(values, key=str.encode))


def _normalize_facts(
    facts: pd.DataFrame, *, bottom: tuple[str, ...]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if not isinstance(facts, pd.DataFrame):
        raise HierarchyError("hierarchy facts must be a pandas DataFrame")
    if facts.columns.has_duplicates:
        raise HierarchyError("hierarchy facts have duplicate column labels")
    for column in facts.columns:
        if not isinstance(column, str):
            raise HierarchyError("hierarchy fact column labels must be strings")
        _require_utf8(column, name="hierarchy fact column label")
    if SERIES_KEY not in facts.columns:
        raise HierarchyError(f"hierarchy facts are missing required column {SERIES_KEY!r}")
    attributes = tuple(
        sorted((column for column in facts.columns if column != SERIES_KEY), key=str.encode)
    )
    if not attributes:
        raise HierarchyError("hierarchy facts require at least one attribute column")

    normalized = facts.loc[:, [SERIES_KEY, *attributes]].copy(deep=True)
    keys = [_normalize_fact_key(value) for value in normalized[SERIES_KEY]]
    if len(set(keys)) != len(keys):
        raise HierarchyError("hierarchy fact series keys collide after string normalization")
    normalized[SERIES_KEY] = keys
    present = set(keys)
    expected = set(bottom)
    if present != expected:
        missing = sorted(expected - present, key=str.encode)
        extra = sorted(present - expected, key=str.encode)
        raise HierarchyError(
            f"hierarchy facts must cover bottom series exactly; missing={missing}, extra={extra}"
        )
    for attribute in attributes:
        normalized[attribute] = pd.Series(
            [
                _canonical_attribute_value(value, attribute=attribute)
                for value in normalized[attribute]
            ],
            index=normalized.index,
            dtype="object",
        )
    normalized = normalized.sort_values(SERIES_KEY, key=lambda column: column.map(str.encode))
    return normalized.reset_index(drop=True), attributes


def _normalize_fact_key(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise HierarchyError("hierarchy fact series keys must be non-empty strings")
    _require_utf8(value, name="hierarchy fact series key")
    return value


def _canonical_attribute_value(value: object, *, attribute: str) -> _AttributeValue:
    if not pd.api.types.is_scalar(value):
        raise HierarchyError(
            f"hierarchy attribute {attribute!r} must use a string, boolean, "
            "integer, or float scalar"
        )
    if _is_missing(value):
        raise HierarchyError(f"hierarchy attribute {attribute!r} cannot be missing")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        _require_utf8(value, name=f"hierarchy attribute {attribute!r} value")
        return _AttributeValue("s", value, (3, value.encode()))
    if isinstance(value, bool):
        payload = "true" if value else "false"
        return _AttributeValue("b", payload, (0, value))
    if isinstance(value, Integral):
        normalized = int(value)
        return _AttributeValue("i", str(normalized), (1, normalized))
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise HierarchyError(f"hierarchy attribute {attribute!r} must be finite")
        if normalized == 0.0:
            normalized = 0.0
        return _AttributeValue("f", normalized.hex(), (2, normalized))
    raise HierarchyError(
        f"hierarchy attribute {attribute!r} must use a string, boolean, integer, or float scalar"
    )


def _aggregate_label(attribute: str, value: _AttributeValue) -> str:
    return f"{AGGREGATE_NODE_PREFIX}:{quote(attribute, safe='')}:{value.label_token}"


def _coherent_sum(
    observations: dict[str, Any],
    *,
    node: HierarchyNode,
    timestamp: pd.Timestamp,
) -> float:
    operands: list[float] = []
    for member in node.members:
        if member not in observations or _is_missing(observations[member]):
            return math.nan
        try:
            operands.append(float(observations[member]))
        except (OverflowError, TypeError, ValueError) as error:
            raise HierarchyError(
                f"cannot aggregate node {node.label!r} at {timestamp!s} as float"
            ) from error
    try:
        return math.fsum(operands)
    except ValueError as error:
        raise HierarchyError(
            f"cannot aggregate non-finite values for node {node.label!r} at {timestamp!s}"
        ) from error


def _append_aggregate_rows(
    source: pd.DataFrame,
    *,
    labels: pd.Series,
    timestamps: pd.Series,
    values: pd.Series,
) -> pd.DataFrame:
    row_count = len(labels)
    columns: dict[str, pd.Series] = {}
    for column in source.columns:
        head = source[column]
        if column == SERIES_KEY:
            tail = labels
        elif column == TIMESTAMP:
            tail = timestamps
        elif column == OBSERVED_VALUE:
            head = head.astype("float64")
            tail = values
        elif column == CENSOR_STATUS:
            tail = pd.Series(
                [UNDECLARED_CENSORING] * row_count,
                dtype=head.dtype,
            )
        else:
            head = _numeric_with_missing_capacity(head)
            tail = pd.Series([pd.NA] * row_count, dtype=head.dtype)
        columns[column] = pd.concat([head, tail], ignore_index=True)
    return pd.DataFrame(columns)


def _numeric_with_missing_capacity(series: pd.Series) -> pd.Series:
    dtype = series.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in {"i", "u"}:
        nullable_types = {
            ("i", 1): pd.Int8Dtype(),
            ("i", 2): pd.Int16Dtype(),
            ("i", 4): pd.Int32Dtype(),
            ("i", 8): pd.Int64Dtype(),
            ("u", 1): pd.UInt8Dtype(),
            ("u", 2): pd.UInt16Dtype(),
            ("u", 4): pd.UInt32Dtype(),
            ("u", 8): pd.UInt64Dtype(),
        }
        return series.astype(nullable_types[(dtype.kind, dtype.itemsize)])
    return series.copy(deep=True)


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    return False


def _require_utf8(value: str, *, name: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise HierarchyError(f"{name} must be valid UTF-8") from error
