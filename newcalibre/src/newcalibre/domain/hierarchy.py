"""Compile hierarchy facts and evaluate coherent aggregate cross-sections."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real
from typing import Final
from urllib.parse import quote

import numpy as np
import pandas as pd

from newcalibre.domain.forecast_frame import SERIES_KEY

TOTAL_NODE_LABEL: Final = "__total__"
AGGREGATE_NODE_PREFIX: Final = "__aggregate__"


class HierarchyError(ValueError):
    """Report invalid hierarchy facts, nodes, or observations."""


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

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise HierarchyError("hierarchy node label must be a non-empty string")
        _require_utf8(self.label, name="hierarchy node label")
        if not isinstance(self.kind, HierarchyNodeKind):
            raise HierarchyError("hierarchy node kind must be a HierarchyNodeKind")
        if not isinstance(self.members, tuple) or not self.members:
            raise HierarchyError("hierarchy node members must be a non-empty tuple")
        for member in self.members:
            if not isinstance(member, str) or not member:
                raise HierarchyError("hierarchy node members must be non-empty strings")
            _require_utf8(member, name="hierarchy node member")
        if len(set(self.members)) != len(self.members):
            raise HierarchyError("hierarchy node members must be unique")

        has_aggregate_prefix = self.label.startswith(AGGREGATE_NODE_PREFIX)
        has_aggregate_label = self.label.startswith(f"{AGGREGATE_NODE_PREFIX}:")
        if self.kind is HierarchyNodeKind.AGGREGATE:
            if not has_aggregate_label:
                raise HierarchyError("aggregate node labels must use the aggregate prefix")
        elif has_aggregate_prefix:
            raise HierarchyError("only aggregate nodes may use the aggregate label prefix")

        is_total_label = self.label == TOTAL_NODE_LABEL
        if (self.kind is HierarchyNodeKind.TOTAL) != is_total_label:
            raise HierarchyError("the total label and total node kind must be used together")
        if self.kind is HierarchyNodeKind.BOTTOM and self.members != (self.label,):
            raise HierarchyError("bottom node label must equal its sole member")
        if self.kind is not HierarchyNodeKind.BOTTOM and self.label in self.members:
            raise HierarchyError("an aggregate or total node label cannot also be its member")
        if self.kind is not HierarchyNodeKind.BOTTOM and any(
            member == TOTAL_NODE_LABEL or member.startswith(AGGREGATE_NODE_PREFIX)
            for member in self.members
        ):
            raise HierarchyError("aggregate and total node members must be bottom series labels")

    @property
    def expected_member_count(self) -> int:
        """Return the member count derived from the immutable member tuple."""
        return len(self.members)


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

    @classmethod
    def flat(cls, bottom_series: Iterable[str]) -> HierarchyIndex:
        """Compile the canonical bottom-plus-total lattice for a flat panel."""
        bottom = _canonical_bottom_series(bottom_series)
        return _build_hierarchy_index(
            cls,
            bottom=bottom,
            attributes=(),
            bottom_nodes=_bottom_nodes(bottom),
            aggregate_nodes=(),
        )

    @classmethod
    def from_facts(
        cls,
        facts: pd.DataFrame,
        *,
        bottom_series: Iterable[str],
    ) -> HierarchyIndex:
        """Compile per-bottom hierarchy facts into a deterministic lattice."""
        bottom = _canonical_bottom_series(bottom_series)
        normalized, attributes = _normalize_facts(facts, bottom=bottom)
        bottom_nodes = _bottom_nodes(bottom)

        aggregate_nodes: list[HierarchyNode] = []
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
                aggregate_nodes.append(
                    HierarchyNode(
                        label=label,
                        kind=HierarchyNodeKind.AGGREGATE,
                        members=members,
                    )
                )

        return _build_hierarchy_index(
            cls,
            bottom=bottom,
            attributes=attributes,
            bottom_nodes=bottom_nodes,
            aggregate_nodes=tuple(aggregate_nodes),
        )

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

    def aggregate(
        self,
        bottom_values: Mapping[str, object],
        *,
        node_labels: Iterable[str] | None = None,
    ) -> dict[str, int | float | None]:
        """Evaluate selected nodes for one bottom-series cross-section.

        Missing or absent members make only their containing nodes undefined.
        Results retain canonical lattice order regardless of selection order.
        Only observations belonging to selected nodes are interpreted.
        Integral sums remain exact Python integers; mixed or floating sums use
        deterministic ``math.fsum``. Storage and batching stay with callers.
        """
        observations = _validate_observation_mapping(
            bottom_values,
            bottom_series=self._bottom_series,
        )
        selected_nodes = _select_nodes(self._nodes, node_labels=node_labels)
        return {node.label: _coherent_sum(observations, node=node) for node in selected_nodes}


def _build_hierarchy_index(
    cls: type[HierarchyIndex],
    *,
    bottom: tuple[str, ...],
    attributes: tuple[str, ...],
    bottom_nodes: tuple[HierarchyNode, ...],
    aggregate_nodes: tuple[HierarchyNode, ...],
) -> HierarchyIndex:
    generated_labels = {node.label for node in aggregate_nodes} | {TOTAL_NODE_LABEL}
    bottom_collisions = sorted(set(bottom) & generated_labels, key=str.encode)
    if bottom_collisions:
        raise HierarchyError(
            f"hierarchy node labels collide with bottom series keys: {bottom_collisions}"
        )
    nodes = (
        *bottom_nodes,
        *aggregate_nodes,
        HierarchyNode(
            label=TOTAL_NODE_LABEL,
            kind=HierarchyNodeKind.TOTAL,
            members=bottom,
        ),
    )
    instance = object.__new__(cls)
    object.__setattr__(instance, "_bottom_series", bottom)
    object.__setattr__(instance, "_attribute_names", attributes)
    object.__setattr__(instance, "_nodes", nodes)
    return instance


def _bottom_nodes(bottom: tuple[str, ...]) -> tuple[HierarchyNode, ...]:
    return tuple(
        HierarchyNode(
            label=series_key,
            kind=HierarchyNodeKind.BOTTOM,
            members=(series_key,),
        )
        for series_key in bottom
    )


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


def _validate_observation_mapping(
    bottom_values: Mapping[str, object],
    *,
    bottom_series: tuple[str, ...],
) -> Mapping[str, object]:
    if not isinstance(bottom_values, Mapping):
        raise HierarchyError("bottom_values must be a mapping of series keys to observations")
    expected = set(bottom_series)
    unknown: list[str] = []
    for label in bottom_values:
        if not isinstance(label, str) or not label:
            raise HierarchyError("bottom_values keys must be non-empty strings")
        _require_utf8(label, name="bottom_values key")
        if label not in expected:
            unknown.append(label)
    if unknown:
        raise HierarchyError(
            f"bottom_values contain unknown bottom series: {sorted(unknown, key=str.encode)}"
        )
    return bottom_values


def _select_nodes(
    nodes: tuple[HierarchyNode, ...],
    *,
    node_labels: Iterable[str] | None,
) -> tuple[HierarchyNode, ...]:
    if node_labels is None:
        return nodes
    if isinstance(node_labels, (str, bytes)):
        raise HierarchyError("node_labels must be an iterable of node labels")
    try:
        labels = tuple(node_labels)
    except TypeError as error:
        raise HierarchyError("node_labels must be an iterable of node labels") from error
    for label in labels:
        if not isinstance(label, str) or not label:
            raise HierarchyError("node_labels must contain non-empty strings")
        _require_utf8(label, name="node label")
    if len(set(labels)) != len(labels):
        raise HierarchyError("node_labels must be unique")
    known = {node.label for node in nodes}
    unknown = sorted(set(labels) - known, key=str.encode)
    if unknown:
        raise HierarchyError(f"unknown hierarchy node labels: {unknown}")
    selected = set(labels)
    return tuple(node for node in nodes if node.label in selected)


def _coherent_sum(
    observations: Mapping[str, object],
    *,
    node: HierarchyNode,
) -> int | float | None:
    operands: list[int | float] = []
    undefined = False
    for member in node.members:
        if member not in observations:
            undefined = True
            continue
        value = _normalize_observation(observations[member], series_key=member)
        if value is None:
            undefined = True
        else:
            operands.append(value)
    if undefined:
        return None
    integer_operands = [value for value in operands if isinstance(value, int)]
    if len(integer_operands) == len(operands):
        return sum(integer_operands)
    try:
        return math.fsum(operands)
    except (OverflowError, ValueError) as error:
        raise HierarchyError(f"cannot aggregate node {node.label!r} as a finite float") from error


def _normalize_observation(value: object, *, series_key: str) -> int | float | None:
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HierarchyError(
            f"observation for bottom series {series_key!r} must be an integer, float, or missing"
        )
    if isinstance(value, Integral):
        return int(value)
    return float(value)


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
