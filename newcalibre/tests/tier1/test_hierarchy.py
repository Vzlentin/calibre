"""Exercise hierarchy compilation and aggregate-history coherence."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from newcalibre.domain import (
    AGGREGATE_NODE_PREFIX,
    AVAILABILITY_BOUND,
    CENSOR_STATUS,
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    TOTAL_NODE_LABEL,
    UNDECLARED_CENSORING,
    Calendar,
    HierarchyError,
    HierarchyIndex,
    HierarchyNodeKind,
    Panel,
    Scope,
)


def _panel(
    rows: Sequence[tuple[str, str, float]],
    *,
    frequency: str = "D",
    extras: bool = False,
) -> Panel:
    frame = pd.DataFrame(rows, columns=[SERIES_KEY, TIMESTAMP, OBSERVED_VALUE])
    frame[SERIES_KEY] = frame[SERIES_KEY].astype("string[pyarrow]")
    frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP]).astype("datetime64[us]")
    frame[OBSERVED_VALUE] = frame[OBSERVED_VALUE].astype("float64")
    if extras:
        frame[CENSOR_STATUS] = pd.Series(["uncensored"] * len(frame), dtype="string[pyarrow]")
        frame[AVAILABILITY_BOUND] = np.arange(len(frame), dtype="int64")
        frame["price"] = np.arange(10, 10 + len(frame), dtype="int32")
    return Panel.from_frame(frame, calendar=Calendar(frequency))


def _facts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: ["sku-c", "sku-a", "sku-b"],
            "location": ["north", "north", "south"],
            "category": ["drink", "food", "food"],
        }
    )


def _index(panel: Panel, facts: pd.DataFrame | None = None) -> HierarchyIndex:
    return HierarchyIndex.from_facts(
        _facts() if facts is None else facts,
        bottom_series=panel.series_keys,
        calendar=panel.calendar,
    )


def test_compiles_the_direct_overlapping_lattice_in_exact_canonical_order() -> None:
    panel = _panel(
        [("sku-a", "2026-01-01", 1), ("sku-b", "2026-01-01", 2), ("sku-c", "2026-01-01", 3)]
    )
    index = _index(panel)

    assert index.bottom_series == ("sku-a", "sku-b", "sku-c")
    assert index.attribute_names == ("category", "location")
    assert index.node_labels == (
        "sku-a",
        "sku-b",
        "sku-c",
        f"{AGGREGATE_NODE_PREFIX}:category:s:drink",
        f"{AGGREGATE_NODE_PREFIX}:category:s:food",
        f"{AGGREGATE_NODE_PREFIX}:location:s:north",
        f"{AGGREGATE_NODE_PREFIX}:location:s:south",
        TOTAL_NODE_LABEL,
    )
    assert [node.kind for node in index.nodes] == [
        HierarchyNodeKind.BOTTOM,
        HierarchyNodeKind.BOTTOM,
        HierarchyNodeKind.BOTTOM,
        HierarchyNodeKind.AGGREGATE,
        HierarchyNodeKind.AGGREGATE,
        HierarchyNodeKind.AGGREGATE,
        HierarchyNodeKind.AGGREGATE,
        HierarchyNodeKind.TOTAL,
    ]
    assert [node.members for node in index.nodes[3:]] == [
        ("sku-c",),
        ("sku-a", "sku-b"),
        ("sku-a", "sku-c"),
        ("sku-b",),
        ("sku-a", "sku-b", "sku-c"),
    ]
    assert [node.expected_member_count for node in index.nodes] == [1, 1, 1, 1, 2, 2, 1, 3]
    assert len(index.nodes) == 3 + 2 + 2 + 1


def test_expand_history_sums_complete_members_and_poison_only_containing_nodes() -> None:
    panel = _panel(
        [
            ("sku-a", "2026-01-01", 1),
            ("sku-b", "2026-01-01", 2),
            ("sku-c", "2026-01-01", 3),
            ("sku-a", "2026-01-02", 0),
            ("sku-c", "2026-01-02", 4),
            ("sku-a", "2026-01-03", 5),
            ("sku-b", "2026-01-03", math.nan),
            ("sku-c", "2026-01-03", 6),
            ("sku-a", "2026-01-04", 0),
            ("sku-b", "2026-01-04", 0),
            ("sku-c", "2026-01-04", 0),
        ],
        extras=True,
    )
    index = _index(panel)
    expanded = index.expand_history(panel)
    frame = expanded.frame.set_index([SERIES_KEY, TIMESTAMP])[OBSERVED_VALUE]

    category_drink = f"{AGGREGATE_NODE_PREFIX}:category:s:drink"
    category_food = f"{AGGREGATE_NODE_PREFIX}:category:s:food"
    location_north = f"{AGGREGATE_NODE_PREFIX}:location:s:north"
    location_south = f"{AGGREGATE_NODE_PREFIX}:location:s:south"
    at = lambda label, day: frame.loc[(label, pd.Timestamp(day))]  # noqa: E731

    assert at(category_drink, "2026-01-01") == 3.0
    assert at(category_food, "2026-01-01") == 3.0
    assert at(location_north, "2026-01-01") == 4.0
    assert at(TOTAL_NODE_LABEL, "2026-01-01") == 6.0
    assert at(category_drink, "2026-01-02") == 4.0
    assert math.isnan(at(category_food, "2026-01-02"))
    assert at(location_north, "2026-01-02") == 4.0
    assert math.isnan(at(location_south, "2026-01-02"))
    assert math.isnan(at(TOTAL_NODE_LABEL, "2026-01-02"))
    assert at(category_drink, "2026-01-03") == 6.0
    assert math.isnan(at(category_food, "2026-01-03"))
    assert at(location_north, "2026-01-03") == 11.0
    assert math.isnan(at(TOTAL_NODE_LABEL, "2026-01-03"))
    assert at(TOTAL_NODE_LABEL, "2026-01-04") == 0.0

    expected_panel_order = tuple(sorted(index.node_labels, key=str.encode))
    assert expanded.series_keys == expected_panel_order
    assert expanded.calendar == panel.calendar
    assert expanded.frame[SERIES_KEY].drop_duplicates().tolist() == list(expected_panel_order)
    aggregate_rows = expanded.frame[~expanded.frame[SERIES_KEY].isin(index.bottom_series)]
    assert set(aggregate_rows[CENSOR_STATUS]) == {UNDECLARED_CENSORING}
    assert aggregate_rows[[AVAILABILITY_BOUND, "price"]].isna().all(axis=None)


def test_expanded_history_preserves_the_panel_contract_through_task_construction() -> None:
    panel = _panel(
        [("sku-a", "2026-01-01", 1), ("sku-b", "2026-01-01", 2), ("sku-c", "2026-01-01", 3)]
    )
    expanded = _index(panel).expand_history(panel)

    reingested = Panel.from_frame(expanded.frame, calendar=expanded.calendar)
    (task,) = expanded.forecast_tasks(
        origin=pd.Timestamp("2026-01-02"),
        horizon=1,
        scope=Scope.GLOBAL,
        model_config={"backend": "test"},
    )

    pd.testing.assert_frame_equal(expanded.frame, reingested.frame)
    assert task.series_keys == expanded.series_keys
    assert set(task.history[SERIES_KEY]) == set(expanded.series_keys)


def test_compilation_and_expansion_do_not_retain_or_expose_mutable_callers() -> None:
    source = pd.DataFrame(
        {
            SERIES_KEY: ["a", "b"],
            "group": ["one", "one"],
        }
    )
    panel = _panel([("a", "2026-01-01", 1), ("b", "2026-01-01", 2)])
    index = _index(panel, source)
    source.loc[:, "group"] = "mutated"
    exposed = panel.frame
    exposed.loc[:, OBSERVED_VALUE] = 999

    expanded = index.expand_history(panel)
    returned = expanded.frame
    returned.loc[:, OBSERVED_VALUE] = -1

    assert index.nodes[2].members == ("a", "b")
    assert (
        expanded.frame.loc[expanded.frame[SERIES_KEY] == TOTAL_NODE_LABEL, OBSERVED_VALUE].item()
        == 3.0
    )


@pytest.mark.parametrize(
    ("facts", "pattern"),
    [
        (pd.DataFrame({"group": ["x"]}), "missing required"),
        (pd.DataFrame({SERIES_KEY: ["a"]}), "attribute"),
        (pd.DataFrame({SERIES_KEY: ["a", "a"], "group": ["x", "y"]}), "collide"),
        (pd.DataFrame({SERIES_KEY: [1, "1"], "group": ["x", "y"]}), "non-empty strings"),
        (pd.DataFrame({SERIES_KEY: ["a"], "group": [None]}), "cannot be missing"),
        (pd.DataFrame({SERIES_KEY: ["a"], "group": [["x"]]}), "must use a string"),
    ],
)
def test_rejects_invalid_fact_shapes_and_values(facts: pd.DataFrame, pattern: str) -> None:
    panel = _panel([("a", "2026-01-01", 1)])
    with pytest.raises(HierarchyError, match=pattern):
        _index(panel, facts)


def test_rejects_missing_extra_facts_and_reserved_or_generated_label_collisions() -> None:
    panel = _panel([("a", "2026-01-01", 1), ("b", "2026-01-01", 2)])
    with pytest.raises(HierarchyError, match="cover bottom series exactly"):
        _index(panel, pd.DataFrame({SERIES_KEY: ["a"], "group": ["x"]}))
    with pytest.raises(HierarchyError, match="cover bottom series exactly"):
        _index(
            panel,
            pd.DataFrame({SERIES_KEY: ["a", "b", "c"], "group": ["x", "x", "x"]}),
        )

    numeric_fact_panel = _panel([("1", "2026-01-01", 1)])
    with pytest.raises(HierarchyError, match="non-empty strings"):
        _index(
            numeric_fact_panel,
            pd.DataFrame({SERIES_KEY: [1], "group": ["x"]}),
        )

    total_panel = _panel([(TOTAL_NODE_LABEL, "2026-01-01", 1)])
    with pytest.raises(HierarchyError, match="collide with bottom"):
        _index(
            total_panel,
            pd.DataFrame({SERIES_KEY: [TOTAL_NODE_LABEL], "group": ["x"]}),
        )
    generated = f"{AGGREGATE_NODE_PREFIX}:group:s:x"
    generated_panel = _panel([(generated, "2026-01-01", 1)])
    with pytest.raises(HierarchyError, match="collide with bottom"):
        _index(
            generated_panel,
            pd.DataFrame({SERIES_KEY: [generated], "group": ["x"]}),
        )


def test_typed_escaped_labels_and_permuted_facts_are_deterministic() -> None:
    panel = _panel([("a", "2026-01-01", 1), ("b", "2026-01-01", 2), ("c", "2026-01-01", 3)])
    facts = pd.DataFrame({SERIES_KEY: ["a", "b", "c"], "a:b": [1, "1", "x/y% z"]})
    first = _index(panel, facts)
    second = _index(panel, facts.iloc[::-1][["a:b", SERIES_KEY]])

    assert first == second
    assert first.node_labels[3:-1] == (
        f"{AGGREGATE_NODE_PREFIX}:a%3Ab:i:1",
        f"{AGGREGATE_NODE_PREFIX}:a%3Ab:s:1",
        f"{AGGREGATE_NODE_PREFIX}:a%3Ab:s:x%2Fy%25%20z",
    )


def test_expansion_rejects_wrong_calendar_unknown_nodes_and_already_expanded_panels() -> None:
    daily = _panel([("a", "2026-01-05", 1), ("b", "2026-01-05", 2)])
    facts = pd.DataFrame({SERIES_KEY: ["a", "b"], "group": ["x", "x"]})
    index = _index(daily, facts)
    weekly = _panel([("a", "2026-01-05", 1), ("b", "2026-01-05", 2)], frequency="W-MON")
    with pytest.raises(HierarchyError, match="calendar is incompatible"):
        index.expand_history(weekly)

    unknown = _panel([("a", "2026-01-05", 1), ("c", "2026-01-05", 2)])
    with pytest.raises(HierarchyError, match="missing=.*b.*unexpected=.*c"):
        index.expand_history(unknown)
    expanded = index.expand_history(daily)
    with pytest.raises(HierarchyError, match="exactly the hierarchy bottom"):
        index.expand_history(expanded)


@given(
    n_bottom=st.integers(min_value=1, max_value=12),
    n_attributes=st.integers(min_value=1, max_value=5),
)
def test_generated_lattice_obeys_the_node_count_identity(n_bottom: int, n_attributes: int) -> None:
    keys = [f"s{index:02d}" for index in range(n_bottom)]
    panel = _panel([(key, "2026-01-01", float(index)) for index, key in enumerate(keys)])
    data: dict[str, list[object]] = {SERIES_KEY: keys}
    distinct_counts: list[int] = []
    for attribute_index in range(n_attributes):
        cardinality = min(n_bottom, attribute_index + 1)
        data[f"attribute-{attribute_index}"] = [index % cardinality for index in range(n_bottom)]
        distinct_counts.append(cardinality)

    index = _index(panel, pd.DataFrame(data))

    assert len(index.nodes) == n_bottom + sum(distinct_counts) + 1
