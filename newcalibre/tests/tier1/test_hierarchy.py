"""Exercise hierarchy compilation and bounded coherent aggregation."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from newcalibre.domain import (
    AGGREGATE_NODE_PREFIX,
    SERIES_KEY,
    TOTAL_NODE_LABEL,
    HierarchyError,
    HierarchyIndex,
    HierarchyNode,
    HierarchyNodeKind,
)


def _facts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: ["sku-c", "sku-a", "sku-b"],
            "location": ["north", "north", "south"],
            "category": ["drink", "food", "food"],
        }
    )


def _index(
    facts: pd.DataFrame | None = None,
    *,
    bottom_series: tuple[str, ...] = ("sku-a", "sku-b", "sku-c"),
) -> HierarchyIndex:
    return HierarchyIndex.from_facts(
        _facts() if facts is None else facts,
        bottom_series=bottom_series,
    )


def test_compiles_the_direct_overlapping_lattice_in_exact_canonical_order() -> None:
    index = _index()

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


def test_aggregates_exact_values_in_canonical_lattice_order() -> None:
    index = _index()

    result = index.aggregate({"sku-a": 1, "sku-b": 2, "sku-c": 3})

    assert tuple(result) == index.node_labels
    assert tuple(result.values()) == (1, 2, 3, 3, 3, 4, 2, 6)

    cancellation = index.aggregate(
        {"sku-a": 10**16, "sku-b": 1.0, "sku-c": -(10**16)},
        node_labels=[TOTAL_NODE_LABEL],
    )
    assert cancellation == {TOTAL_NODE_LABEL: 1.0}


def test_missing_members_poison_only_the_nodes_that_contain_them() -> None:
    index = _index()
    category_drink = f"{AGGREGATE_NODE_PREFIX}:category:s:drink"
    category_food = f"{AGGREGATE_NODE_PREFIX}:category:s:food"
    location_north = f"{AGGREGATE_NODE_PREFIX}:location:s:north"
    location_south = f"{AGGREGATE_NODE_PREFIX}:location:s:south"

    absent = index.aggregate({"sku-a": 1, "sku-c": 3})
    assert absent == {
        "sku-a": 1,
        "sku-b": None,
        "sku-c": 3,
        category_drink: 3,
        category_food: None,
        location_north: 4,
        location_south: None,
        TOTAL_NODE_LABEL: None,
    }

    explicit = index.aggregate({"sku-a": 1, "sku-b": 2, "sku-c": pd.NA})
    assert explicit[category_drink] is None
    assert explicit[category_food] == 3
    assert explicit[location_north] is None
    assert explicit[location_south] == 2
    assert explicit[TOTAL_NODE_LABEL] is None


def test_selected_nodes_are_evaluated_only_over_their_members() -> None:
    index = _index()
    location_north = f"{AGGREGATE_NODE_PREFIX}:location:s:north"
    observations = {"sku-a": 1, "sku-b": object(), "sku-c": 3}
    selection = [location_north, "sku-a"]

    result = index.aggregate(observations, node_labels=selection)

    assert result == {"sku-a": 1, location_north: 4}
    assert selection == [location_north, "sku-a"]
    assert observations["sku-a"] == 1


def test_integral_aggregation_remains_exact_above_float_and_uint64_boundaries() -> None:
    index = _index()
    category_food = f"{AGGREGATE_NODE_PREFIX}:category:s:food"
    high = np.uint64(2**64 - 1)

    result = index.aggregate(
        {"sku-a": high, "sku-b": np.uint64(7), "sku-c": np.uint64(2)},
        node_labels=[category_food, TOTAL_NODE_LABEL],
    )

    assert result == {
        category_food: 2**64 + 6,
        TOTAL_NODE_LABEL: 2**64 + 8,
    }
    assert all(isinstance(value, int) for value in result.values())


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"label": "", "kind": HierarchyNodeKind.BOTTOM, "members": ("a",)}, "label"),
        (
            {"label": "\ud800", "kind": HierarchyNodeKind.BOTTOM, "members": ("\ud800",)},
            "UTF-8",
        ),
        ({"label": "a", "kind": "bottom", "members": ("a",)}, "kind"),
        ({"label": "a", "kind": HierarchyNodeKind.BOTTOM, "members": ["a"]}, "tuple"),
        ({"label": "a", "kind": HierarchyNodeKind.BOTTOM, "members": ()}, "tuple"),
        ({"label": "a", "kind": HierarchyNodeKind.BOTTOM, "members": (1,)}, "strings"),
        ({"label": "a", "kind": HierarchyNodeKind.BOTTOM, "members": ("",)}, "strings"),
        ({"label": "a", "kind": HierarchyNodeKind.BOTTOM, "members": ("a", "a")}, "unique"),
        ({"label": "a", "kind": HierarchyNodeKind.BOTTOM, "members": ("b",)}, "sole member"),
        (
            {"label": "group:x", "kind": HierarchyNodeKind.AGGREGATE, "members": ("a",)},
            "aggregate prefix",
        ),
        (
            {
                "label": f"{AGGREGATE_NODE_PREFIX}:group:s:x",
                "kind": HierarchyNodeKind.BOTTOM,
                "members": (f"{AGGREGATE_NODE_PREFIX}:group:s:x",),
            },
            "only aggregate",
        ),
        ({"label": "a", "kind": HierarchyNodeKind.TOTAL, "members": ("a",)}, "total"),
        (
            {
                "label": TOTAL_NODE_LABEL,
                "kind": HierarchyNodeKind.BOTTOM,
                "members": (TOTAL_NODE_LABEL,),
            },
            "total",
        ),
        (
            {
                "label": TOTAL_NODE_LABEL,
                "kind": HierarchyNodeKind.TOTAL,
                "members": (TOTAL_NODE_LABEL,),
            },
            "cannot also be its member",
        ),
        (
            {
                "label": f"{AGGREGATE_NODE_PREFIX}:group:s:x",
                "kind": HierarchyNodeKind.AGGREGATE,
                "members": (TOTAL_NODE_LABEL,),
            },
            "bottom series labels",
        ),
        (
            {
                "label": TOTAL_NODE_LABEL,
                "kind": HierarchyNodeKind.TOTAL,
                "members": (f"{AGGREGATE_NODE_PREFIX}:group:s:x",),
            },
            "bottom series labels",
        ),
    ],
)
def test_public_node_constructor_rejects_invalid_states(
    kwargs: dict[str, Any], pattern: str
) -> None:
    with pytest.raises(HierarchyError, match=pattern):
        HierarchyNode(**kwargs)


def test_public_node_derives_its_member_count() -> None:
    node = HierarchyNode(
        label=f"{AGGREGATE_NODE_PREFIX}:group:s:x",
        kind=HierarchyNodeKind.AGGREGATE,
        members=("a", "b"),
    )

    assert node.expected_member_count == 2
    constructor = cast(Any, HierarchyNode)
    with pytest.raises(TypeError, match="expected_member_count"):
        constructor(
            label=f"{AGGREGATE_NODE_PREFIX}:group:s:x",
            kind=HierarchyNodeKind.AGGREGATE,
            members=("a", "b"),
            expected_member_count=99,
        )


@pytest.mark.parametrize(
    ("facts", "pattern"),
    [
        (pd.DataFrame({"group": ["x"]}), "missing required"),
        (pd.DataFrame({SERIES_KEY: ["a"]}), "attribute"),
        (pd.DataFrame({SERIES_KEY: ["a", "a"], "group": ["x", "y"]}), "collide"),
        (pd.DataFrame({SERIES_KEY: [1], "group": ["x"]}), "non-empty strings"),
        (pd.DataFrame({SERIES_KEY: ["a"], "group": [None]}), "cannot be missing"),
        (pd.DataFrame({SERIES_KEY: ["a"], "group": [["x"]]}), "must use a string"),
    ],
)
def test_rejects_invalid_fact_shapes_and_values(facts: pd.DataFrame, pattern: str) -> None:
    with pytest.raises(HierarchyError, match=pattern):
        _index(facts, bottom_series=("a",))


def test_rejects_missing_extra_facts_and_reserved_or_generated_label_collisions() -> None:
    with pytest.raises(HierarchyError, match="cover bottom series exactly"):
        _index(
            pd.DataFrame({SERIES_KEY: ["a"], "group": ["x"]}),
            bottom_series=("a", "b"),
        )
    with pytest.raises(HierarchyError, match="cover bottom series exactly"):
        _index(
            pd.DataFrame({SERIES_KEY: ["a", "b", "c"], "group": ["x", "x", "x"]}),
            bottom_series=("a", "b"),
        )
    with pytest.raises(HierarchyError, match="total label"):
        _index(
            pd.DataFrame({SERIES_KEY: [TOTAL_NODE_LABEL], "group": ["x"]}),
            bottom_series=(TOTAL_NODE_LABEL,),
        )
    generated = f"{AGGREGATE_NODE_PREFIX}:group:s:x"
    with pytest.raises(HierarchyError, match="only aggregate"):
        _index(
            pd.DataFrame({SERIES_KEY: [generated], "group": ["x"]}),
            bottom_series=(generated,),
        )


def test_typed_escaped_labels_and_permuted_facts_are_deterministic() -> None:
    keys = ("a", "b", "c", "d", "e")
    facts = pd.DataFrame(
        {
            SERIES_KEY: keys,
            "a:b": pd.Series([True, 1, 1.5, "1", "x/y% z"], dtype="object"),
        }
    )
    first = _index(facts, bottom_series=keys)
    second = _index(facts.iloc[::-1][["a:b", SERIES_KEY]], bottom_series=tuple(reversed(keys)))

    assert first == second
    assert first.node_labels[5:-1] == (
        f"{AGGREGATE_NODE_PREFIX}:a%3Ab:b:true",
        f"{AGGREGATE_NODE_PREFIX}:a%3Ab:i:1",
        f"{AGGREGATE_NODE_PREFIX}:a%3Ab:f:0x1.8000000000000p%2B0",
        f"{AGGREGATE_NODE_PREFIX}:a%3Ab:s:1",
        f"{AGGREGATE_NODE_PREFIX}:a%3Ab:s:x%2Fy%25%20z",
    )


def test_rejects_unknown_labels_and_malformed_observation_inputs() -> None:
    facts = pd.DataFrame({SERIES_KEY: ["a"], "group": ["x"]})
    index = _index(facts, bottom_series=("a",))

    with pytest.raises(HierarchyError, match="must be a mapping"):
        index.aggregate(cast(Any, [("a", 1)]))
    with pytest.raises(HierarchyError, match="unknown bottom"):
        index.aggregate({"unknown": 1})
    with pytest.raises(HierarchyError, match="keys must be non-empty strings"):
        index.aggregate(cast(Any, {1: 1}))
    with pytest.raises(HierarchyError, match="iterable"):
        index.aggregate({"a": 1}, node_labels="a")
    with pytest.raises(HierarchyError, match="unknown hierarchy node"):
        index.aggregate({"a": 1}, node_labels=["unknown"])
    with pytest.raises(HierarchyError, match="unique"):
        index.aggregate({"a": 1}, node_labels=["a", "a"])

    for invalid in (True, "1", object()):
        with pytest.raises(HierarchyError, match="observation"):
            index.aggregate({"a": invalid}, node_labels=["a"])
    assert index.aggregate({"a": np.nan}, node_labels=["a"]) == {"a": None}
    assert index.aggregate({"a": math.inf}, node_labels=["a"]) == {"a": math.inf}


def test_compilation_and_aggregation_do_not_mutate_or_retain_callers() -> None:
    facts = pd.DataFrame({SERIES_KEY: ["b", "a"], "group": ["one", "one"]})
    bottom = ["b", "a"]
    index = HierarchyIndex.from_facts(facts, bottom_series=bottom)
    facts.loc[:, "group"] = "mutated"
    bottom.append("c")
    observations: dict[str, object] = {"a": 1, "b": 2}

    result = index.aggregate(observations)
    result[TOTAL_NODE_LABEL] = -1

    assert index.nodes[2].members == ("a", "b")
    assert observations == {"a": 1, "b": 2}
    assert index.aggregate(observations)[TOTAL_NODE_LABEL] == 3


@given(
    n_bottom=st.integers(min_value=1, max_value=12),
    n_attributes=st.integers(min_value=1, max_value=5),
)
def test_generated_lattice_obeys_the_node_count_identity(n_bottom: int, n_attributes: int) -> None:
    keys = tuple(f"s{index:02d}" for index in range(n_bottom))
    data: dict[str, list[object]] = {SERIES_KEY: list(keys)}
    distinct_counts: list[int] = []
    for attribute_index in range(n_attributes):
        cardinality = min(n_bottom, attribute_index + 1)
        data[f"attribute-{attribute_index}"] = [index % cardinality for index in range(n_bottom)]
        distinct_counts.append(cardinality)

    index = _index(pd.DataFrame(data), bottom_series=keys)

    assert len(index.nodes) == n_bottom + sum(distinct_counts) + 1
