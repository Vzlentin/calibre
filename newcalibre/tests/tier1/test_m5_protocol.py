"""Prove strict M5 loading and compilation into canonical domain inputs."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import yaml

from newcalibre.domain import (
    AGGREGATE_NODE_PREFIX,
    CENSOR_STATUS,
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    TOTAL_NODE_LABEL,
    HierarchyError,
    HierarchyIndex,
)
from newcalibre.protocols.m5 import load_m5_config
from newcalibre.protocols.m5.compiler import (
    _compile_hierarchy,
    _level_from_node_label,
    compile_m5_protocol,
)
from newcalibre.protocols.m5.loader import M5DataError, load_m5_dataset

_PROJECT_ROOT = Path(__file__).parents[2]
_GATE_C = _PROJECT_ROOT / "benchmarks" / "m5" / "gate-c.yaml"
_DAY_COUNT = 1941
_SOURCE_FACTS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")


def _row(index: int) -> dict[str, object]:
    item = f"ITEM_{index}"
    store = f"STORE_{index % 3}"
    return {
        "id": f"{item}_{store}_evaluation",
        "item_id": item,
        "dept_id": f"DEPT_{index % 2}",
        "cat_id": f"CAT_{index % 2}",
        "store_id": store,
        "state_id": f"STATE_{index % 2}",
    }


def _sales_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    values: dict[str, object] = {
        "id": [row["id"] for row in rows],
        **{name: [row[name] for row in rows] for name in _SOURCE_FACTS},
    }
    for day in range(1, _DAY_COUNT + 1):
        values[f"d_{day}"] = [(index + day) % 11 for index in range(len(rows))]
    return pd.DataFrame(values)


def _calendar_frame(*, start: str = "2011-01-29", extra_days: int = 0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=_DAY_COUNT + extra_days, freq="D")
    return pd.DataFrame(
        {
            "d": [f"d_{index}" for index in range(1, len(dates) + 1)],
            "date": dates.strftime("%Y-%m-%d"),
        }
    )


def _write_release(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    sales_mutation: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    calendar_mutation: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    start: str = "2011-01-29",
    extra_calendar_days: int = 0,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "data"
    target.mkdir()
    sales = _sales_frame(rows or [_row(0), _row(1), _row(2)])
    calendar = _calendar_frame(start=start, extra_days=extra_calendar_days)
    if sales_mutation is not None:
        sales = sales_mutation(sales)
    if calendar_mutation is not None:
        calendar = calendar_mutation(calendar)
    sales_path = target / "sales_train_evaluation.csv"
    calendar_path = target / "calendar.csv"
    sales.to_csv(sales_path, index=False)
    calendar.to_csv(calendar_path, index=False)
    inventory = {
        "schema": 1,
        "dataset": "m5",
        "files": [
            _inventory_entry(calendar_path),
            _inventory_entry(sales_path),
        ],
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    return target


def _inventory_entry(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "name": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _config(tmp_path: Path, *, population: dict[str, object] | None = None):
    payload = yaml.safe_load(_GATE_C.read_text(encoding="utf-8"))
    payload["dataset"]["data_dir"] = "data"
    payload["dataset"]["inventory"] = "inventory.json"
    if population is not None:
        payload["protocol"]["population"] = population
    path = tmp_path / "gate-c.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_m5_config(path)


def test_loader_binds_both_input_paths_to_configured_project_root(tmp_path: Path) -> None:
    target = _write_release(tmp_path)
    alternate_target = _write_release(tmp_path / "alternate")
    config = _config(tmp_path)

    assert tuple(inspect.signature(load_m5_dataset).parameters) == ("project_root", "config")
    assert load_m5_dataset(tmp_path, config).bottom_series
    with pytest.raises(TypeError):
        load_m5_dataset(target, alternate_target.parent / "inventory.json", config)  # type: ignore[call-arg]


def test_loader_verifies_before_any_csv_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _write_release(tmp_path)
    (target / "calendar.csv").write_bytes(b"invalid after inventory")
    calls = 0

    def forbidden_parser(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        raise AssertionError("parser must not run")

    monkeypatch.setattr(pd, "read_csv", forbidden_parser)
    with pytest.raises(M5DataError, match="size|sha256"):
        load_m5_dataset(tmp_path, _config(tmp_path))
    assert calls == 0


def test_loader_rehashes_each_selected_input_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _write_release(tmp_path)
    original = pd.read_csv
    calls = 0

    def mutate_after_first_parse(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        frame = original(*args, **kwargs)
        if calls == 1:
            payload = (target / "sales_train_evaluation.csv").read_bytes()
            (target / "sales_train_evaluation.csv").write_bytes(payload[:-1] + b"X")
        return frame

    monkeypatch.setattr(pd, "read_csv", mutate_after_first_parse)
    with pytest.raises(M5DataError, match="sha256"):
        load_m5_dataset(tmp_path, _config(tmp_path))
    assert calls == 1


@pytest.mark.parametrize(
    ("sales_mutation", "match"),
    [
        (lambda frame: frame[[*frame.columns[:6], "d_2", "d_1", *frame.columns[8:]]], "day"),
        (lambda frame: frame.rename(columns={"d_2": "d_2000"}), "day"),
        (lambda frame: frame.drop(columns="d_1941"), "day"),
        (lambda frame: frame.assign(dept_id=None), "hierarchy"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate"),
        (lambda frame: frame.assign(d_10=-1), "non-negative"),
        (lambda frame: frame.assign(d_10=1.5), "integral"),
        (lambda frame: frame.assign(d_10=None), "integral"),
    ],
)
def test_loader_rejects_invalid_evaluation_sales(
    tmp_path: Path,
    sales_mutation: Callable[[pd.DataFrame], pd.DataFrame],
    match: str,
) -> None:
    _write_release(tmp_path, sales_mutation=sales_mutation)
    with pytest.raises(M5DataError, match=match):
        load_m5_dataset(tmp_path, _config(tmp_path))


def test_loader_preserves_integer_above_float_exact_precision(tmp_path: Path) -> None:
    exact_count = 2**53 + 1
    _write_release(tmp_path, sales_mutation=lambda frame: frame.assign(d_10=exact_count))

    dataset = load_m5_dataset(tmp_path, _config(tmp_path))

    assert set(dataset.sales["d_10"]) == {exact_count}


def test_loader_rejects_bottom_label_collision(tmp_path: Path) -> None:
    rows = [_row(0), _row(1)]
    rows[0].update(item_id="A_B", store_id="C")
    rows[1].update(item_id="A", store_id="B_C")
    _write_release(tmp_path, rows=rows)
    with pytest.raises(M5DataError, match="bottom label collision"):
        load_m5_dataset(tmp_path, _config(tmp_path))


@pytest.mark.parametrize(
    ("calendar_mutation", "match"),
    [
        (
            lambda frame: frame.assign(d=lambda value: value["d"].mask(value.index == 1, "d_1")),
            "unique",
        ),
        (
            lambda frame: frame.assign(
                date=lambda value: value["date"].mask(value.index == 1, value.loc[0, "date"])
            ),
            "unique",
        ),
        (lambda frame: frame.drop(index=100).reset_index(drop=True), "calendar"),
        (
            lambda frame: frame.drop(columns="d").drop(index=100).reset_index(drop=True),
            "contiguous",
        ),
    ],
)
def test_loader_rejects_invalid_calendar_mapping(
    tmp_path: Path,
    calendar_mutation: Callable[[pd.DataFrame], pd.DataFrame],
    match: str,
) -> None:
    _write_release(tmp_path, calendar_mutation=calendar_mutation)
    with pytest.raises(M5DataError, match=match):
        load_m5_dataset(tmp_path, _config(tmp_path))


def test_label_less_calendar_is_sorted_then_derived(tmp_path: Path) -> None:
    def label_less_reversed(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(columns="d").iloc[::-1].reset_index(drop=True)

    _write_release(tmp_path, calendar_mutation=label_less_reversed)
    dataset = load_m5_dataset(tmp_path, _config(tmp_path))
    assert dataset.history_end == pd.Timestamp("2016-05-22")
    assert dataset.dates[0] == pd.Timestamp("2011-01-29")


def test_history_end_uses_final_consumed_day_not_final_calendar_row(tmp_path: Path) -> None:
    _write_release(tmp_path, extra_calendar_days=28)
    dataset = load_m5_dataset(tmp_path, _config(tmp_path))
    assert dataset.history_end == pd.Timestamp("2016-05-22")


def test_digest_rank_is_repeatable_row_order_independent_nested_and_salt_sensitive(
    tmp_path: Path,
) -> None:
    rows = [_row(index) for index in range(8)]
    first_root = _write_release(tmp_path / "first", rows=rows).parent
    reverse_root = _write_release(tmp_path / "reverse", rows=rows[::-1]).parent

    def selected(project_root: Path, count: int, salt: str) -> tuple[str, ...]:
        config = _config(
            project_root,
            population={"kind": "digest_rank", "bottom_count": count, "salt": salt},
        )
        return load_m5_dataset(project_root, config).bottom_series

    first_three = selected(first_root, 3, "salt-a")
    assert first_three == selected(first_root, 3, "salt-a")
    assert first_three == selected(reverse_root, 3, "salt-a")
    assert set(first_three) < set(selected(first_root, 5, "salt-a"))
    assert first_three != selected(first_root, 3, "salt-b")


def test_full_population_retains_every_validated_bottom_identity(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(6)]
    _write_release(tmp_path, rows=rows[::-1])
    dataset = load_m5_dataset(tmp_path, _config(tmp_path))
    expected = tuple(
        sorted((f"{row['item_id']}_{row['store_id']}" for row in rows), key=str.encode)
    )
    assert dataset.bottom_series == expected


def test_population_selection_precedes_hierarchy_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row(index) for index in range(6)]
    _write_release(tmp_path, rows=rows)
    config = _config(
        tmp_path,
        population={"kind": "digest_rank", "bottom_count": 2, "salt": "selection"},
    )
    dataset = load_m5_dataset(tmp_path, config)
    original = HierarchyIndex.from_facts.__func__
    observed_rows: list[int] = []

    def wrapped(cls: type[HierarchyIndex], facts: pd.DataFrame, *, bottom_series: object):
        observed_rows.append(len(facts))
        return original(cls, facts, bottom_series=bottom_series)

    monkeypatch.setattr(HierarchyIndex, "from_facts", classmethod(wrapped))
    compiled = compile_m5_protocol(dataset, config)
    assert observed_rows == [2]
    assert len(compiled.hierarchy.bottom_series) == 2


def test_compiler_builds_canonical_panel_hierarchy_origins_and_intent(tmp_path: Path) -> None:
    _write_release(tmp_path, start="2020-01-01")
    config = _config(tmp_path)
    dataset = load_m5_dataset(tmp_path, config)
    compiled = compile_m5_protocol(dataset, config)

    frame = compiled.panel.frame
    assert tuple(frame.columns) == (SERIES_KEY, TIMESTAMP, OBSERVED_VALUE)
    assert CENSOR_STATUS not in frame
    assert not compiled.panel.has_censoring_facts
    assert tuple(frame[SERIES_KEY].unique()) == compiled.panel.series_keys
    assert frame.equals(frame.sort_values([SERIES_KEY, TIMESTAMP]).reset_index(drop=True))
    assert len(compiled.origins) == 64
    assert compiled.origins[-1] == dataset.history_end
    assert compiled.origins[0] == dataset.history_end - pd.Timedelta(days=63)
    assert compiled.config is config
    assert compiled.model_config == config.model_config
    assert compiled.reconciliation_strategy == "wls_struct"
    assert compiled.conformal_partition == "series-horizon"
    assert compiled.execution == config.execution
    assert compiled.output_dir == Path("results/m5/gate-c")


def test_every_hierarchy_node_label_recovers_one_of_seven_level_classes(tmp_path: Path) -> None:
    _write_release(tmp_path)
    config = _config(tmp_path)
    compiled = compile_m5_protocol(load_m5_dataset(tmp_path, config), config)
    levels = {_level_from_node_label(label) for label in compiled.hierarchy.node_labels}
    assert levels == {"bottom", "item", "department", "category", "store", "state", "total"}


def test_canonical_shape_hierarchy_has_exact_node_and_attribute_counts() -> None:
    facts_rows: list[dict[str, str]] = []
    for item_index in range(3049):
        item = f"ITEM_{item_index:04d}"
        for store_index in range(10):
            store = f"STORE_{store_index}"
            facts_rows.append(
                {
                    SERIES_KEY: f"{item}_{store}",
                    "item": item,
                    "department": f"DEPT_{item_index % 7}",
                    "category": f"CAT_{item_index % 3}",
                    "store": store,
                    "state": f"STATE_{store_index % 3}",
                }
            )
    facts = pd.DataFrame(facts_rows)
    hierarchy = _compile_hierarchy(facts, tuple(facts[SERIES_KEY]))
    levels = [_level_from_node_label(label) for label in hierarchy.node_labels]

    assert len(hierarchy.bottom_series) == 30490
    assert len(hierarchy.nodes) == 33563
    assert {level: levels.count(level) for level in set(levels)} == {
        "bottom": 30490,
        "item": 3049,
        "department": 7,
        "category": 3,
        "store": 10,
        "state": 3,
        "total": 1,
    }


def test_aggregate_cross_sections_are_exact_and_missing_members_stay_undefined(
    tmp_path: Path,
) -> None:
    _write_release(tmp_path)
    config = _config(tmp_path)
    compiled = compile_m5_protocol(load_m5_dataset(tmp_path, config), config)
    values = {key: index + 1 for index, key in enumerate(compiled.hierarchy.bottom_series)}
    actual = compiled.hierarchy.aggregate(values)
    for node in compiled.hierarchy.nodes:
        assert actual[node.label] == sum(values[member] for member in node.members)
    missing_key = compiled.hierarchy.bottom_series[0]
    partial = {key: value for key, value in values.items() if key != missing_key}
    undefined = compiled.hierarchy.aggregate(partial)
    for node in compiled.hierarchy.nodes:
        if missing_key in node.members:
            assert undefined[node.label] is None


def test_hierarchy_refuses_missing_extra_or_generated_label_collisions() -> None:
    facts = pd.DataFrame(
        [
            {
                SERIES_KEY: "A",
                "item": "ITEM_A",
                "department": "D",
                "category": "C",
                "store": "S",
                "state": "ST",
            }
        ]
    )
    with pytest.raises(HierarchyError, match="exactly"):
        _compile_hierarchy(facts, ("A", "B"))
    with pytest.raises(HierarchyError, match="exactly"):
        _compile_hierarchy(facts, ("B",))

    collision = f"{AGGREGATE_NODE_PREFIX}:item:s:ITEM_A"
    collision_facts = facts.assign(**{SERIES_KEY: collision})
    with pytest.raises(HierarchyError, match="prefix|collide"):
        _compile_hierarchy(collision_facts, (collision,))
    total_facts = facts.assign(**{SERIES_KEY: TOTAL_NODE_LABEL})
    with pytest.raises(HierarchyError, match="total|collide"):
        _compile_hierarchy(total_facts, (TOTAL_NODE_LABEL,))


def test_m5_package_exposes_only_loading_execution_verification_and_scoring() -> None:
    import newcalibre.protocols.m5 as m5

    assert m5.__all__ == [
        "M5Diagnostics",
        "M5RunResult",
        "load_m5_config",
        "run_m5",
        "score_m5",
        "verify_m5_inputs",
    ]
    forbidden = {
        "M5Lattice",
        "emit_m5_results",
        "promote_m5_results",
        "M5EngineAdapter",
    }
    assert forbidden.isdisjoint(vars(m5))
