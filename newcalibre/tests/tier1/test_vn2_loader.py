"""Exercise VN2 wide-table loading and progressive reveal hygiene.

Schema, calendar, refusal, and missing-as-zero assertions are exact
tolerance-class-1 facts. Reveal perturbation is a non-vacuity class-4 lock.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.vn2_fixtures import (
    BASE_WEEKS,
    INITIAL_STATE_COLUMNS,
    MASTER_ATTRIBUTES,
    REVEAL_WEEKS,
    SALES_FILES,
    STATIC_FILES,
    refresh_inventory,
    synthetic_config_payload,
    write_config,
    write_dataset,
)

from newcalibre.protocols.vn2 import (
    VN2DataError,
    VN2Dataset,
    load_vn2_config,
    load_vn2_dataset,
)

pytestmark = pytest.mark.tier1


def test_dataset_cannot_bypass_loader_validation() -> None:
    with pytest.raises(TypeError, match="load_vn2_dataset"):
        VN2Dataset()


def _load(root: Path):
    data, inventory, config_path = write_dataset(root)
    config = load_vn2_config(config_path)
    return load_vn2_dataset(data, inventory, config), data, inventory, config


def _rewrite(data: Path, inventory: Path, filename: str, frame: pd.DataFrame) -> None:
    frame.to_csv(data / filename, index=False, lineterminator="\n")
    refresh_inventory(data, inventory)


def test_loader_returns_wide_progressive_round_views_without_future_facts(
    tmp_path: Path,
) -> None:
    dataset, _data, _inventory, _config = _load(tmp_path)

    round_one = dataset.round_input(1)
    round_six = dataset.round_input(6)

    assert round_one.reveal_number == 0
    assert round_one.origin == pd.Timestamp("2024-04-15")
    assert tuple(round_one.sales.columns) == ("Store", "Product", *BASE_WEEKS)
    assert round_one.sales[BASE_WEEKS[1]].tolist() == [2.0, 0.0]
    assert all(round_one.sales[column].dtype == np.dtype("float64") for column in BASE_WEEKS)
    assert tuple(round_one.master.columns) == ("Store", "Product", *MASTER_ATTRIBUTES)
    assert tuple(round_one.in_stock.columns) == ("Store", "Product", *BASE_WEEKS)
    assert REVEAL_WEEKS[0] not in round_one.in_stock
    assert tuple(round_one.initial_state.columns) == INITIAL_STATE_COLUMNS
    assert round_one.initial_state["End Inventory"].tolist() == [3.0, 4.0]
    assert round_one.initial_state["In Transit W+1"].tolist() == [5.0, 6.0]
    assert round_one.initial_state["In Transit W+2"].tolist() == [7.0, 8.0]

    assert round_six.reveal_number == 5
    assert round_six.origin == pd.Timestamp("2024-05-20")
    assert tuple(round_six.sales.columns) == (
        "Store",
        "Product",
        *BASE_WEEKS,
        *REVEAL_WEEKS[:5],
    )
    assert REVEAL_WEEKS[5] not in round_six.sales
    assert tuple(round_six.in_stock.columns) == ("Store", "Product", *BASE_WEEKS)


def test_round_views_are_defensive_and_reject_out_of_range_rounds(tmp_path: Path) -> None:
    dataset, _data, _inventory, _config = _load(tmp_path)
    first = dataset.round_input(1)
    first.sales.iloc[0, 2] = 999.0
    first.initial_state.iloc[0, 5] = 999.0

    fresh = dataset.round_input(1)
    assert fresh.sales.iloc[0, 2] == 1.0
    assert fresh.initial_state.iloc[0, 5] == 3.0

    for invalid in (0, 7, True, 1.5):
        with pytest.raises(VN2DataError, match="round"):
            dataset.round_input(invalid)  # type: ignore[arg-type]


def test_weekly_actuals_expose_decision_and_drain_sales_without_wide_future(
    tmp_path: Path,
) -> None:
    dataset, _data, _inventory, _config = _load(tmp_path)

    first = dataset.weekly_actuals(1)
    final_drain = dataset.weekly_actuals(8)

    assert first.week_number == 1
    assert first.period == pd.Timestamp(REVEAL_WEEKS[0])
    assert tuple(first.sales.columns) == ("Store", "Product", "sales")
    assert first.sales["sales"].tolist() == [4.0, 103.0]
    assert final_drain.week_number == 8
    assert final_drain.period == pd.Timestamp(REVEAL_WEEKS[-1])
    assert final_drain.sales["sales"].tolist() == [11.0, 110.0]

    final_drain.sales.iloc[0, 2] = 999.0
    assert dataset.weekly_actuals(8).sales.iloc[0, 2] == 11.0
    for invalid in (0, 9, True, 1.5):
        with pytest.raises(VN2DataError, match="week"):
            dataset.weekly_actuals(invalid)  # type: ignore[arg-type]


def test_history_only_in_stock_is_accepted_and_static_at_every_round(tmp_path: Path) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    frame = pd.read_csv(data / "week_0_in_stock.csv")
    _rewrite(
        data,
        inventory,
        "week_0_in_stock.csv",
        frame[["Store", "Product", *BASE_WEEKS]],
    )

    dataset = load_vn2_dataset(data, inventory, load_vn2_config(config_path))

    expected = dataset.round_input(1).in_stock
    assert tuple(expected.columns) == ("Store", "Product", *BASE_WEEKS)
    for round_number in range(2, 7):
        pd.testing.assert_frame_equal(dataset.round_input(round_number).in_stock, expected)


@pytest.mark.parametrize(
    ("fault", "pattern"),
    [
        ("non-prefix", r"in-stock.*(?:prefix|cadence)"),
        ("future", r"in-stock.*prefix"),
        ("invalid-supplied-boolean", r"in-stock.*boolean"),
    ],
)
def test_in_stock_refuses_non_prefix_future_or_invalid_supplied_facts(
    tmp_path: Path,
    fault: str,
    pattern: str,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    frame = pd.read_csv(data / "week_0_in_stock.csv")
    if fault == "non-prefix":
        frame = frame[["Store", "Product", *BASE_WEEKS, *REVEAL_WEEKS[1:]]]
    elif fault == "future":
        frame["2024-06-10"] = [True, False]
    else:
        frame[REVEAL_WEEKS[-1]] = frame[REVEAL_WEEKS[-1]].astype("object")
        frame.loc[0, REVEAL_WEEKS[-1]] = "unknown"
    _rewrite(data, inventory, "week_0_in_stock.csv", frame)

    with pytest.raises(VN2DataError, match=pattern):
        load_vn2_dataset(data, inventory, load_vn2_config(config_path))


def test_alternate_review_cadence_drives_reveals_round_visibility_and_actuals(
    tmp_path: Path,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    sales_names = tuple(f"week_{reveal}_sales.csv" for reveal in range(7))
    for name in set(SALES_FILES) - set(sales_names):
        (data / name).unlink()
    in_stock = pd.read_csv(data / "week_0_in_stock.csv")
    in_stock = in_stock[["Store", "Product", *BASE_WEEKS]]
    in_stock.to_csv(data / "week_0_in_stock.csv", index=False, lineterminator="\n")
    initial = pd.read_csv(data / "week_0_initial_state.csv").drop(columns=["In Transit W+2"])
    initial.to_csv(data / "week_0_initial_state.csv", index=False, lineterminator="\n")

    payload = synthetic_config_payload()
    payload["decision"] = {
        "round_count": 2,
        "lead_time": 1,
        "review_period": 2,
        "protection_period": 3,
        "task_horizon": 3,
        "drain_periods": 1,
        "origins": ["2024-04-22", "2024-05-06"],
    }
    payload["files"]["sales_reveals"] = list(sales_names)
    payload["columns"]["initial_state_columns"].remove("In Transit W+2")
    payload["columns"]["initial_pipeline"] = ["In Transit W+1"]
    write_config(config_path, payload)
    refresh_inventory(data, inventory, names=(*STATIC_FILES, *sales_names))

    dataset = load_vn2_dataset(data, inventory, load_vn2_config(config_path))
    round_one = dataset.round_input(1)
    round_two = dataset.round_input(2)

    assert round_one.reveal_number == 1
    assert tuple(round_one.sales.columns) == (
        "Store",
        "Product",
        *BASE_WEEKS,
        REVEAL_WEEKS[0],
    )
    assert round_two.reveal_number == 3
    assert tuple(round_two.sales.columns) == (
        "Store",
        "Product",
        *BASE_WEEKS,
        *REVEAL_WEEKS[:3],
    )
    assert [dataset.weekly_actuals(week).period for week in range(1, 6)] == [
        pd.Timestamp(week) for week in REVEAL_WEEKS[1:6]
    ]
    with pytest.raises(VN2DataError, match=r"week.*1\.\.5"):
        dataset.weekly_actuals(6)


def _rewrite_all_keys(
    data: Path,
    inventory: Path,
    *,
    store_values: list[object],
) -> None:
    for path in data.glob("*.csv"):
        frame = pd.read_csv(path, dtype={"Store": "string", "Product": "string"})
        frame["Store"] = pd.Series(store_values, dtype="object")
        frame.to_csv(path, index=False, lineterminator="\n")
    refresh_inventory(data, inventory)


def test_integer_key_normalization_preserves_values_above_float_precision(
    tmp_path: Path,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    exact = 2**53 + 1
    _rewrite_all_keys(data, inventory, store_values=[str(exact), "7"])

    dataset = load_vn2_dataset(data, inventory, load_vn2_config(config_path))

    assert dataset.round_input(1).sales["Store"].tolist() == [exact, 7]


@pytest.mark.parametrize(
    ("invalid", "pattern"),
    [
        (2**63, r"Store key.*signed 64-bit integer"),
        (-1, r"Store key.*signed 64-bit integer"),
        (1.5, r"Store key.*signed 64-bit integer"),
        ("1.0", r"Store key.*signed 64-bit integer"),
        (True, r"Store key.*signed 64-bit integer"),
        (None, r"Store/Product key.*missing"),
    ],
    ids=["above-int64", "negative", "fractional", "fractional-string", "boolean", "missing"],
)
def test_integer_key_normalization_refuses_inexact_or_out_of_range_values(
    tmp_path: Path,
    invalid: object,
    pattern: str,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    frame = pd.read_csv(
        data / "week_0_sales.csv",
        dtype={"Store": "string", "Product": "string"},
    )
    frame["Store"] = frame["Store"].astype("object")
    frame.loc[0, "Store"] = invalid
    _rewrite(data, inventory, "week_0_sales.csv", frame)

    with pytest.raises(VN2DataError, match=pattern):
        load_vn2_dataset(data, inventory, load_vn2_config(config_path))


def test_future_reveal_perturbation_cannot_change_current_history_but_prior_can(
    tmp_path: Path,
) -> None:
    baseline, *_ = _load(tmp_path / "baseline")
    future_data, future_inventory, future_config_path = write_dataset(
        tmp_path / "future",
        future_value=999.0,
    )
    future = load_vn2_dataset(
        future_data,
        future_inventory,
        load_vn2_config(future_config_path),
    )
    prior_data, prior_inventory, prior_config_path = write_dataset(
        tmp_path / "prior",
        base_value=777.0,
    )
    prior = load_vn2_dataset(
        prior_data,
        prior_inventory,
        load_vn2_config(prior_config_path),
    )

    pd.testing.assert_frame_equal(baseline.round_input(1).sales, future.round_input(1).sales)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(baseline.round_input(1).sales, prior.round_input(1).sales)


@pytest.mark.parametrize(
    ("fault", "pattern"),
    [
        ("replace-prior", "previously revealed"),
        ("reorder-columns", "append exactly one"),
        ("append-two", "append exactly one"),
        ("reorder-rows", "key order"),
        ("non-monday", "Monday"),
        ("invalid-date", "valid.*YYYY-MM-DD"),
        ("future-gap", "weekly cadence"),
        ("duplicate-key", "duplicate.*Store.*Product"),
        ("missing-key", "key.*missing"),
        ("negative", "non-negative"),
        ("non-finite", "finite"),
    ],
)
def test_loader_refuses_malformed_reveal_growth_and_values(
    tmp_path: Path,
    fault: str,
    pattern: str,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    path = data / (
        "week_0_sales.csv" if fault in {"duplicate-key", "missing-key"} else "week_1_sales.csv"
    )
    frame = pd.read_csv(path)
    if fault == "replace-prior":
        frame.loc[0, BASE_WEEKS[0]] = 999.0
    elif fault == "reorder-columns":
        frame = frame[
            ["Store", "Product", BASE_WEEKS[1], BASE_WEEKS[0], BASE_WEEKS[2], REVEAL_WEEKS[0]]
        ]
    elif fault == "append-two":
        frame[REVEAL_WEEKS[1]] = [1.0, 2.0]
    elif fault == "reorder-rows":
        frame = frame.iloc[::-1]
    elif fault == "non-monday":
        frame = frame.rename(columns={REVEAL_WEEKS[0]: "2024-04-16"})
    elif fault == "invalid-date":
        frame = frame.rename(columns={REVEAL_WEEKS[0]: "2024-13-40"})
    elif fault == "future-gap":
        frame = frame.rename(columns={REVEAL_WEEKS[0]: REVEAL_WEEKS[1]})
    elif fault == "duplicate-key":
        frame.loc[1, "Store"] = frame.loc[0, "Store"]
        frame.loc[1, "Product"] = frame.loc[0, "Product"]
    elif fault == "missing-key":
        frame.loc[0, "Store"] = np.nan
    elif fault == "negative":
        frame.loc[0, REVEAL_WEEKS[0]] = -1.0
    else:
        frame.loc[0, REVEAL_WEEKS[0]] = np.inf
    _rewrite(data, inventory, path.name, frame)

    with pytest.raises(VN2DataError, match=pattern):
        load_vn2_dataset(data, inventory, load_vn2_config(config_path))


@pytest.mark.parametrize(
    ("filename", "fault", "pattern"),
    [
        ("week_0_master.csv", "extra-column", "master.*exact columns"),
        ("week_0_master.csv", "missing-attribute", "master.*exact columns"),
        ("week_0_in_stock.csv", "not-boolean", "in-stock.*boolean"),
        ("week_0_initial_state.csv", "negative-state", "initial.*non-negative"),
        ("week_0_initial_state.csv", "missing-state-column", "initial.*exact columns"),
        ("week_0_initial_state.csv", "wrong-key", "key order"),
    ],
)
def test_loader_refuses_malformed_static_surfaces(
    tmp_path: Path,
    filename: str,
    fault: str,
    pattern: str,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    frame = pd.read_csv(data / filename)
    if fault == "extra-column":
        frame["Unexpected"] = 1
    elif fault == "missing-attribute":
        frame = frame.drop(columns=[MASTER_ATTRIBUTES[-1]])
    elif fault == "not-boolean":
        frame[BASE_WEEKS[0]] = frame[BASE_WEEKS[0]].astype("object")
        frame.loc[0, BASE_WEEKS[0]] = "unknown"
    elif fault == "negative-state":
        frame.loc[0, "In Transit W+1"] = -1.0
    elif fault == "missing-state-column":
        frame = frame.drop(columns=["In Transit W+2"])
    else:
        frame.loc[0, "Product"] = 999
    _rewrite(data, inventory, filename, frame)

    with pytest.raises(VN2DataError, match=pattern):
        load_vn2_dataset(data, inventory, load_vn2_config(config_path))


def test_loader_verifies_cache_before_attempting_any_csv_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    (data / "week_8_sales.csv").write_bytes(b"poison")
    parse_calls = 0

    def fail_if_called(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal parse_calls
        parse_calls += 1
        raise AssertionError("CSV parsing began before exact inventory verification")

    monkeypatch.setattr(pd, "read_csv", fail_if_called)

    with pytest.raises(VN2DataError, match=r"week_8_sales\.csv.*size"):
        load_vn2_dataset(data, inventory, load_vn2_config(config_path))

    assert parse_calls == 0


def test_loader_rehashes_each_selected_file_immediately_before_its_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    later_reveal = data / "week_2_sales.csv"
    original_read_csv = pd.read_csv
    parse_calls = 0

    def mutate_later_reveal_after_first_parse(
        *args: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        nonlocal parse_calls
        frame = original_read_csv(*args, **kwargs)
        parse_calls += 1
        if parse_calls == 1:
            mutated = bytearray(later_reveal.read_bytes())
            mutated[-2] ^= 1
            later_reveal.write_bytes(bytes(mutated))
        return frame

    monkeypatch.setattr(pd, "read_csv", mutate_later_reveal_after_first_parse)

    with pytest.raises(VN2DataError, match=r"week_2_sales\.csv.*sha256"):
        load_vn2_dataset(data, inventory, load_vn2_config(config_path))

    assert parse_calls == 2
