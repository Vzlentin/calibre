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
    refresh_inventory,
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
    assert REVEAL_WEEKS[5] not in round_six.in_stock


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


def test_loader_reverifies_exact_directory_before_each_csv_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    original_read_csv = pd.read_csv
    parse_calls = 0

    def add_late_extra_after_first_parse(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal parse_calls
        parse_calls += 1
        frame = original_read_csv(*args, **kwargs)
        if parse_calls == 1:
            (data / "late-extra.txt").write_text("not approved", encoding="utf-8")
        return frame

    monkeypatch.setattr(pd, "read_csv", add_late_extra_after_first_parse)

    with pytest.raises(VN2DataError, match=r"file-set mismatch.*extra=.*late-extra\.txt"):
        load_vn2_dataset(data, inventory, load_vn2_config(config_path))

    assert parse_calls == 1
