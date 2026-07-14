"""Build small successor-owned VN2 fixtures for tier-1 conformance tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SALES_FILES = tuple(f"week_{reveal}_sales.csv" for reveal in range(9))
STATIC_FILES = (
    "week_0_master.csv",
    "week_0_in_stock.csv",
    "week_0_initial_state.csv",
)
EXPECTED_FILES = (*STATIC_FILES, *SALES_FILES)
KEY_COLUMNS = ("Store", "Product")
MASTER_ATTRIBUTES = (
    "ProductGroup",
    "Division",
    "Department",
    "DepartmentGroup",
    "StoreFormat",
    "Format",
)
INITIAL_STATE_COLUMNS = (
    "Store",
    "Product",
    "Start Inventory",
    "Sales",
    "Missed Sales",
    "End Inventory",
    "In Transit W+1",
    "In Transit W+2",
    "Holding Cost",
    "Shortage Cost",
    "Cumulative Holding Cost",
    "Cumulative Shortage Cost",
)
BASE_WEEKS = tuple(pd.date_range("2024-03-25", periods=3, freq="W-MON").strftime("%Y-%m-%d"))
REVEAL_WEEKS = tuple(pd.date_range("2024-04-15", periods=8, freq="W-MON").strftime("%Y-%m-%d"))
DECISION_ORIGINS = REVEAL_WEEKS[:6]


def synthetic_config_payload() -> dict[str, Any]:
    """Return a complete small-data configuration with production VN2 cadence."""
    return {
        "schema": 1,
        "dataset": "vn2",
        "series_count": 2,
        "calendar_frequency": "W-MON",
        "history": {
            "first_week": BASE_WEEKS[0],
            "initial_last_week": BASE_WEEKS[-1],
            "initial_periods": len(BASE_WEEKS),
        },
        "decision": {
            "round_count": 6,
            "lead_time": 2,
            "review_period": 1,
            "protection_period": 3,
            "task_horizon": 3,
            "drain_periods": 2,
            "origins": list(DECISION_ORIGINS),
        },
        "cost": {
            "currency": "EUR",
            "underage_rate": 1.0,
            "overage_rate": 0.2,
            "holding_rate": 0.2,
            "shortage_rate": 1.0,
        },
        "actuals_semantics": "censored_sales_surrogate",
        "stockout_rule": "lost-sales",
        "files": {
            "sales_reveals": list(SALES_FILES),
            "master": STATIC_FILES[0],
            "in_stock": STATIC_FILES[1],
            "initial_state": STATIC_FILES[2],
        },
        "columns": {
            "series_keys": list(KEY_COLUMNS),
            "master_attributes": list(MASTER_ATTRIBUTES),
            "initial_state_columns": list(INITIAL_STATE_COLUMNS),
            "initial_on_hand": "End Inventory",
            "initial_pipeline": ["In Transit W+1", "In Transit W+2"],
        },
        "model_config": {
            "backend": "vn2-seasonal-naive-native-median",
            "m": 52,
            "model_name": "vn2-seasonal-naive-native-median",
            "quantile_levels": [0.5],
            "censoring_aware": False,
        },
        "conformal_config": None,
        "ordering_policy": {
            "name": "rs",
            "coverage": None,
            "quantile": 0.5,
            "explicit_decision_fractile": None,
            "reorder_point": None,
            "reorder_point_scale": None,
            "target_cap": None,
            "target_floor": None,
            "target_scale": None,
        },
    }


def write_config(path: Path, payload: dict[str, Any] | None = None) -> Path:
    """Write one test-only YAML configuration and return its path."""
    path.write_text(
        yaml.safe_dump(payload or synthetic_config_payload(), sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_dataset(
    root: Path,
    *,
    base_value: float = 1.0,
    future_value: float = 20.0,
) -> tuple[Path, Path, Path]:
    """Write twelve small challenge-shaped CSVs, config, and digest inventory."""
    data = root / "data"
    data.mkdir(parents=True)
    keys = pd.DataFrame({"Store": [0, 1], "Product": [126, 127]})
    all_weeks = (*BASE_WEEKS, *REVEAL_WEEKS)

    values: dict[str, list[float | None]] = {}
    for index, week in enumerate(all_weeks):
        values[week] = [base_value + index, float(100 + index)]
    values[BASE_WEEKS[1]][1] = None
    values[REVEAL_WEEKS[1]][0] = future_value

    for reveal, filename in enumerate(SALES_FILES):
        visible = (*BASE_WEEKS, *REVEAL_WEEKS[:reveal])
        frame = keys.copy()
        for week in visible:
            frame[week] = values[week]
        frame.to_csv(data / filename, index=False, lineterminator="\n")

    master = keys.copy()
    for index, attribute in enumerate(MASTER_ATTRIBUTES, start=1):
        master[attribute] = [index, index + 10]
    master.to_csv(data / STATIC_FILES[0], index=False, lineterminator="\n")

    in_stock = keys.copy()
    for week in all_weeks:
        in_stock[week] = [True, week != BASE_WEEKS[0]]
    in_stock.to_csv(data / STATIC_FILES[1], index=False, lineterminator="\n")

    initial = keys.copy()
    for column in INITIAL_STATE_COLUMNS[2:]:
        initial[column] = [1.0, 2.0]
    initial["End Inventory"] = [3.0, 4.0]
    initial["In Transit W+1"] = [5.0, 6.0]
    initial["In Transit W+2"] = [7.0, 8.0]
    initial.to_csv(data / STATIC_FILES[2], index=False, lineterminator="\n")

    inventory = refresh_inventory(data, root / "vn2-input-digests.json")
    config = write_config(root / "protocol.yaml")
    return data, inventory, config


def refresh_inventory(data: Path, path: Path) -> Path:
    """Mint only a synthetic test inventory for fixture bytes."""
    entries = []
    for name in EXPECTED_FILES:
        payload = (data / name).read_bytes()
        entries.append(
            {
                "bytes": len(payload),
                "name": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    inventory = {
        "dataset": "vn2",
        "files": entries,
        "minted_run_id": "12345",
        "minted_sha": "a" * 40,
        "schema": 1,
        "source_manifest": "synthetic-vn2-source.json",
        "source_manifest_sha256": hashlib.sha256(b"synthetic-source").hexdigest(),
    }
    path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
