"""Reproduce the advisory calibrated VN2 result without numeric acceptance gates."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from newcalibre.protocols.vn2 import (
    PLATFORM,
    load_vn2_config,
    load_vn2_dataset,
    render_advisory_result,
    run_vn2,
    verify_vn2_inputs,
)

pytestmark = [
    pytest.mark.tier4,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="VN2 advisory evidence is recorded on Ubuntu 24.04 x86_64",
    ),
]

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = PROJECT_ROOT / "benchmarks" / "vn2" / "gate-b-split-window-sum.yaml"
INVENTORY = PROJECT_ROOT / "benchmarks" / "vn2" / "vn2-input-digests.json"
LOCK = PROJECT_ROOT / "uv.lock"
DATA = PROJECT_ROOT / "data" / "vn2"
RESULT = PROJECT_ROOT / "benchmarks" / "vn2" / "results" / "gate-b-advisory.json"


def test_gate_b_advisory_result_reproduces_from_the_generic_engine() -> None:
    """Require exact advisory bytes and recomputable ordinary-ledger facts."""
    verify_vn2_inputs(DATA, INVENTORY)
    config = load_vn2_config(CONFIG)
    result = run_vn2(load_vn2_dataset(DATA, INVENTORY, config))
    rendered = render_advisory_result(
        result,
        config=config,
        config_path=CONFIG,
        input_inventory_path=INVENTORY,
        lock_path=LOCK,
    )

    assert rendered == RESULT.read_bytes()
    advisory = json.loads(rendered)
    counts = advisory["calibration"]["counts"]
    assert advisory["status"] == "advisory"
    assert counts["total"] == len(result.forecasts)
    assert counts["pending"] == 0
    assert counts["resolved"] == counts["scored"] + counts["unscored"]
    assert sum(advisory["calibration"]["unscored_by_cause"].values()) == counts["unscored"]
    assert counts["covered"] <= counts["scored"]
    assert math.isfinite(advisory["calibration"]["realized_coverage"])
    assert all(
        math.isfinite(value) for value in advisory["calibration"]["widths"]["quantiles"].values()
    )

    holding = math.fsum(record.holding.amount for record in result.settlements)
    shortage = math.fsum(record.shortage.amount for record in result.settlements)
    assert advisory["costs"]["holding"] == holding
    assert advisory["costs"]["shortage"] == shortage
    assert advisory["costs"]["total"] == math.fsum((holding, shortage))
    assert all(
        math.isfinite(value) for value in advisory["costs"].values() if isinstance(value, float)
    )

    assert advisory["identity"] == {
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "input_inventory_sha256": hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
        "platform": PLATFORM,
        "session_id": result.session.value,
    }
    assert advisory["semantics"] == {
        "actuals": "censored_sales_surrogate",
        "censoring_contract": "recorded-sales-under-censored-sales-surrogate",
        "scored_series": "recorded-sales",
    }
    assert advisory["clamps"] == {
        "configured": {"upper_cap": None, "upper_floor": None},
        "issued": [],
    }
