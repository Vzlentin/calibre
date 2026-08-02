"""Validate the one committed full-M5 Gate C result and disposition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from newcalibre.benchmarking import load_m5_gate_c_result, recompute_gate_c_failures

pytestmark = pytest.mark.tier4

PROJECT_ROOT = Path(__file__).parents[2]
RESULT = PROJECT_ROOT / "benchmarks" / "results" / "m5-gate-c"
CONFIG = PROJECT_ROOT / "benchmarks" / "m5" / "gate-c.yaml"
INVENTORY = PROJECT_ROOT / "benchmarks" / "m5" / "m5-inputs.json"
LOCK = PROJECT_ROOT / "uv.lock"

if not RESULT.exists():
    pytest.skip(
        "the one retained-host Gate C result has not been committed", allow_module_level=True
    )


def _result():
    return load_m5_gate_c_result(
        RESULT,
        config_path=CONFIG,
        inventory_path=INVENTORY,
        lock_path=LOCK,
    )


def test_committed_result_recomputes_one_truthful_disposition() -> None:
    """Recompute the exact full identity, completeness, and all four budgets."""
    result = _result()

    assert result.disposition in {"GO", "NO-GO"}
    assert result.disposition == ("GO" if not result.failures else "NO-GO")
    assert result.artifacts.nodes.num_rows == 33_563
    assert result.artifacts.summary["context"]["origin_count"] == 64  # type: ignore[index]
    assert result.artifacts.summary["context"]["horizon"] == 28  # type: ignore[index]


def test_sales_coverage_values_cannot_change_the_disposition() -> None:
    """Keep descriptive sales rates outside every Gate C verdict input."""
    result = _result()
    summary = deepcopy(result.artifacts.summary)
    summary["population"]["coverage"] = 0.0  # type: ignore[index]
    summary["population"]["deviation"] = -0.9  # type: ignore[index]
    for level in summary["levels"]:  # type: ignore[index]
        level["mean_node_coverage"] = 0.0
        level["pooled_coverage"] = 0.0
        level["mean_node_deviation"] = -0.9
        level["pooled_deviation"] = -0.9
    mutated = replace(result.artifacts, summary=summary)

    assert recompute_gate_c_failures(artifacts=mutated, profile=result.profile) == result.failures
