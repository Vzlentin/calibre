"""M5 coverage-neutrality gate: diff a CA-slice run against a frozen baseline.

The gate proves a PR did not move M5 per-node interval coverage. It runs the
full predict -> ``wls_struct`` reconcile -> ``mscp``/perhorizon conformal -> score
pipeline on the committed CA_1+CA_2 slice (``tests/baselines/m5/ca-subset-data``)
and diffs the resulting ``coverage-by-node.parquet`` against a baseline resolved
through ``baseline-manifest.json`` under the fixed ``ca-subset`` key.

Two guards keep this green and fast until the baseline lands:

* **Baseline-absent skip.** The baseline parquet is minted on Linux by a separate
  step (issue #279). Until it exists every test here skips, so the apparatus
  merges with green, fast CI.
* **Linux-only guard.** The baseline is Linux-x86_64 arch-bound: the sparse MinT
  (bicgstab) reconcile diverges across architectures and per-node coverage is
  integer-count-derived, so only a same-arch run reproduces it. The heavy
  pipeline test therefore runs on Linux alone; the perturbation cases, which only
  exercise the diff logic on an in-memory frame, are arch-independent.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.m5.coverage import LEVEL_COLUMN, score_resolved_ledger
from calibre.cli.commands import run_config
from calibre.cli.config import load_config
from calibre.core.forecast_frame import MODEL_NAME, UNIQUE_ID, H
from calibre.execution.ledger import resolved_ledger_uri

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / "tests" / "baselines" / "m5" / "baseline-manifest.json"
_CA_ENTRY = json.loads(_MANIFEST.read_text(encoding="utf-8"))["baselines"]["ca-subset"]
_BASELINE_PATH = _REPO_ROOT / _CA_ENTRY["coverage_by_node"]
_CONFIG_PATH = _REPO_ROOT / _CA_ENTRY["config"]

_BASELINE_PRESENT = _BASELINE_PATH.exists()
_IS_LINUX = platform.system() == "Linux"

_SKIP_NO_BASELINE = pytest.mark.skipif(
    not _BASELINE_PRESENT,
    reason=f"ca-subset coverage baseline absent ({_BASELINE_PATH}); minted on Linux by #279",
)
_SKIP_NON_LINUX = pytest.mark.skipif(
    not _IS_LINUX,
    reason="ca-subset baseline is Linux-x86_64 arch-bound; cross-arch reconcile diverges",
)

# Diff column groups against the coverage-by-node schema (see m5_coverage.py).
_KEY_COLS = [UNIQUE_ID, LEVEL_COLUMN, H, MODEL_NAME]
_COUNT_COLS = ["total_rows", "resolved_rows", "unresolved_rows", "scored_rows", "unscored_rows"]
# Bottom-level floats tolerate same-arch summation noise; aggregate floats must
# match exactly (aggregate coverage is integer-count-derived and coherent).
_BOTTOM_ATOL_COLS = [
    "coverage",
    "coverage_error",
    "abs_coverage_error",
    "mean_interval_width",
    "median_interval_width",
    "min_interval_width",
    "max_interval_width",
]
_BOTTOM_LEVEL = "bottom"
_ATOL = 1e-9


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(_KEY_COLS, kind="stable").reset_index(drop=True)


def assert_coverage_by_node_neutral(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    atol: float = _ATOL,
) -> None:
    """Assert ``candidate`` reproduces ``baseline`` per-node coverage.

    Raises ``AssertionError`` on any drift: row-count or per-level cardinality
    change, a regrouped/relabelled node, an aggregate-level row that is not bit
    identical, or a bottom-level coverage/error/width beyond ``atol``.
    """
    assert len(candidate) == len(baseline), (
        f"row count drift: baseline {len(baseline)} vs candidate {len(candidate)}"
    )

    base_card = baseline[LEVEL_COLUMN].value_counts().sort_index()
    cand_card = candidate[LEVEL_COLUMN].value_counts().sort_index()
    assert base_card.equals(cand_card), (
        f"per-level cardinality drift:\nbaseline\n{base_card}\ncandidate\n{cand_card}"
    )

    base = _sorted(baseline)
    cand = _sorted(candidate[list(baseline.columns)])

    pd.testing.assert_frame_equal(
        cand[_KEY_COLS],
        base[_KEY_COLS],
        check_dtype=False,
        obj="node identity (regrouped/relabelled node)",
    )

    agg = base[LEVEL_COLUMN] != _BOTTOM_LEVEL
    pd.testing.assert_frame_equal(
        cand[agg].reset_index(drop=True),
        base[agg].reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
        obj="aggregate-level coverage rows (must be exact)",
    )

    bottom = base[LEVEL_COLUMN] == _BOTTOM_LEVEL
    for col in _COUNT_COLS:
        np.testing.assert_array_equal(
            cand.loc[bottom, col].to_numpy(),
            base.loc[bottom, col].to_numpy(),
            err_msg=f"bottom-level count column {col!r} drifted",
        )
    for col in _BOTTOM_ATOL_COLS:
        np.testing.assert_allclose(
            cand.loc[bottom, col].to_numpy(dtype=float),
            base.loc[bottom, col].to_numpy(dtype=float),
            rtol=0.0,
            atol=atol,
            equal_nan=True,
            err_msg=f"bottom-level column {col!r} drifted beyond atol={atol}",
        )


@_SKIP_NO_BASELINE
@_SKIP_NON_LINUX
@pytest.mark.regression
def test_ca_subset_coverage_is_neutral_against_baseline(tmp_path: Path) -> None:
    """The CA-slice pipeline reproduces the frozen per-node coverage baseline."""
    ledger_path = tmp_path / "forecast-ledger.parquet"
    config = load_config(_CONFIG_PATH)
    config = config.model_copy(
        update={"output": config.output.model_copy(update={"ledger_path": str(ledger_path)})}
    )

    run_config(config)

    resolved_path = Path(resolved_ledger_uri(ledger_path))
    assert resolved_path.exists(), f"streaming run wrote no resolved ledger at {resolved_path}"
    coverage = config.conformal.coverage if config.conformal is not None else 0.9
    artifacts = score_resolved_ledger(resolved_path, coverage=coverage, output_dir=tmp_path)

    candidate = pd.read_parquet(artifacts.coverage_by_node_path)
    baseline = pd.read_parquet(_BASELINE_PATH)

    assert int(candidate["scored_rows"].sum()) > 0, "produced coverage frame is degenerate"
    assert_coverage_by_node_neutral(baseline, candidate)


def _perturb_aggregate_level(baseline: pd.DataFrame, level: str) -> pd.DataFrame:
    mutated = baseline.copy()
    idx = mutated.index[mutated[LEVEL_COLUMN] == level]
    assert len(idx) > 0, f"baseline carries no {level!r} rows to perturb"
    mutated.loc[idx[0], "coverage"] = float(mutated.loc[idx[0], "coverage"]) + 0.25
    return mutated


def _perturb_bottom_beyond_atol(baseline: pd.DataFrame) -> pd.DataFrame:
    mutated = baseline.copy()
    idx = mutated.index[mutated[LEVEL_COLUMN] == _BOTTOM_LEVEL]
    assert len(idx) > 0, "baseline carries no bottom rows to perturb"
    mutated.loc[idx[0], "coverage"] = float(mutated.loc[idx[0], "coverage"]) + 1e-3
    return mutated


def _perturb_regrouped_node(baseline: pd.DataFrame) -> pd.DataFrame:
    mutated = baseline.copy()
    idx = mutated.index[mutated[LEVEL_COLUMN] == _BOTTOM_LEVEL]
    assert len(idx) > 0, "baseline carries no bottom rows to regroup"
    mutated.loc[idx[0], UNIQUE_ID] = str(mutated.loc[idx[0], UNIQUE_ID]) + "__REGROUPED"
    return mutated


_PERTURBATIONS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "total_row_coverage": lambda b: _perturb_aggregate_level(b, "total"),
    "item_id_row_coverage": lambda b: _perturb_aggregate_level(b, "item_id"),
    "bottom_coverage_beyond_atol": _perturb_bottom_beyond_atol,
    "regrouped_node_label": _perturb_regrouped_node,
}


@_SKIP_NO_BASELINE
@pytest.mark.parametrize("name", sorted(_PERTURBATIONS))
def test_perturbed_baseline_trips_the_neutrality_diff(name: str) -> None:
    """Each synthetic perturbation of the baseline drives the diff RED."""
    baseline = pd.read_parquet(_BASELINE_PATH)
    candidate = _PERTURBATIONS[name](baseline)

    with pytest.raises(AssertionError):
        assert_coverage_by_node_neutral(baseline, candidate)
