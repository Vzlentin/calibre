"""Gate M5 per-node coverage neutrality by diffing a CA-slice run vs a baseline.

The gate proves a PR did not move M5 per-node interval coverage. It runs the full
predict -> ``wls_struct`` reconcile -> ``mscp``/perhorizon conformal -> score
pipeline on the committed CA_1+CA_2 slice (``tests/baselines/m5/ca-subset-data``)
and diffs the resulting ``coverage-by-node.parquet`` against a baseline resolved
through ``baseline-manifest.json`` under the fixed ``ca-subset`` key.

The diff itself (:func:`assert_coverage_by_node_neutral`) is unit-tested
unconditionally on a synthetic coverage-by-node frame -- a green path plus one
red perturbation per diff branch -- so a broken comparator is caught in CI even
before the baseline lands. The end-to-end pipeline check is the gated part:

* **Baseline-absent skip.** The baseline parquet is minted on Linux by a separate
  step (issue #279). Until it exists the pipeline check skips, so the apparatus
  merges with green, fast CI.
* **Linux-only guard.** The baseline is Linux-x86_64 arch-bound: the sparse MinT
  (bicgstab) reconcile diverges across architectures and per-node coverage is
  integer-count-derived, so only a same-arch run reproduces it.
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
from calibre.reconciliation.summing import TOTAL_LABEL

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
    identical, or a bottom-level count/coverage/error/width beyond ``atol``.
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


def _synthetic_coverage_by_node() -> pd.DataFrame:
    """Build a schema-faithful coverage-by-node frame for unit-testing the diff.

    Carries every column the diff reads across total / item_id / dept_id / bottom
    levels, with a finite-coverage row per level (for value perturbations) and one
    unscored NaN-coverage bottom row (so the ``equal_nan`` path is exercised).
    """
    rows: list[dict[str, object]] = []

    def add(uid: str, level: str, h: int, coverage: float, width: float, scored: int) -> None:
        total = 20
        coverage_error = coverage - 0.9 if pd.notna(coverage) else np.nan
        rows.append(
            {
                UNIQUE_ID: uid,
                LEVEL_COLUMN: level,
                H: h,
                MODEL_NAME: "SeasonalNaive",
                "target_coverage": 0.9,
                "total_rows": total,
                "resolved_rows": total,
                "unresolved_rows": 0,
                "scored_rows": scored,
                "unscored_rows": total - scored,
                "coverage": coverage,
                "mean_interval_width": width,
                "median_interval_width": width,
                "min_interval_width": width,
                "max_interval_width": width,
                "coverage_error": coverage_error,
                "abs_coverage_error": abs(coverage_error) if pd.notna(coverage_error) else np.nan,
                "is_outlier": bool(pd.notna(coverage) and abs(coverage - 0.9) > 0.1),
            }
        )

    add(TOTAL_LABEL, "total", 1, 0.90, 12.0, 20)
    add(TOTAL_LABEL, "total", 2, 0.88, 13.5, 20)
    add("item_id=FOODS_1_001", "item_id", 1, 0.92, 6.0, 18)
    add("item_id=FOODS_1_001", "item_id", 2, 0.85, 6.5, 18)
    add("dept_id=FOODS_1", "dept_id", 1, 0.91, 8.0, 19)
    add("dept_id=FOODS_1", "dept_id", 2, 0.89, 8.5, 19)
    add("FOODS_1_001_CA_1", "bottom", 1, 0.80, 2.0, 10)
    add("FOODS_1_001_CA_1", "bottom", 2, float("nan"), float("nan"), 0)
    add("FOODS_1_002_CA_1", "bottom", 1, 1.00, 1.5, 8)
    return pd.DataFrame(rows)


def _finite_index(frame: pd.DataFrame, level: str) -> int:
    scored = frame.index[(frame[LEVEL_COLUMN] == level) & frame["coverage"].notna()]
    assert len(scored) > 0, f"no scored {level!r} row to perturb"
    return int(scored[0])


def _bottom_index(frame: pd.DataFrame) -> int:
    bottom = frame.index[frame[LEVEL_COLUMN] == _BOTTOM_LEVEL]
    assert len(bottom) > 0, "no bottom row to perturb"
    return int(bottom[0])


def _perturb_aggregate_coverage(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    mutated = frame.copy()
    mutated.loc[_finite_index(mutated, level), "coverage"] += 0.25
    return mutated


def _perturb_bottom_coverage_beyond_atol(frame: pd.DataFrame) -> pd.DataFrame:
    mutated = frame.copy()
    mutated.loc[_finite_index(mutated, _BOTTOM_LEVEL), "coverage"] += 1e-3
    return mutated


def _perturb_bottom_count(frame: pd.DataFrame) -> pd.DataFrame:
    mutated = frame.copy()
    mutated.loc[_bottom_index(mutated), "scored_rows"] += 1
    return mutated


def _perturb_regrouped_node(frame: pd.DataFrame) -> pd.DataFrame:
    mutated = frame.copy()
    idx = _bottom_index(mutated)
    mutated.loc[idx, UNIQUE_ID] = str(mutated.loc[idx, UNIQUE_ID]) + "__REGROUPED"
    return mutated


def _perturb_drop_row(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(frame.index[-1]).reset_index(drop=True)


# Each case maps to the diff branch it must trip; ``match`` anchors that branch's
# failure, not the perturb helper's own setup assert.
_PERTURBATIONS: dict[str, tuple[Callable[[pd.DataFrame], pd.DataFrame], str]] = {
    "total_row_coverage": (
        lambda f: _perturb_aggregate_coverage(f, "total"),
        r"aggregate-level coverage rows",
    ),
    "item_id_row_coverage": (
        lambda f: _perturb_aggregate_coverage(f, "item_id"),
        r"aggregate-level coverage rows",
    ),
    "bottom_coverage_beyond_atol": (
        _perturb_bottom_coverage_beyond_atol,
        r"bottom-level column 'coverage' drifted",
    ),
    "bottom_count_drift": (_perturb_bottom_count, r"bottom-level count column"),
    "regrouped_node_label": (_perturb_regrouped_node, r"node identity"),
    "dropped_node_row": (_perturb_drop_row, r"row count drift"),
}


def test_identical_coverage_is_neutral() -> None:
    """An unchanged frame diffs clean (no false positives)."""
    frame = _synthetic_coverage_by_node()
    assert_coverage_by_node_neutral(frame, frame.copy())


def test_benign_row_reorder_is_neutral() -> None:
    """A pure row reorder is benign -- the diff aligns by key."""
    frame = _synthetic_coverage_by_node()
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert_coverage_by_node_neutral(frame, shuffled)


def test_within_atol_bottom_width_wiggle_is_neutral() -> None:
    """A bottom-level width wiggle within atol does not trip the diff."""
    frame = _synthetic_coverage_by_node()
    nudged = frame.copy()
    nudged.loc[_finite_index(nudged, _BOTTOM_LEVEL), "mean_interval_width"] += 5e-10
    assert_coverage_by_node_neutral(frame, nudged)


def test_tiny_aggregate_width_wiggle_trips_the_diff() -> None:
    """Aggregate rows are exact: even a 5e-10 width wiggle goes red."""
    frame = _synthetic_coverage_by_node()
    nudged = frame.copy()
    nudged.loc[_finite_index(nudged, "total"), "mean_interval_width"] += 5e-10
    with pytest.raises(AssertionError, match=r"aggregate-level coverage rows"):
        assert_coverage_by_node_neutral(frame, nudged)


@pytest.mark.parametrize("name", sorted(_PERTURBATIONS))
def test_perturbation_trips_the_neutrality_diff(name: str) -> None:
    """Each synthetic perturbation drives its target diff branch RED."""
    baseline = _synthetic_coverage_by_node()
    perturb, pattern = _PERTURBATIONS[name]
    candidate = perturb(baseline)
    with pytest.raises(AssertionError, match=pattern):
        assert_coverage_by_node_neutral(baseline, candidate)


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
