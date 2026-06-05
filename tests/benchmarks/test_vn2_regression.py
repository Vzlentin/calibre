"""x86_64-gated VN2 winning-config regression gate (architecture-deepenings U0).

Pins the documented baseline ``total_cost = 4992.20`` (EUR) — the cross-cutting
invariant the C1–C5 architecture deepenings must preserve. The number lives in
CLAUDE.md / README as prose; this module makes it *executable* so it is enforced
on every merge rather than merely documented.

Two independent production code paths reproduce the baseline; the gate pins both,
so a refactor that drifts either entry point is caught:

* ``run_benchmark(tune=False)`` — the ``DecisionLoop`` winning path, i.e. what
  ``python -m benchmarks.vn2 --config winning.yaml`` runs.
* ``replay_cached_cost(build_replay_cache())`` — the ``BackendEngine``
  multi-origin warmup + manual ``VN2Simulator`` replay used by the HPO cost
  oracle. Distinct orchestration, same engine / ledger / conformal / ordering
  stack underneath.

Both reproduce the baseline bit-for-bit on x86_64 (total + holding + shortage),
so all three buckets are pinned — that catches compensating per-bucket drift the
bare scalar would miss.

**Platform.** 4992.20 is the **x86_64/Linux** value, which is where CI runs. On
arm64/macOS the same config deterministically yields ~5011.20 — cross-arch
LightGBM float divergence (SIMD/FMA/libm), **not** a regression — so the gate
skips there instead of failing. See CLAUDE.md "Gotchas".

**Cost.** Each path trains an 800-estimator global LightGBM over the full
600-series VN2 panel across 6 decision rounds (~90s each), so the two heavy
tests carry the ``regression`` marker. Deselect locally with
``-m "not regression"``; CI runs the full suite, so the gate fires on every PR.
"""

from __future__ import annotations

import platform

import pandas as pd
import pytest

from benchmarks.vn2.replay import build_replay_cache, replay_cached_cost
from benchmarks.vn2.run_benchmark import run_benchmark

# Golden baseline (x86_64/Linux), verified 2026-06-05 against both paths.
# Costs are unit-quantized (holding 0.20 EUR/unit-week, shortage 1.00 EUR/unit),
# so the baseline is exact; the tolerance only absorbs float summation order and
# is far below the smallest meaningful drift (one unit-week of holding = 0.20).
BASELINE_TOTAL_COST = 4992.20
BASELINE_HOLDING_COST = 2488.20
BASELINE_SHORTAGE_COST = 2504.00
BASELINE_ABS_TOL = 0.01

_IS_X86_64 = platform.machine().lower() in {"x86_64", "amd64"}

# Applied to the two model-running paths only — the bites test below is pure
# arithmetic, so it stays cheap, platform-independent, and always selected.
_x86_64_gate = pytest.mark.skipif(
    not _IS_X86_64,
    reason=(
        f"VN2 baseline {BASELINE_TOTAL_COST} is the x86_64/Linux gate value; "
        f"on {platform.machine()} the same config deterministically yields a "
        "different cost (cross-arch LightGBM float divergence, not a "
        "regression). See CLAUDE.md Gotchas."
    ),
)


def _assert_baseline(summary: pd.DataFrame, *, path: str) -> None:
    """Assert a per-product cost summary reproduces the pinned VN2 baseline."""
    total = float(summary["total_cost"].sum())
    holding = float(summary["holding_cost"].sum())
    shortage = float(summary["shortage_cost"].sum())
    assert total == pytest.approx(BASELINE_TOTAL_COST, abs=BASELINE_ABS_TOL), (
        f"{path}: VN2 total_cost drifted to {total:.4f}; the x86_64 baseline is "
        f"{BASELINE_TOTAL_COST}. Do not loosen this — see CLAUDE.md Gotchas."
    )
    assert holding == pytest.approx(BASELINE_HOLDING_COST, abs=BASELINE_ABS_TOL), (
        f"{path}: holding_cost drifted to {holding:.4f} (baseline {BASELINE_HOLDING_COST})."
    )
    assert shortage == pytest.approx(BASELINE_SHORTAGE_COST, abs=BASELINE_ABS_TOL), (
        f"{path}: shortage_cost drifted to {shortage:.4f} (baseline {BASELINE_SHORTAGE_COST})."
    )


@pytest.mark.regression
@_x86_64_gate
def test_winning_config_baseline_via_run_benchmark(data_dir) -> None:
    """Path 1 — the DecisionLoop winning path pins total_cost=4992.20."""
    summary = run_benchmark(data_dir=data_dir, verbose=False)
    _assert_baseline(summary, path="run_benchmark/DecisionLoop")


@pytest.mark.regression
@_x86_64_gate
def test_winning_config_baseline_via_replay(data_dir) -> None:
    """Path 2 — BackendEngine multi-origin warmup + VN2Simulator replay."""
    cache = build_replay_cache(data_dir=data_dir)
    result = replay_cached_cost(cache)
    _assert_baseline(result.summary, path="replay/BackendEngine-multi-origin")


def test_gate_bites_on_sub_euro_drift() -> None:
    """The tolerance rejects even a single unit-week of drift (no model run).

    Proves the gate actually bites: a witness at the exact baseline passes, but
    perturbing any cost bucket — by ±1.00 EUR or even the 0.20 EUR holding
    quantum — fails ``_assert_baseline``.
    """
    witness = pd.DataFrame(
        {
            "unique_id": ["x"],
            "holding_cost": [BASELINE_HOLDING_COST],
            "shortage_cost": [BASELINE_SHORTAGE_COST],
            "total_cost": [BASELINE_TOTAL_COST],
        }
    )
    _assert_baseline(witness, path="witness")  # exact baseline must pass

    for column, delta in [
        ("total_cost", 1.0),
        ("total_cost", -1.0),
        ("holding_cost", 0.20),
        ("shortage_cost", 1.0),
    ]:
        perturbed = witness.copy()
        perturbed[column] = perturbed[column] + delta
        with pytest.raises(AssertionError):
            _assert_baseline(perturbed, path=f"perturbed[{column}+{delta}]")
