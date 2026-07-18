"""Hand-checkable 12-node crossed lattice fixture for the Gate B prototype.

Six bottom series crossed by two attributes (``channel`` x ``region``) make a
lattice, not a tree: every bottom series has two aggregate parents, so the
fixture exercises the overlapping-membership construction the spec demands
(`[REC-11]`) rather than a single-parent tree. Node rows, in order: bottom
``s1``..``s6``; ``channel=a,b,c``; ``region=east,west``; ``__total__`` — 12
nodes, 24 nonzeros in S, structural weights
``[1,1,1,1,1,1, 2,2,2, 3,3, 6]``.

All numbers are small integers or terminating decimals so a reviewer can
re-derive every recorded result by hand (chapter 50: fixture-arithmetic
numbers are not apparatus — the derivation travels with the assertion).
"""

from __future__ import annotations

import numpy as np

from gate_b_proto.lattice import Lattice, build_lattice

BOTTOM_IDS: tuple[str, ...] = ("s1", "s2", "s3", "s4", "s5", "s6")
ATTRIBUTE_VALUES: dict[str, tuple[str, ...]] = {
    "channel": ("a", "b", "c", "a", "b", "c"),
    "region": ("east", "east", "east", "west", "west", "west"),
}

# Deliberately incoherent base forecast (aggregates are perturbed sums) so the
# projection has work to do. True coherent sums: channel a=50, b=70, c=90;
# region east=60, west=150; total=210.
BASE_FORECAST = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 48.0, 75.0, 95.0, 55.0, 160.0, 200.0])

# Residual basis patterns over T=8 aligned in-sample periods (each mean-free).
_P1 = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
_P2 = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
_P3 = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])

# Per-node residual coefficients on (_P1, _P2, _P3). Aggregate rows carry
# larger scales than bottom rows (variance grows with level, the realistic
# shape), and shared basis patterns induce cross-node correlation so the
# shrinkage estimator has something to shrink.
_RESIDUAL_COEFFICIENTS: tuple[tuple[float, float, float], ...] = (
    (2.0, 1.0, 0.0),  # s1
    (1.0, -2.0, 0.5),  # s2
    (-1.0, 1.0, -0.5),  # s3
    (0.0, 2.0, 1.0),  # s4
    (2.0, 0.0, -1.0),  # s5
    (1.0, 1.0, 1.0),  # s6
    (4.0, 2.0, 1.0),  # channel=a
    (3.0, -1.0, 2.0),  # channel=b
    (2.0, 3.0, -1.0),  # channel=c
    (6.0, 3.0, 1.5),  # region=east
    (5.0, -2.0, -2.0),  # region=west
    (8.0, 4.0, 2.0),  # __total__
)

RESIDUAL_PERIODS = 8


def fixture_lattice() -> Lattice:
    """Return the validated 12-node fixture lattice."""
    return build_lattice(BOTTOM_IDS, ATTRIBUTE_VALUES)


def fixture_residuals() -> np.ndarray:
    """Return the ``(n_nodes, T)`` in-sample residual matrix, one row per node.

    Stands in for the per-origin fitted-values sidecar the production engine
    widens from `(series key, timestamp, model name)` rows (`[REC-5]`): already
    complete, timestamp-aligned, and finite, so the formulations can consume it
    without re-implementing the widening contract.
    """
    basis = np.vstack([_P1, _P2, _P3])
    coefficients = np.asarray(_RESIDUAL_COEFFICIENTS, dtype=np.float64)
    residuals = coefficients @ basis
    if residuals.shape != (12, RESIDUAL_PERIODS):
        raise ValueError("fixture residual shape drifted")
    return residuals
