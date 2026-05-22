from __future__ import annotations

import pandas as pd

from calibre.evaluation.regret import compute_regret


def test_compute_regret_sums_positive_excess() -> None:
    realized = pd.Series([10.0, 8.0, 12.0])
    oracle = pd.Series([7.0, 9.0, 10.0])
    assert compute_regret(realized, oracle) == 5.0


def test_compute_regret_zero_when_realized_below_oracle() -> None:
    realized = pd.Series([5.0, 6.0])
    oracle = pd.Series([10.0, 10.0])
    assert compute_regret(realized, oracle) == 0.0


def test_compute_regret_aligns_indexes() -> None:
    realized = pd.Series([10.0, 8.0], index=["a", "b"])
    oracle = pd.Series([6.0, 9.0], index=["b", "a"])
    assert compute_regret(realized, oracle) == 1.0 + 2.0


def test_compute_regret_handles_empty_series() -> None:
    assert compute_regret(pd.Series([], dtype=float), pd.Series([], dtype=float)) == 0.0
