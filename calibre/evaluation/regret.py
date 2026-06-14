"""Regret of a realized cost path against a perfect-foresight oracle."""

from __future__ import annotations

import pandas as pd


def compute_regret(realized: pd.Series, oracle: pd.Series) -> float:
    """Return total positive regret of ``realized`` over ``oracle``.

    Regret is ``(realized - oracle).clip(lower=0).sum()`` — the cumulative
    excess cost a policy paid relative to the perfect-foresight oracle.
    Both series must be aligned by index.
    """
    realized_aligned, oracle_aligned = realized.align(oracle)
    diff = (realized_aligned.astype(float) - oracle_aligned.astype(float)).clip(lower=0.0)
    return float(diff.sum())
