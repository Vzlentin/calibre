"""Compare calibrators on the same forecasts: one row of metrics per method."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from calibre.backtest import Forecasts, Replay, replay
from calibre.conformal import Calibrator, Score, Target
from calibre.metrics import coverage, interval_score, pinball, width


def compare(
    forecasts: Forecasts,
    actuals: np.ndarray,
    calibrators: Mapping[str, Calibrator],
    target: Target,
    score: Score,
    level: float,
    censored: np.ndarray | None = None,
    by: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Replay]]:
    """Replay each calibrator and measure it on the cells where it was ready.

    `by` labels nodes, for example `hierarchy.level`, to get one row per method and
    label. A two-sided score reports coverage, width, and interval score. A one-sided
    upper bound reports coverage and the pinball loss of the bound at `level`, the level
    that the calibrators target.
    `ready` is the share of cells with a finite threshold. The runs are returned too.
    """
    rows, runs = [], {}
    labels = np.zeros(actuals.shape[0], dtype=int) if by is None else np.asarray(by)
    for name, calibrator in calibrators.items():
        run = replay(
            forecasts,
            actuals,
            target=target,
            score=score,
            calibrator=calibrator,
            censored=censored,
        )
        runs[name] = run
        for label in np.unique(labels):
            nodes = labels == label
            rows.append({"method": name, "by": label, **_metrics(run, nodes, level)})
    table = pd.DataFrame(rows).set_index(["method", "by"])
    return (table.droplevel("by") if by is None else table), runs


def _metrics(run: Replay, nodes: np.ndarray, level: float) -> dict[str, float]:
    ready = np.isfinite(run.threshold[:, nodes])
    target = np.where(ready, run.target[:, nodes], np.nan)
    lower, upper = run.lower[:, nodes], run.upper[:, nodes]
    metrics = {"ready": float(ready.mean()), "coverage": float(coverage(target, lower, upper))}
    if np.isneginf(lower).all():
        metrics["pinball"] = float(pinball(target, upper, level))
    else:
        metrics["width"] = float(width(target, lower, upper))
        metrics["interval_score"] = float(interval_score(target, lower, upper, level))
    return metrics
