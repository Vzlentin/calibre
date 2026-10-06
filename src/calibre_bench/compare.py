"""Compare calibrators on the same forecasts: one row of metrics per method."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from calibre.conformal import Calibrator, Score
from calibre.conformal.origins import Calibrated, Conformal
from calibre.evaluate import coverage, interval_score, pinball, width
from calibre.forecast.origins import Forecasts


def compare(
    forecasts: Forecasts,
    actuals: np.ndarray,
    calibrators: Mapping[str, Calibrator],
    score: Score,
    level: float,
    censored: np.ndarray | None = None,
    by: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Calibrated]]:
    """Replay each calibrator and measure it on the cells where it was ready.

    `by` labels nodes, for example `hierarchy.level`, to get one row per method and
    label. A two-sided score reports coverage, width, and interval score. A one-sided
    upper bound reports coverage and the pinball loss of the bound at `level`.
    `ready` is the share of cells with a finite threshold. The runs are returned too.
    """
    rows, runs = [], {}
    labels = np.zeros(actuals.shape[0], dtype=int) if by is None else np.asarray(by)
    for name, calibrator in calibrators.items():
        run = Conformal(score, calibrator, level).replay(forecasts, actuals, censored)
        runs[name] = run
        for label in np.unique(labels):
            nodes = labels == label
            rows.append({"method": name, "by": label, **_metrics(run, nodes, level)})
    table = pd.DataFrame(rows).set_index(["method", "by"])
    return (table.droplevel("by") if by is None else table), runs


def _metrics(run: Calibrated, nodes: np.ndarray, level: float) -> dict[str, float]:
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
