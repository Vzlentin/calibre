"""Compare calibrators on the same forecasts: one row of metrics per method.

uv run python bench/compare.py data/vn2

The VN2 script compares EnbPI and SPCI on three-week sales totals at level 0.9,
using the same seasonal-naive forecasts, window, and known historical errors.
It evaluates the six decision origins. Sales during stockouts stay observed.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from calibre.backtest import Forecasts, Replay, replay
from calibre.conformal import Calibrator, Score, State, Target
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
    states: Mapping[str, State] | None = None,
) -> tuple[pd.DataFrame, dict[str, Replay]]:
    """Replay each calibrator and measure it on the cells where it was ready.

    `by` labels nodes, for example `hierarchy.level`, to get one row per method and
    label. A two-sided score reports coverage, width, and interval score. A one-sided
    upper bound reports coverage and the pinball loss of the bound at `level`, the level
    that the calibrators target.
    `states` supplies seeded or saved online states by method name.
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
            state=None if states is None else states.get(name),
        )
        runs[name] = run
        for label in np.unique(labels):
            nodes = labels == label
            rows.append({"method": name, "by": label, **_metrics(run, nodes, level)})
    table = pd.DataFrame(rows).set_index(["method", "by"])
    return (table.droplevel("by") if by is None else table), runs


def _metrics(run: Replay, nodes: np.ndarray, level: float) -> dict[str, float]:
    ready = np.isfinite(run.threshold[:, nodes])
    if ready.ndim == run.target.ndim + 1:
        ready = ready.all(axis=-1)
    target = np.where(ready, run.target[:, nodes], np.nan)
    lower, upper = run.lower[:, nodes], run.upper[:, nodes]
    metrics = {"ready": float(ready.mean()), "coverage": float(coverage(target, lower, upper))}
    if np.isneginf(lower).all():
        metrics["pinball"] = float(pinball(target, upper, level))
    else:
        metrics["width"] = float(width(target, lower, upper))
        metrics["interval_score"] = float(interval_score(target, lower, upper, level))
    return metrics


if __name__ == "__main__":
    from argparse import ArgumentParser

    import vn2
    from calibre import (
        Asymmetric,
        BottomUp,
        EnbPI,
        LeadTime,
        SeasonalNaive,
        initial_state,
        rolling_forecasts,
    )
    from calibre.conformal.calibrators.spci import SPCI

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Directory containing the VN2 release CSVs.")
    args = parser.parse_args()
    data = vn2.load(args.path)
    n_series = len(data.panel.series)
    origins = np.arange(60, int(data.decisions[-1]) + 1)
    forecasts = rolling_forecasts(
        data.panel,
        data.hierarchy,
        SeasonalNaive(52),
        BottomUp(data.hierarchy),
        origins,
        vn2.PROTECTION,
    )
    # Compare bottom series only, without mixing their widths with aggregate totals.
    forecasts = Forecasts(forecasts.origins, forecasts.points[:, :n_series])
    level, window = 0.9, 128
    target = LeadTime(vn2.PROTECTION)
    calibrators = {
        "enbpi": EnbPI(level, window),
        "spci": SPCI(level, window, n_lags=7, ahead=vn2.PROTECTION),
    }
    # Seed only errors whose full target is known at the first decision.
    known = forecasts.origins + vn2.PROTECTION <= data.decisions[0]
    residuals = forecasts.residuals(data.panel.values)[known].sum(axis=-1, keepdims=True)
    states = {}
    for name, calibrator in calibrators.items():
        state = initial_state(target, calibrator, n_series, vn2.PROTECTION)
        if isinstance(calibrator, SPCI):
            state["calibrator"] = calibrator.seed(
                state["calibrator"], residuals, origins=forecasts.origins[known]
            )
        else:
            state["calibrator"] = calibrator.seed(state["calibrator"], residuals)
        states[name] = state
    decisions = np.isin(forecasts.origins, data.decisions)
    table, _ = compare(
        Forecasts(forecasts.origins[decisions], forecasts.points[decisions]),
        data.panel.values,
        calibrators,
        target,
        Asymmetric(),
        level,
        data.panel.censored,
        states=states,
    )
    print(table.round(3).to_string())
