"""VN2: calibrate a three-week decision bound several ways, then play the orders.

uv run --locked python bench/vn2_conformal.py data/vn2
"""

import sys

import numpy as np
from statsforecast.models import WindowAverage

import vn2
from calibre import (
    ACI,
    BottomUp,
    LeadTime,
    QuantileTracker,
    Signed,
    SplitQuantile,
    critical_ratio,
    rolling_forecasts,
)
from calibre.forecast.models.statsforecast import StatsForecastModel
from compare import compare

data = vn2.load(sys.argv[1])
hierarchy, n_series = data.hierarchy, len(data.panel.series)
level = float(critical_ratio(vn2.HOLDING, vn2.SHORTAGE))

# Weekly origins from week 60 to the last decision. The last six are the decisions.
origins = np.arange(60, int(data.decisions[-1]) + 1)
model = StatsForecastModel(WindowAverage(window_size=13))
run = rolling_forecasts(data.panel, hierarchy, model, BottomUp(hierarchy), origins, vn2.PROTECTION)
actuals = hierarchy.aggregate(data.panel.values)
if data.panel.censored is None:
    raise ValueError("the VN2 panel has no censored mask")
censored = hierarchy.any_bottom(data.panel.censored)

calibrators = {
    "split": SplitQuantile(level, window=52),
    "split-pooled": SplitQuantile(level, window=52, groups=hierarchy.level),
    "aci": ACI(SplitQuantile(level, window=52), gamma=0.01),
    "tracker": QuantileTracker(level, lr=0.5),
}
table, runs = compare(
    run, actuals, calibrators, LeadTime(vn2.PROTECTION), Signed(), level, censored
)
decisions = np.searchsorted(origins, data.decisions)
table["vn2_cost"] = [
    vn2.play(data, runs[name].upper[decisions, :n_series, 0]).total_cost for name in table.index
]
print(table.round(3).to_string())
