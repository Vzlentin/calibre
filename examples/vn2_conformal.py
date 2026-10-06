"""VN2: calibrate a three-week decision bound several ways, then play the orders.

uv run --locked python examples/vn2_conformal.py data/vn2
"""

import sys

import numpy as np
from statsforecast.models import WindowAverage

from calibre import ACI, BottomUp, QuantileTracker, SplitQuantile, WindowSum, forecast_origins
from calibre.forecast.statsforecast import StatsForecastModel
from calibre_bench import vn2
from calibre_bench.compare import compare

data = vn2.load(sys.argv[1])
hierarchy, n_series = data.hierarchy, len(data.panel.series)
level = vn2.SHORTAGE / (vn2.SHORTAGE + vn2.HOLDING)  # newsvendor critical fractile

# Weekly origins from week 60 to the last decision. The last six are the decisions.
origins = np.arange(60, int(data.decisions[-1]) + 1)
model = StatsForecastModel(WindowAverage(window_size=13))
run = forecast_origins(data.panel, hierarchy, model, BottomUp(hierarchy), origins, vn2.PROTECTION)
actuals = hierarchy.aggregate(data.panel.values)
censored = hierarchy.any_bottom(data.panel.censored)

calibrators = {
    "split": SplitQuantile(window=52),
    "split-pooled": SplitQuantile(window=52, groups=hierarchy.level),
    "aci": ACI(SplitQuantile(window=52), gamma=0.01),
    "tracker": QuantileTracker(lr=0.5),
}
table, runs = compare(run, actuals, calibrators, WindowSum(vn2.PROTECTION), level, censored)
decisions = np.searchsorted(origins, data.decisions)
table["vn2_cost"] = [
    vn2.play(data, runs[name].upper[decisions, :n_series, 0]).total_cost for name in table.index
]
print(table.round(3).to_string())
