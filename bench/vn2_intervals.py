"""VN2: compare EnbPI and SPCI intervals on the six decisions.

uv run python bench/vn2_intervals.py data/vn2

Both methods use three-week sales totals at level 0.9, the same seasonal-naive
forecasts, window, and known historical errors. Stockouts stay observed as sales.
"""

from argparse import ArgumentParser

import numpy as np

import vn2
from calibre import (
    Asymmetric,
    BottomUp,
    EnbPI,
    Forecasts,
    LeadTime,
    SeasonalNaive,
    initial_state,
    rolling_forecasts,
)
from calibre.conformal.calibrators.spci import SPCI
from calibre.conformal.targets import columns
from compare import compare

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
enbpi = EnbPI(level, window)
spci = SPCI(level, window, n_lags=7, ahead=vn2.PROTECTION)

# Seed only errors whose full target is known at the first decision.
known = forecasts.origins + vn2.PROTECTION <= data.decisions[0]
residuals = columns(forecasts.residuals(data.panel.values)[known], target.cover(vn2.PROTECTION))
enbpi_state = initial_state(target, enbpi, n_series, vn2.PROTECTION)
enbpi_state["calibrator"] = enbpi.seed(enbpi_state["calibrator"], residuals)
spci_state = initial_state(target, spci, n_series, vn2.PROTECTION)
spci_state["calibrator"] = spci.seed(
    spci_state["calibrator"], residuals, origins=forecasts.origins[known]
)

decisions = np.isin(forecasts.origins, data.decisions)
table, _ = compare(
    Forecasts(forecasts.origins[decisions], forecasts.points[decisions]),
    data.panel.values,
    {"enbpi": enbpi, "spci": spci},
    target,
    Asymmetric(),
    level,
    data.panel.censored,
    states={"enbpi": enbpi_state, "spci": spci_state},
)
print(table.round(3).to_string())
