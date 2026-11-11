"""The run of gate G1, in the child process: full M5 through v2.

    python bench/gate_m5_run.py DATA OUT

It prints `start` before it reads the data and `end` after its last array. Then it
writes the nodes, the origins, the reconciled points and the two thresholds to OUT.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from conformalforecast.nonconformity import AbsoluteResidual, SignedResidual
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

import m5
from calibre import (
    BottomUp,
    Hierarchy,
    Panel,
    Span,
    Split,
    Step,
    reconcile,
    replay,
    rolling_forecasts,
)

FIRST_ORIGIN = "2016-02-21"
LAST_ORIGIN = "2016-04-24"
HORIZON = 28
WINDOW = 28
LEVEL = 0.9


def main(data: Path, out: Path) -> None:
    print("start", flush=True)
    raw = m5.read(data)
    n_series, n_days = raw.values.shape
    panel = Panel(
        series=raw.series,
        periods=raw.dates,
        freq="D",
        data=raw.values.ravel(),
        indptr=np.arange(0, n_series * n_days + 1, n_days, dtype=np.int64),
        starts=np.zeros(n_series, dtype=np.int64),
    )
    hierarchy = Hierarchy.from_attributes(raw.series, raw.attributes, m5.LEVELS)
    truth = hierarchy.aggregate(panel)
    origins = pd.date_range(FIRST_ORIGIN, LAST_ORIGIN, freq="D")
    engine = StatsForecast(models=[SeasonalNaive(season_length=7)], freq="D", n_jobs=12)
    forecasts = rolling_forecasts(panel, [engine], origins, HORIZON).select("SeasonalNaive")
    nodes = reconcile(forecasts, hierarchy, BottomUp())
    band = replay(
        nodes,
        truth,
        target=Step(),
        method=Split(score=AbsoluteResidual(), window=WINDOW),
        level=LEVEL,
    )
    bound = replay(
        nodes,
        truth,
        target=Span(1, 7),
        method=Split(score=SignedResidual(), window=WINDOW),
        level=LEVEL,
    )
    print("end", flush=True)

    total = int(np.flatnonzero(truth.series == "total")[0])
    values = truth.data[truth.indptr[total] : truth.indptr[total + 1]]
    np.save(out / "nodes.npy", np.asarray(hierarchy.nodes, dtype=str))
    np.save(out / "origins.npy", np.asarray(nodes.origins, dtype="datetime64[ns]"))
    np.save(out / "points.npy", nodes.points[0])
    np.save(out / "band.npy", band.threshold)
    np.save(out / "bound.npy", bound.threshold)
    meta = {"total_sum": float(values.astype(np.float64).sum())}
    (out / "meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
