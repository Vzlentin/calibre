"""The run of gate G2, in the child process: the VN2 decision bound through v2.

    python bench/gate_vn2_run.py DATA OUT

It prints `start` before it reads the data and `end` after its last array. Then it
writes the nodes, the origins, the bottom points and the upper bounds to OUT.
"""

import sys
from pathlib import Path

import numpy as np
from conformalforecast.nonconformity import SignedResidual
from statsforecast import StatsForecast
from statsforecast.models import WindowAverage

import vn2
from calibre import BottomUp, Hierarchy, Panel, Span, Split, reconcile, replay, rolling_forecasts
from calibre.decision import critical_ratio

FIRST_ORIGIN = 60
WINDOW = 52


def main(data: Path, out: Path) -> None:
    print("start", flush=True)
    raw = vn2.read(data)
    n_series, n_weeks = raw.values.shape
    panel = Panel(
        series=raw.series,
        periods=raw.weeks,
        freq=vn2.FREQ,
        data=raw.values.ravel(),
        indptr=np.arange(0, n_series * n_weeks + 1, n_weeks, dtype=np.int64),
        starts=np.zeros(n_series, dtype=np.int64),
        censored=raw.censored.ravel(),
    )
    hierarchy = Hierarchy.from_attributes(raw.series, raw.attributes, vn2.LEVELS)
    truth = hierarchy.aggregate(panel)
    origins = raw.weeks[FIRST_ORIGIN : int(raw.decisions[-1]) + 1]
    engine = StatsForecast(models=[WindowAverage(window_size=13)], freq=vn2.FREQ)
    forecasts = rolling_forecasts(panel, [engine], origins, vn2.PROTECTION)
    forecasts = forecasts.select("WindowAverage")
    nodes = reconcile(forecasts, hierarchy, BottomUp())
    bound = replay(
        nodes,
        truth,
        target=Span(1, vn2.PROTECTION),
        method=Split(score=SignedResidual(), window=WINDOW),
        level=critical_ratio(vn2.HOLDING, vn2.SHORTAGE),
    )
    print("end", flush=True)

    np.save(out / "nodes.npy", np.asarray(hierarchy.nodes, dtype=str))
    np.save(out / "origins.npy", np.asarray(nodes.origins, dtype="datetime64[ns]"))
    np.save(out / "points.npy", forecasts.points[0])
    np.save(out / "upper.npy", bound.upper)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
