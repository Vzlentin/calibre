"""Gate G2: the cost of the VN2 decision bound against 0.1. It imports no `calibre`.

    uv run python bench/gate_vn2.py data/vn2

One run of `gate_vn2_run.py`, then a checker in another process. Pass: the bottom points
equal the 0.1 baseline bit for bit, the six decision bounds are within 1e-5 of the
baseline, the total cost is at most the baseline cost times 1.0001, and the coverage of
the bound is between tau - 0.01 and tau + 1/53 + 0.01.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import gate
import vn2

SHAPE = (599, 165)
VALUE_SUM = 290_618.0
CENSORED = 10_517
FIRST_ORIGIN = 60
WINDOW = 52
BOUND_TOLERANCE = 1e-5
COST_FACTOR = 1.0001
COVERAGE_MARGIN = 0.01
BASELINE = Path(__file__).resolve().parent / "baselines"


def main(data: Path) -> int:
    raw = vn2.read(data)
    gate.require(raw.values.shape == SHAPE, f"VN2 shape {raw.values.shape} != {SHAPE}")
    total = float(raw.values.astype(np.float64).sum())
    gate.require(total == VALUE_SUM, f"VN2 sum {total} != {VALUE_SUM}")
    gate.require(int(raw.censored.sum()) == CENSORED, "VN2 censored count differs")
    out = Path(tempfile.mkdtemp(prefix="calibre-gate-vn2-"))
    gate.run("gate_vn2_run.py", data, out)
    try:
        gate.check("gate_vn2.py", str(data), str(out))
    finally:
        shutil.rmtree(out)
    print("G2 passes")
    return 0


def checker(data: Path, out: Path) -> None:
    raw = vn2.read(data)
    baseline = np.load(BASELINE / "vn2-0.1.npz")
    summary = json.loads((BASELINE / "vn2-0.1.json").read_text())
    nodes, summing = gate.hierarchy(raw.series, raw.attributes, vn2.LEVELS)
    gate.require(np.array_equal(np.load(out / "nodes.npy"), nodes), "nodes differ from H1")
    origins = np.load(out / "origins.npy")
    expected = raw.weeks[FIRST_ORIGIN : int(raw.decisions[-1]) + 1].to_numpy()
    gate.require(np.array_equal(origins, expected), "the origins differ")

    points = np.load(out / "points.npy")  # [B, 3, O]
    row = {series: index for index, series in enumerate(raw.series)}
    rows = np.array([row[series] for series in baseline["series"]])
    column = {origin: index for index, origin in enumerate(origins)}
    columns = np.array([column[origin] for origin in baseline["origins"]])
    gate.require(points.dtype == np.float32, "points are not float32")
    gate.require(
        np.array_equal(points[rows][:, :, columns], baseline["points"]),
        "the bottom points differ from the baseline",
    )

    upper = np.load(out / "upper.npy")  # [N, 1, O]
    gate.require(
        upper.shape == (len(nodes), 1, len(origins)),
        f"upper shape {upper.shape} != {(len(nodes), 1, len(origins))}",
    )
    gate.require(upper.dtype == np.float32, "upper is not float32")
    decisions = np.array([column[date] for date in baseline["decisions"]])
    bounds = upper[rows][:, 0, decisions]  # [S, 6]
    gate.require(bool(np.isfinite(bounds).all()), "a decision bound is not ready")
    scale = np.maximum(1.0, np.abs(baseline["bounds"]))
    gap = float(np.max(np.abs(bounds.astype(np.float64) - baseline["bounds"]) / scale))
    gate.require(gap <= BOUND_TOLERANCE, f"decision bounds differ by {gap:.3g} of scale")

    in_order = np.empty_like(bounds)
    in_order[rows] = bounds
    played = vn2.play(raw, in_order.T)
    limit = summary["total_cost"] * COST_FACTOR
    gate.require(played.total_cost <= limit, f"cost {played.total_cost:.3f} > {limit:.3f}")

    tau = summary["level"]
    truth = (summing @ raw.values.astype(np.float64)).astype(np.float32)  # [N, T]
    index = raw.weeks.get_indexer(origins)
    steps = np.arange(1, vn2.PROTECTION + 1)
    targets = index[None, :] + steps[:, None]  # [3, O]
    known = targets[-1] < truth.shape[1]
    known_before = np.searchsorted(index + vn2.PROTECTION, index, side="right")
    n_scores = np.minimum(known_before, WINDOW)
    need = int(np.ceil(np.round(1.0 / (1.0 - tau), 9)))
    ready = np.broadcast_to((n_scores >= need)[None, :], (len(nodes), len(origins)))
    gate.require(
        np.array_equal(np.isfinite(upper[:, 0, :]), ready),
        "ready cells of upper differ from the expected readiness",
    )
    gate.require(
        bool(np.all(upper[:, 0, :][~ready] == np.float32(np.inf))),
        "a cell that is not ready is not +inf",
    )
    total = truth[:, np.minimum(targets, truth.shape[1] - 1)].astype(np.float64).sum(axis=1)
    total = total.astype(np.float32)  # [N, O]
    cells = ready & known[None, :]
    coverage = float((total[cells] <= upper[:, 0, :][cells]).mean())
    low, high = tau - COVERAGE_MARGIN, tau + 1 / (WINDOW + 1) + COVERAGE_MARGIN
    gate.require(low <= coverage <= high, f"coverage {coverage:.4f} not in [{low:.4f}, {high:.4f}]")
    print(
        f"checker: cost {played.total_cost:.3f} (baseline {summary['total_cost']:.3f}), "
        f"coverage {coverage:.4f}, bounds within {gap:.2g}"
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check"]:
        try:
            checker(Path(sys.argv[2]), Path(sys.argv[3]))
        except gate.GateError as error:
            print(f"checker: {error}")
            sys.exit(1)
        sys.exit(0)
    try:
        sys.exit(main(Path(sys.argv[1])))
    except gate.GateError as error:
        print(f"G2 fails: {error}")
        sys.exit(1)
