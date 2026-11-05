"""Gate G1: speed and memory of v2 on full M5. It imports no `calibre`.

    uv run python bench/gate_m5.py data/m5

One warm-up run, then three timed runs of `gate_m5_run.py`. Pass: the median wall time is
under 19.0 s and each peak under 11.9 GB. The checker then recomputes from the raw data
every point, every `BottomUp` sum, and the thresholds of 10,000 cells of each replay,
drawn with a seed that it picks at run time.
"""

import json
import secrets
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
from conformalforecast._quantiles import conformal_quantile
from conformalforecast.base import InsufficientCalibrationSamples
from conformalforecast.nonconformity import AbsoluteResidual, SignedResidual

import gate
import m5

WALL_LIMIT = 19.0
PEAK_LIMIT = 11.9e9
TIMED_RUNS = 3
SHAPE = (30_490, 1_941)
VALUE_SUM = 66_927_173.0
N_NODES = 42_840
FIRST_ORIGIN = 1_849
N_ORIGINS = 64
HORIZON = 28
SEASON = 7
WINDOW = 28
LEVEL = 0.9
SPAN = 7
SAMPLE = 10_000


def main(data: Path) -> int:
    raw = m5.read(data)
    gate.require(raw.values.shape == SHAPE, f"M5 shape {raw.values.shape} != {SHAPE}")
    total = float(raw.values.astype(np.float64).sum())
    gate.require(total == VALUE_SUM, f"M5 sum {total} != {VALUE_SUM}")
    runs, out = [], None
    for index in range(1 + TIMED_RUNS):
        if out is not None:
            shutil.rmtree(out)
        out = Path(tempfile.mkdtemp(prefix="calibre-gate-m5-"))
        result = gate.run("gate_m5_run.py", data, out)
        label = "warm-up" if index == 0 else f"run {index}"
        print(f"{label}: {result.wall:.2f} s, peak {result.peak / 1e9:.2f} GB", flush=True)
        if index > 0:
            runs.append(result)
    assert out is not None
    gate.check("gate_m5.py", str(data), str(out))
    shutil.rmtree(out)
    wall = statistics.median(run.wall for run in runs)
    peak = max(run.peak for run in runs)
    print(f"median {wall:.2f} s (limit {WALL_LIMIT} s), peak {peak / 1e9:.2f} GB (limit 11.9 GB)")
    passed = wall < WALL_LIMIT and peak < PEAK_LIMIT
    print("G1 passes" if passed else "G1 fails")
    return 0 if passed else 1


def checker(data: Path, out: Path) -> None:
    raw = m5.read(data)
    nodes, summing = gate.hierarchy(raw.series, raw.attributes, m5.LEVELS)
    gate.require(len(nodes) == N_NODES, f"{len(nodes)} nodes != {N_NODES}")
    written = np.load(out / "nodes.npy")
    gate.require(np.array_equal(written, nodes), "the nodes differ from the rule H1")
    meta = json.loads((out / "meta.json").read_text())
    gate.require(meta["total_sum"] == VALUE_SUM, f"total node sum {meta['total_sum']}")
    expected_origins = raw.dates[FIRST_ORIGIN : FIRST_ORIGIN + N_ORIGINS].to_numpy()
    gate.require(
        np.array_equal(np.load(out / "origins.npy"), expected_origins), "the origins differ"
    )

    origins = np.arange(FIRST_ORIGIN, FIRST_ORIGIN + N_ORIGINS)
    steps = np.arange(HORIZON)
    lag = origins[None, :] - SEASON + 1 + (steps[:, None] % SEASON)  # [H, O]
    bottom = raw.values[:, lag]  # [B, H, O]: the same weekday in the last week of the window
    points = (summing @ bottom.reshape(len(raw.series), -1).astype(np.float64)).astype(np.float32)
    points = points.reshape(len(nodes), HORIZON, N_ORIGINS)
    written = np.load(out / "points.npy")
    gate.require(written.dtype == np.float32, "points are not float32")
    gate.require(np.array_equal(written, points), "the reconciled points differ")

    truth = (summing @ raw.values.astype(np.float64)).astype(np.float32)  # [N, T]
    rng = np.random.default_rng(secrets.randbits(64))
    _thresholds(rng, out / "band.npy", points, truth, origins, AbsoluteResidual(), span=None)
    _thresholds(rng, out / "bound.npy", points, truth, origins, SignedResidual(), span=SPAN)
    print("checker: points, sums and sampled thresholds agree")


def _thresholds(rng, path, points, truth, origins, score, span) -> None:
    written = np.load(path)
    n_nodes = points.shape[0]
    n_columns = HORIZON if span is None else 1
    gate.require(written.shape == (n_nodes, n_columns, N_ORIGINS), f"{path.name} shape")
    alpha = 1.0 - LEVEL
    for _ in range(SAMPLE):
        node, column, origin = (
            int(rng.integers(n_nodes)),
            int(rng.integers(n_columns)),
            int(rng.integers(N_ORIGINS)),
        )
        last = column + 1 if span is None else span
        first = column + 1 if span is None else 1
        known = origin - last  # daily origins: the score of origin k is known at k + last
        window = np.arange(max(0, known - WINDOW + 1), known + 1)
        if len(window):
            point = points[node, first - 1 : last][:, window].astype(np.float64).sum(axis=0)
            target_index = origins[window][None, :] + np.arange(first, last + 1)[:, None]
            value = truth[node][target_index].astype(np.float64).sum(axis=0)
            scores = score.score(point.astype(np.float32), value.astype(np.float32))
            try:
                expected = conformal_quantile(scores[None, :, None], alpha)[0, 0]
            except InsufficientCalibrationSamples:
                expected = np.float32(np.inf)
        else:
            expected = np.float32(np.inf)
        got = written[node, column, origin]
        gate.require(
            np.float32(got) == np.float32(expected),
            f"{path.name}[{node}, {column}, {origin}] = {got}, expected {expected}",
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
        print(f"G1 fails: {error}")
        sys.exit(1)
