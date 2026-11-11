"""Direct numpy computations of the expected values of `replay`, from the rules of the contract."""

import numpy as np

NOISE_DECIMALS = 9  # conformalforecast rounds to 9 decimals before each ceiling


def cover_step(n_nodes: int, horizon: int) -> np.ndarray:
    """[N, H, H] bool: column c covers step c + 1."""
    return np.broadcast_to(np.eye(horizon, dtype=bool), (n_nodes, horizon, horizon)).copy()


def cover_span(n_nodes: int, horizon: int, first, last) -> np.ndarray:
    """[N, 1, H] bool: the steps first to last of each node, 1-based."""
    first = np.broadcast_to(np.asarray(first), (n_nodes,))
    last = np.broadcast_to(np.asarray(last), (n_nodes,))
    steps = np.arange(1, horizon + 1)
    covered = (steps[None, :] >= first[:, None]) & (steps[None, :] <= last[:, None])
    return covered[:, None, :]


def columns(points: np.ndarray, values: np.ndarray, origin_idx: np.ndarray, cover: np.ndarray):
    """Point and truth [N, C, O] float32 of each column, summed in float64.

    points: [N, H, O] float32. values: [N, T] float64, NaN outside spans. A truth with a step
    after the last period is NaN.
    """
    n_nodes, horizon, n_origins = points.shape
    n_periods = values.shape[1]
    padded = np.concatenate([values, np.full((n_nodes, horizon), np.nan)], axis=1)
    steps = origin_idx[:, None] + np.arange(1, horizon + 1)[None, :]  # [O, H]
    step_truth = padded[:, steps]  # [N, O, H]
    step_truth[:, steps >= n_periods] = np.nan
    step_points = np.moveaxis(points.astype(np.float64), 2, 1)  # [N, O, H]
    point = np.empty((n_nodes, cover.shape[1], n_origins), np.float32)
    truth = np.empty_like(point)
    for c in range(cover.shape[1]):
        weights = cover[:, c, :]  # [N, H]
        point[:, c, :] = masked_sum(step_points, weights)
        truth[:, c, :] = masked_sum(step_truth, weights)
    return point, truth


def masked_sum(step_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sum [N, O, H] over the covered steps [N, H]: NaN when one covered value is NaN."""
    covered = np.where(weights[:, None, :], step_values, 0.0)
    total = covered.sum(axis=2)
    total[np.isnan(covered).any(axis=2)] = np.nan
    return total.astype(np.float32)


def censored_cells(censored, origin_idx, cover, truth) -> np.ndarray:
    """[N, C, O] bool: one censored step value, and a truth that is not NaN."""
    n_nodes, n_periods = censored.shape
    horizon = cover.shape[2]
    padded = np.concatenate([censored, np.zeros((n_nodes, horizon), bool)], axis=1)
    steps = origin_idx[:, None] + np.arange(1, horizon + 1)[None, :]
    step_flags = padded[:, steps]  # [N, O, H]
    out = np.zeros(truth.shape, bool)
    for c in range(cover.shape[1]):
        out[:, c, :] = (step_flags & cover[:, c, None, :]).any(axis=2)
    return out & ~np.isnan(truth)


def scores(point: np.ndarray, truth: np.ndarray, *, signed: bool) -> np.ndarray:
    residual = truth.astype(np.float64) - point.astype(np.float64)
    return (residual if signed else np.abs(residual)).astype(np.float32)


def last_steps(cover: np.ndarray) -> np.ndarray:
    """[N, C]: the last step of each column, 1-based."""
    horizon = cover.shape[2]
    return horizon - np.argmax(cover[:, :, ::-1], axis=2)


def members(origin_idx: np.ndarray, last: np.ndarray, window: int) -> list:
    """For each node, column and origin, the origins of its score window, oldest first.

    A score of origin j is known at origin o when origin_idx[j] + last <= origin_idx[o]. The
    window holds the `window` latest known origins.
    """
    n_nodes, n_cols = last.shape
    out = [[[None] * len(origin_idx) for _ in range(n_cols)] for _ in range(n_nodes)]
    for n in range(n_nodes):
        for c in range(n_cols):
            known_until = np.searchsorted(origin_idx + last[n, c], origin_idx, side="right")
            for o, count in enumerate(known_until):
                out[n][c][o] = np.arange(max(0, count - window), count)
    return out


def ceil9(x: float) -> int:
    return int(np.ceil(np.round(np.float64(x), NOISE_DECIMALS)))


def rank_thresholds(score: np.ndarray, windows: list, alphas: np.ndarray) -> np.ndarray:
    """The conformal rank of the finite scores of each window, +inf below the minimum."""
    n_nodes, n_cols, n_origins = score.shape
    out = np.full(score.shape, np.inf, np.float32)
    for n in range(n_nodes):
        alpha = float(alphas[n])
        need = ceil9(1.0 / alpha)
        for c in range(n_cols):
            for o in range(n_origins):
                values = score[n, c, windows[n][c][o]]
                values = np.sort(values[np.isfinite(values)])
                count = len(values)
                if count >= need:
                    k = min(ceil9((1.0 - alpha) * (count + 1)), count)
                    out[n, c, o] = values[k - 1]
    return out


def window_counts(score: np.ndarray, windows: list) -> np.ndarray:
    """[N, C, O]: the number of finite scores in each window."""
    out = np.zeros(score.shape, np.int64)
    for n in range(score.shape[0]):
        for c in range(score.shape[1]):
            for o in range(score.shape[2]):
                out[n, c, o] = np.isfinite(score[n, c, windows[n][c][o]]).sum()
    return out


def assert_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    """Same dtype, shape and NaN cells, and the same bits in every other cell."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    nan_a = np.isnan(actual)
    nan_e = np.isnan(expected)
    bad_nan = np.argwhere(nan_a != nan_e)
    assert bad_nan.size == 0, f"NaN differs at {bad_nan[:5].tolist()}"
    width = {2: np.uint16, 4: np.uint32, 8: np.uint64}[actual.dtype.itemsize]
    same = actual.view(width) == expected.view(width)
    bad = np.argwhere(~same & ~nan_a)
    assert bad.size == 0, (
        f"{len(bad)} cells differ, first at {bad[:5].tolist()}: "
        f"{actual[tuple(bad[0])]!r} != {expected[tuple(bad[0])]!r}"
    )
