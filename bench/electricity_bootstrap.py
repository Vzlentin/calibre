"""Paired temporal bootstrap of fixed electricity test losses, without refitting.

Pointwise empirical sensitivity intervals are conditional on this record, not a
frequentist 95% guarantee or simultaneous inference. Circular sampling mixes reset
regimes. A 48-hour block cannot preserve weekly dependence; even 336 hours spans
only 2.5 block lengths in the default 35-day record.
"""

from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import binom

from electricity_conformal import INTERVAL_METRICS, POINT_METRICS, TestLosses


@dataclass(frozen=True)
class BootstrapResult:
    """Unique point [B,F,3] and interval [B,F,M,2] draws and long summaries."""

    point_summary: pd.DataFrame
    interval_summary: pd.DataFrame
    point_draws: np.ndarray
    interval_draws: np.ndarray


def _integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_losses(losses: TestLosses, period: int) -> None:
    arrays = (
        losses.steps,
        losses.fold_sizes,
        losses.families,
        losses.methods,
        losses.point,
        losses.interval,
        losses.ready,
    )
    if any(not isinstance(value, np.ndarray) for value in arrays):
        raise ValueError("TestLosses fields must be numpy arrays")
    steps, sizes, families, methods, point, interval, ready = arrays
    if (
        steps.ndim != 1
        or steps.dtype != np.dtype("int64")
        or not len(steps)
        or (steps < 0).any()
        or not (np.diff(steps) == 1).all()
    ):
        raise ValueError("steps must be nonempty consecutive nonnegative int64 target indexes")
    if (
        sizes.ndim != 1
        or sizes.dtype.kind not in "iu"
        or not len(sizes)
        or (sizes <= 0).any()
        or sum(map(int, sizes)) != len(steps)
        or (sizes % period).any()
    ):
        raise ValueError("fold_sizes must be positive integer whole cycles summing to n_test")
    for name, labels in (("families", families), ("methods", methods)):
        if (
            labels.ndim != 1
            or labels.dtype.kind != "U"
            or not len(labels)
            or (labels == "").any()
            or len(np.unique(labels)) != len(labels)
        ):
            raise ValueError(f"{name} must contain unique nonempty string labels")
    shape = (len(steps), len(families), len(methods))
    if point.shape != shape[:2] + (3,) or point.dtype != np.dtype("float64"):
        raise ValueError("point must be float64 [T,F,3] loss contributions")
    if interval.shape != shape + (2,) or interval.dtype != np.dtype("float64"):
        raise ValueError("interval must be float64 [T,F,M,2] loss contributions")
    if ready.shape != shape or ready.dtype != np.dtype("bool"):
        raise ValueError("ready must be bool [T,F,M]")


def _metric_columns(losses: TestLosses, interval: bool) -> pd.DataFrame:
    axes = [losses.families]
    names = ["family"]
    if interval:
        axes.append(losses.methods)
        names.append("method")
    axes.append(INTERVAL_METRICS if interval else POINT_METRICS)
    return pd.MultiIndex.from_product(axes, names=[*names, "metric"]).to_frame(index=False)


def _statistics(cycles: np.ndarray, fold_cycles: np.ndarray, *, point: bool) -> np.ndarray:
    """Recompute metrics within original fold-sized slots, then average folds equally."""
    starts = np.r_[0, np.cumsum(fold_cycles)[:-1]]
    divisor = fold_cycles.reshape((1, -1) + (1,) * (cycles.ndim - 2))
    with np.errstate(invalid="ignore", over="ignore"):
        folds = np.add.reduceat(cycles, starts, axis=1) / divisor
        if point:
            folds[..., 0] = np.sqrt(folds[..., 0])
        return folds.mean(axis=1)


def _circular_indices(starts: np.ndarray, block_cycles: int, n_cycles: int) -> np.ndarray:
    """Expand aligned cycle starts circularly, concatenate blocks, and truncate."""
    return ((starts[..., None] + np.arange(block_cycles)) % n_cycles).reshape(len(starts), -1)[
        :, :n_cycles
    ]


def _rank(probability: Fraction, count: int) -> int:
    """One-based inverse-empirical-CDF rank, with exact decimal probability arithmetic."""
    return max(
        1, (count * probability.numerator + probability.denominator - 1) // probability.denominator
    )


def _summary(
    labels: pd.DataFrame,
    source: np.ndarray,
    estimate: np.ndarray,
    draws: np.ndarray,
    unready: np.ndarray,
    tail: Fraction,
    metadata: dict,
) -> pd.DataFrame:
    count = len(draws)
    finite_source = np.isfinite(source).all(axis=0)
    finite = np.isfinite(draws).sum(axis=0)
    estimate = np.where(unready, np.nan, estimate)
    usable = ~unready & finite_source & np.isfinite(estimate) & (finite == count)
    bounds = np.full((2, *estimate.shape), np.nan)
    mc = np.full((2, 2, *estimate.shape), np.nan)
    if usable.any():
        # Select complete metrics, never filter dates or replicas within a metric.
        ordered = np.sort(draws[:, usable], axis=0)
        ranks = [metadata["lower_rank"], metadata["upper_rank"]]
        bounds[:, usable] = ordered[np.asarray(ranks) - 1]
        for endpoint, probability in enumerate((tail, 1 - tail)):
            # Binomial endpoint brackets describe simulation noise, not data uncertainty.
            # Ranks 0 and B+1 intentionally retain their infinite endpoints at small B.
            low = int(binom.ppf(0.025, count, float(probability)))
            high = int(binom.ppf(0.975, count, float(probability))) + 1
            mc[endpoint, 0, usable] = -np.inf if low == 0 else ordered[low - 1]
            mc[endpoint, 1, usable] = np.inf if high > count else ordered[high - 1]
    with np.errstate(invalid="ignore", over="ignore"):
        bootstrap_mean = draws.mean(axis=0)
        bootstrap_bias = bootstrap_mean - estimate
    fields = {
        "estimate": estimate,
        "lower": bounds[0],
        "upper": bounds[1],
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_bias": bootstrap_bias,
        "finite_replicates": finite,
        "status": np.where(
            unready,
            "unready_source",
            np.where(
                ~finite_source, "nonfinite_source", np.where(usable, "ok", "nonfinite_replicates")
            ),
        ),
        "lower_mc_lower": mc[0, 0],
        "lower_mc_upper": mc[0, 1],
        "upper_mc_lower": mc[1, 0],
        "upper_mc_upper": mc[1, 1],
    }
    return labels.assign(**{name: value.reshape(-1) for name, value in fields.items()}, **metadata)


def bootstrap_metrics(
    losses: TestLosses,
    *,
    n_bootstrap: int = 10_000,
    block_length: int = 48,
    period: int = 24,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Resample shared phase-aligned circular blocks across the entire fixed test record.

    Cycle loss means suffice: retain original fold-sized slots and take each slot's
    RMSE square root before averaging folds. All metrics share sampled indexes; no
    models are refitted. Bounds are exact-rank empirical percentiles, not guarantees.
    Any nonfinite source invalidates its metric's CI; any unready source invalidates
    both interval metrics for that pair, including their estimates. Raw draws remain
    intact, with no finite-only filtering, clipping, or percentile interpolation.
    """
    n_bootstrap = _integer("n_bootstrap", n_bootstrap, 2)
    period = _integer("period", period)
    block_length = _integer("block_length", block_length)
    seed = _integer("seed", seed, 0)
    if (
        isinstance(confidence, (bool, np.bool_))
        or not isinstance(confidence, Real)
        or not np.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be a finite number in (0, 1)")
    _validate_losses(losses, period)
    n_test, n_families, n_methods = losses.ready.shape
    if block_length % period or not period <= block_length < n_test:
        raise ValueError("block_length must be whole cycles, shorter than the record")
    n_cycles = n_test // period
    fold_cycles = (losses.fold_sizes // period).astype(np.int64)
    block_cycles = block_length // period
    blocks_per_draw = (n_cycles + block_cycles - 1) // block_cycles
    with np.errstate(invalid="ignore", over="ignore"):
        point_cycles = losses.point.reshape(n_cycles, period, n_families, 3).mean(axis=1)
        interval_cycles = losses.interval.reshape(n_cycles, period, n_families, n_methods, 2).mean(
            axis=1
        )
    point_draws = np.empty((n_bootstrap, n_families, 3), dtype=np.float64)
    interval_draws = np.empty((n_bootstrap, n_families, n_methods, 2), dtype=np.float64)
    rng = np.random.default_rng(seed)  # PCG64, shared by every family, method and metric.
    for first in range(0, n_bootstrap, 256):
        stop = min(first + 256, n_bootstrap)
        starts = rng.integers(n_cycles, size=(stop - first, blocks_per_draw))
        indexes = _circular_indices(starts, block_cycles, n_cycles)
        point_draws[first:stop] = _statistics(point_cycles[indexes], fold_cycles, point=True)
        interval_draws[first:stop] = _statistics(interval_cycles[indexes], fold_cycles, point=False)
    tail = (1 - Fraction(str(confidence))) / 2
    metadata = dict(
        confidence=confidence,
        block_length=block_length,
        period=period,
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_test=n_test,
        n_folds=len(fold_cycles),
        n_cycles=n_cycles,
        block_lengths_in_record=n_test / block_length,
        blocks_per_replicate=blocks_per_draw,
        lower_rank=_rank(tail, n_bootstrap),
        upper_rank=_rank(1 - tail, n_bootstrap),
        mc_confidence=0.95,
    )
    point_summary = _summary(
        _metric_columns(losses, False),
        losses.point,
        _statistics(point_cycles[None], fold_cycles, point=True)[0],
        point_draws,
        np.zeros((n_families, 3), dtype=bool),
        tail,
        metadata,
    )
    interval_summary = _summary(
        _metric_columns(losses, True),
        losses.interval,
        _statistics(interval_cycles[None], fold_cycles, point=False)[0],
        interval_draws,
        np.broadcast_to(~losses.ready.all(axis=0)[..., None], (n_families, n_methods, 2)),
        tail,
        metadata,
    )
    return BootstrapResult(point_summary, interval_summary, point_draws, interval_draws)


def _diagnostics(
    labels: pd.DataFrame, cells: np.ndarray, unready: np.ndarray, period: int
) -> pd.DataFrame:
    finite_source = np.isfinite(cells).all(axis=0)
    n_test = len(cells)
    with np.errstate(invalid="ignore", over="ignore"):
        phase_means = cells.reshape(-1, period, *cells.shape[1:]).mean(axis=0)
        phase_centered = cells - phase_means[np.arange(n_test) % period]
    frames = []
    for lag in (period, 7 * period):
        frame = labels.assign(lag=lag, n_pairs=max(0, n_test - lag))
        for name, values in (("raw", cells), ("phase_centered", phase_centered)):
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                centered = values - values.mean(axis=0)
                denominator = (centered**2).sum(axis=0)
                acf = (
                    (centered[:-lag] * centered[lag:]).sum(axis=0) / denominator
                    if lag < n_test
                    else np.full_like(denominator, np.nan)
                )
            status = np.where(
                unready,
                "unready_source",
                np.where(
                    ~finite_source,
                    "nonfinite_source",
                    np.where(
                        lag >= n_test,
                        "insufficient_history",
                        np.where(
                            denominator == 0,
                            "constant",
                            np.where(np.isfinite(acf), "ok", "nonfinite"),
                        ),
                    ),
                ),
            )
            frame[f"{name}_acf"] = np.where(status == "ok", acf, np.nan).reshape(-1)
            frame[f"{name}_status"] = status.reshape(-1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def loss_diagnostics(losses: TestLosses, period: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return point/interval loss ACF at period and 7*period, with explicit statuses.

    Raw biased ACF uses the global mean and full squared-deviation denominator.
    Positional phase means use the entire record, only for diagnosis. Pair counts
    include every date, never a finite-only subset. Unready interval pairs are
    unavailable. These descriptive ACFs do not establish stationarity or CI validity.
    """
    period = _integer("period", period)
    _validate_losses(losses, period)
    return (
        _diagnostics(
            _metric_columns(losses, False),
            losses.point,
            np.zeros(losses.point.shape[1:], dtype=bool),
            period,
        ),
        _diagnostics(
            _metric_columns(losses, True),
            losses.interval,
            np.broadcast_to(~losses.ready.all(axis=0)[..., None], losses.interval.shape[1:]),
            period,
        ),
    )
