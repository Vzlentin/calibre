"""Seeded synthetic panels for the contract checks.

Values are multiples of 1/64 below 2**17, so float32 holds each value exactly and a float64
sum of a few of them is exact in any order. Each panel is held as a dense [S, T] float64
array with NaN outside the span of each series.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from calibre import Panel

LIMIT = 2**17
FINE = 2**23  # values are integers below FINE, divided by 64


@dataclass(frozen=True)
class Dense:
    ids: tuple[str, ...]
    periods: pd.DatetimeIndex
    freq: str
    values: np.ndarray  # [S, T] float64, NaN outside each span
    censored: np.ndarray | None = None  # [S, T] bool, False outside each span

    @property
    def starts(self) -> np.ndarray:
        return np.array([np.flatnonzero(np.isfinite(row))[0] for row in self.values])

    @property
    def ends(self) -> np.ndarray:
        """Index after the last period of each series."""
        return np.array([np.flatnonzero(np.isfinite(row))[-1] + 1 for row in self.values])


def grid(rng: np.random.Generator, shape, *, coarse: bool = False) -> np.ndarray:
    """Multiples of 1/64 below 2**17. Coarse values take few distinct values, to tie."""
    if coarse:
        return rng.integers(0, 6, shape) * 16.0
    return rng.integers(0, FINE, shape) / 64.0


def make_dense(
    rng: np.random.Generator,
    ids,
    periods: pd.DatetimeIndex,
    freq: str,
    spans,
    *,
    coarse=(),
    censored_share: float | None = None,
) -> Dense:
    """One series per id, with values on the periods [start, end) of its span."""
    values = np.full((len(ids), len(periods)), np.nan)
    for s, (start, end) in enumerate(spans):
        values[s, start:end] = grid(rng, end - start, coarse=s in coarse)
    censored = None
    if censored_share is not None:
        censored = (rng.random(values.shape) < censored_share) & np.isfinite(values)
    return Dense(tuple(ids), periods, freq, values, censored)


def to_panel(dense: Dense) -> Panel:
    """Build the panel with the constructor, in the order of `dense.ids`."""
    finite = np.isfinite(dense.values)
    lengths = finite.sum(axis=1)
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    data = dense.values[finite].astype(np.float32)
    censored = None if dense.censored is None else dense.censored[finite]
    return Panel(
        series=np.array(dense.ids),
        periods=dense.periods,
        freq=dense.freq,
        data=data,
        indptr=indptr,
        starts=dense.starts.astype(np.int64),
        censored=censored,
    )


def to_frame(dense: Dense, *, censored_col: str | None = None) -> pd.DataFrame:
    """The long frame of the panel: unique_id, ds, and y as float32."""
    rows, cols = np.nonzero(np.isfinite(dense.values))
    frame = pd.DataFrame(
        {
            "unique_id": np.array(dense.ids)[rows],
            "ds": dense.periods[cols],
            "y": dense.values[rows, cols].astype(np.float32),
        }
    )
    if censored_col is not None:
        assert dense.censored is not None
        frame[censored_col] = dense.censored[rows, cols]
    return frame


def from_panel(panel: Panel) -> np.ndarray:
    """The dense [S, T] float64 values of a panel, NaN outside each span."""
    values = np.full((len(panel.series), len(panel.periods)), np.nan)
    for s in range(len(panel.series)):
        chunk = panel.data[panel.indptr[s] : panel.indptr[s + 1]]
        values[s, panel.starts[s] : panel.starts[s] + len(chunk)] = chunk
    return values


def with_values_after(dense: Dense, t: int, fill) -> Dense:
    """Replace every value after period t: by `fill`, a scalar or a [S, T] array."""
    values = dense.values.copy()
    after = np.zeros(values.shape, bool)
    after[:, t + 1 :] = True
    after &= np.isfinite(values)
    values[after] = np.broadcast_to(fill, values.shape)[after]
    return replace(dense, values=values)


def truncated(dense: Dense, t: int) -> Dense:
    """Remove the data after period t. The calendar ends at the latest remaining value."""
    values = dense.values[:, : t + 1]
    last = int(np.flatnonzero(np.isfinite(values).any(axis=0))[-1])
    censored = None if dense.censored is None else dense.censored[:, : last + 1]
    return replace(
        dense, periods=dense.periods[: last + 1], values=values[:, : last + 1], censored=censored
    )


def nan_outside(points: np.ndarray, dense: Dense, origin_idx: np.ndarray) -> np.ndarray:
    """Set NaN the points [N, H, O] of each node at the origins outside its span."""
    out = points.copy()
    inside = (dense.starts[:, None] <= origin_idx[None, :]) & (
        origin_idx[None, :] < dense.ends[:, None]
    )
    out[np.broadcast_to(~inside[:, None, :], out.shape)] = np.nan
    return out
