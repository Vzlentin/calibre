"""Hierarchy: node labels, the summing matrix from bottoms to nodes, and `aggregate`."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from calibre.data.panel import Panel

_CHUNK = 128  # periods summed at once: bounds the float64 copies


def _series(series: object) -> np.ndarray:
    if not isinstance(series, np.ndarray):
        raise TypeError("series must be a numpy array")
    if series.dtype.kind not in "UO" or series.ndim != 1:
        raise TypeError("series must be a 1-D array of str")
    labels = np.array(series, dtype=str)
    if len(labels) < 2:
        raise ValueError("a hierarchy needs at least two series")
    if len(np.unique(labels)) != len(labels):
        raise ValueError("series ids are not unique")
    return labels


def _levels(levels: object, columns: pd.Index) -> list[list[str]]:
    if levels is None:
        return [[name] for name in columns]
    if isinstance(levels, str) or not isinstance(levels, Sequence):
        raise TypeError("levels must be a sequence of column names or lists of column names")
    parsed = []
    for level in levels:
        names = [level] if isinstance(level, str) else level
        if not isinstance(names, Sequence) or isinstance(names, str):
            raise TypeError("a level is a column name or a list of column names")
        names = list(names)
        if not names or not all(isinstance(name, str) for name in names):
            raise TypeError("a level is a column name or a non-empty list of column names")
        unknown = [name for name in names if name not in columns]
        if unknown:
            raise ValueError(f"attributes have no column {unknown}")
        parsed.append(names)
    return parsed


@dataclass(frozen=True)
class Hierarchy:
    """Nodes `[N]`, bottoms first. `summing` `[N, B]` maps bottoms to nodes.

    `level` is 0 for bottoms, the position of the level from 1 for aggregates, and the
    number of levels plus 1 for the total.
    """

    nodes: np.ndarray
    summing: sp.csr_array
    level: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, np.ndarray) or self.nodes.dtype.kind not in "UO":
            raise TypeError("nodes must be a numpy array of str")
        if not isinstance(self.summing, sp.csr_array):
            raise TypeError("summing must be a scipy csr_array")
        if not isinstance(self.level, np.ndarray) or self.level.dtype.kind not in "iu":
            raise TypeError("level must be a numpy array of int")
        nodes = np.array(self.nodes, dtype=str)
        summing = sp.csr_array(self.summing, dtype=np.float32, copy=True)
        summing.sum_duplicates()
        summing.sort_indices()
        level = np.array(self.level, dtype=np.int32)
        n_nodes, n_bottom = summing.shape
        if nodes.ndim != 1 or len(nodes) != n_nodes or level.shape != (n_nodes,):
            raise ValueError("nodes, summing rows and level need one entry per node")
        if n_nodes < n_bottom or n_bottom == 0:
            raise ValueError("summing needs at least one bottom and one row per bottom")
        if len(np.unique(nodes)) != n_nodes:
            raise ValueError("node labels are not unique")
        if (summing[:n_bottom] != sp.eye_array(n_bottom, dtype=np.float32, format="csr")).nnz:
            raise ValueError("the bottom rows of summing must be the identity")
        if ((summing.data != 0) & (summing.data != 1)).any():
            raise ValueError("summing holds only zeros and ones")
        if (np.diff(summing.indptr) == 0).any():
            raise ValueError("each node needs at least one bottom")
        for array in (nodes, level, summing.data, summing.indices, summing.indptr):
            array.flags.writeable = False
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "summing", summing)
        object.__setattr__(self, "level", level)

    @property
    def n_bottom(self) -> int:
        return int(self.summing.shape[1])

    @classmethod
    def flat(cls, series: np.ndarray) -> "Hierarchy":
        """The bottoms only, with no aggregate."""
        labels = _series(series)
        n = len(labels)
        return cls(labels, sp.eye_array(n, dtype=np.float32, format="csr"), np.zeros(n, np.int32))

    @classmethod
    def from_attributes(
        cls,
        series: np.ndarray,
        attributes: pd.DataFrame,
        levels: Sequence[str | Sequence[str]] | None = None,
    ) -> "Hierarchy":
        """One aggregate per distinct value of each level, then the total."""
        labels = _series(series)
        if not isinstance(attributes, pd.DataFrame):
            raise TypeError("attributes must be a pandas DataFrame")
        parsed = _levels(levels, attributes.columns)
        index = attributes.index.astype(str)
        if index.has_duplicates:
            raise ValueError("attributes has a duplicate series id")
        position = index.get_indexer(labels)
        if (position < 0).any():
            raise ValueError("attributes does not hold every series")
        table = attributes.iloc[position].reset_index(drop=True)
        if table.isna().to_numpy().any():
            raise ValueError("attributes has a missing value")
        n = len(labels)
        seen = {(index,) for index in range(n)} | {tuple(range(n))}
        rows, columns, names, ids = [], [], [], []
        for level_id, columns_of_level in enumerate(parsed, start=1):
            grouped = table.groupby(columns_of_level, sort=True)
            codes = grouped.ngroup().to_numpy()
            keys = grouped.size().index
            order = np.argsort(codes, kind="stable")
            bounds = np.searchsorted(codes[order], np.arange(len(keys) + 1))
            for group, key in enumerate(keys):
                members = order[bounds[group] : bounds[group + 1]]
                signature = tuple(members.tolist())
                if signature in seen:
                    continue
                seen.add(signature)
                values = key if isinstance(key, tuple) else (key,)
                pairs = zip(columns_of_level, values, strict=True)
                names.append("/".join(f"{name}={str(value)}" for name, value in pairs))
                rows.append(np.full(len(members), len(names) - 1, dtype=np.int64))
                columns.append(members)
                ids.append(level_id)
        n_aggregates = len(names)
        row = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        column = np.concatenate(columns) if columns else np.zeros(0, np.int64)
        aggregates = sp.csr_array(
            (np.ones(len(row), np.float32), (row, column)), shape=(n_aggregates, n)
        )
        summing = sp.vstack(
            [
                sp.eye_array(n, dtype=np.float32, format="csr"),
                aggregates,
                sp.csr_array(np.ones((1, n), np.float32)),
            ],
            format="csr",
        )
        nodes = np.concatenate([labels, np.array(names, dtype=str), np.array(["total"])])
        level = np.concatenate(
            [np.zeros(n, np.int32), np.array(ids, np.int32), [len(parsed) + 1]]
        ).astype(np.int32)
        return cls(nodes, sp.csr_array(summing), level)

    def aggregate(self, panel: Panel) -> Panel:
        """The panel of the nodes: sums of member values, in float64, stored as float32."""
        if not isinstance(panel, Panel):
            raise TypeError("aggregate takes a Panel")
        n_bottom = self.n_bottom
        if not np.array_equal(panel.series, self.nodes[:n_bottom]):
            raise ValueError("the panel series must be the bottoms of the hierarchy, in order")
        n_periods = len(panel.periods)
        lengths = np.diff(panel.indptr)
        rows = np.repeat(np.arange(n_bottom), lengths)
        cols = (
            np.repeat(panel.starts, lengths)
            + np.arange(len(panel.data))
            - np.repeat(panel.indptr[:-1], lengths)
        )
        dense = np.zeros((n_bottom, n_periods), np.float32)
        dense[rows, cols] = panel.data
        summing = self.summing.astype(np.float64)
        values = np.empty((len(self.nodes), n_periods), np.float32)
        for first_period in range(0, n_periods, _CHUNK):
            chunk = slice(first_period, first_period + _CHUNK)
            values[:, chunk] = summing @ dense[:, chunk].astype(np.float64)
        del dense
        lasts = panel.starts + lengths - 1
        member_starts = panel.starts[self.summing.indices]
        member_lasts = lasts[self.summing.indices]
        first = np.minimum.reduceat(member_starts, self.summing.indptr[:-1])
        last = np.maximum.reduceat(member_lasts, self.summing.indptr[:-1])
        period = np.arange(n_periods)
        inside = (period >= first[:, None]) & (period <= last[:, None])
        data = values[inside]
        del values
        censored = None
        if panel.censored is not None:
            flags = np.zeros((n_bottom, n_periods), np.float32)
            flags[rows, cols] = panel.censored
            censored = (self.summing @ flags)[inside] > 0
        indptr = np.concatenate([[0], np.cumsum(last - first + 1)]).astype(np.int64)
        return Panel(
            series=self.nodes,
            periods=panel.periods,
            freq=panel.freq,
            data=data,
            indptr=indptr,
            starts=first.astype(np.int64),
            censored=censored,
        )
