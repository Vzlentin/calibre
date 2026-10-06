"""Hierarchy: node labels and the summing matrix from bottom series to nodes."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class Hierarchy:
    """Rows of `summing` `[N, B]` follow `nodes`: bottoms first, as the identity.

    `level[node]` is 0 for bottoms, one id per attribute column in order, and the last
    id for the total.
    """

    nodes: np.ndarray
    summing: sp.csr_array
    level: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.array(self.nodes, dtype=str)
        summing = sp.csr_array(self.summing, dtype=np.float32, copy=True)
        level = np.array(self.level, dtype=np.int32)
        n_nodes, n_bottom = summing.shape
        if n_nodes < n_bottom:
            raise ValueError("hierarchy summing has fewer rows than bottom series")
        if len(nodes) != n_nodes or level.shape != (n_nodes,):
            raise ValueError(
                "hierarchy nodes, summing rows, and level must have one entry per node"
            )
        if len(np.unique(nodes)) != n_nodes:
            raise ValueError("hierarchy node labels are not unique")
        if (summing[:n_bottom] != sp.eye_array(n_bottom, dtype=np.float32)).nnz:
            raise ValueError("hierarchy summing rows must start with the bottom identity")
        for array in (nodes, level, summing.data, summing.indices, summing.indptr):
            array.flags.writeable = False
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "summing", summing)
        object.__setattr__(self, "level", level)

    @property
    def n_bottom(self) -> int:
        return self.summing.shape[1]

    def aggregate(self, values: np.ndarray) -> np.ndarray:
        """Node values `[N, T]` float32 from bottom values `[B, T]`: sums."""
        return np.asarray(self.summing @ values, dtype=np.float32)

    def any_bottom(self, flags: np.ndarray) -> np.ndarray:
        """Node flags `[N, T]`: True when any bottom series of the node is True."""
        return np.asarray(self.summing @ flags.astype(np.float32)) > 0

    @classmethod
    def flat(cls, series: np.ndarray) -> "Hierarchy":
        """Bottom series only, no aggregates."""
        n = len(series)
        return cls(series, sp.eye_array(n, format="csr"), np.zeros(n))

    @classmethod
    def from_attributes(
        cls,
        series: np.ndarray,
        attributes: pd.DataFrame,
        levels: Sequence[str | Sequence[str]] | None = None,
    ) -> "Hierarchy":
        """One aggregate node per distinct value of each level, plus a total.

        `attributes` is indexed by series id, one column per attribute. A level is one
        column or a list of columns to cross. None makes one level per column. Labels are
        `column=value`, joined by `/` for crossed columns. An aggregate with the same
        bottom series as a bottom, the total, or an earlier aggregate is dropped.
        """
        table = attributes.loc[series]
        if levels is None:
            levels = list(table.columns)
        n = len(series)
        seen = {(index,) for index in range(n)} | {tuple(range(n))}
        groups, labels, level_ids = [], [], []
        for level, names in enumerate(levels, start=1):
            names = [names] if isinstance(names, str) else list(names)
            for key, members in table.groupby(names, sort=True).indices.items():
                members = tuple(np.sort(members).tolist())
                if members in seen:
                    continue
                seen.add(members)
                groups.append(members)
                values = key if isinstance(key, tuple) else (key,)
                pairs = zip(names, values, strict=True)
                labels.append("/".join(f"{name}={value}" for name, value in pairs))
                level_ids.append(level)
        sizes = [len(members) for members in groups]
        rows = np.repeat(np.arange(len(groups)), sizes)
        columns = np.fromiter(chain.from_iterable(groups), dtype=np.int64, count=sum(sizes))
        aggregates = sp.csr_array((np.ones(len(rows)), (rows, columns)), shape=(len(groups), n))
        return cls(
            np.concatenate([np.asarray(series, dtype=str), labels, ["total"]]),
            sp.vstack([sp.eye_array(n), aggregates, sp.csr_array(np.ones((1, n)))]),
            np.concatenate([np.zeros(n), level_ids, [len(levels) + 1]]),
        )
