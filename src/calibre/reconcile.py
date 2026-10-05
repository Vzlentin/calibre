"""Reconcilers: linear maps from base forecasts `[S, H]` to coherent node points `[N, H]`.

S is B for `BottomUp` and N for `Identity` and `WlsStruct`. Nodes are ordered bottoms
first, so the forecast series are always the first S nodes.
"""

from typing import Protocol

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve

from calibre.hierarchy import Hierarchy


class Reconciler(Protocol):
    @property
    def shape(self) -> tuple[int, int]:
        """`(N, S)`: nodes out, forecast series in."""
        ...

    def __call__(self, base: np.ndarray) -> np.ndarray:
        """Return node points `[N, H]` float32 from base forecasts `[S, H]`."""
        ...


class BottomUp:
    """Forecast the bottom series and sum them."""

    def __init__(self, hierarchy: Hierarchy) -> None:
        self.summing = hierarchy.summing
        self.shape = self.summing.shape

    def __call__(self, base: np.ndarray) -> np.ndarray:
        return np.asarray(self.summing @ base, dtype=np.float32)


class Identity:
    """Forecast every node and keep the base forecasts. They are not coherent."""

    def __init__(self, hierarchy: Hierarchy) -> None:
        n_nodes = len(hierarchy.nodes)
        self.shape = (n_nodes, n_nodes)

    def __call__(self, base: np.ndarray) -> np.ndarray:
        return np.asarray(base, dtype=np.float32)


class WlsStruct:
    """MinT with structural weights, solved through the aggregation constraints.

    Coherent points `y` satisfy `J y = 0` with `J = [A, -I]`, `A` the aggregate rows of
    the summing matrix. With `W = diag(S 1)`, the leaf count per node, the projection of
    base forecasts `b` is `y = b - W J^T (J W J^T)^-1 J b`, the same estimate as
    `S (S^T W^-1 S)^-1 S^T W^-1 b`. It factors only the `[A, A]` aggregate system, once.
    """

    def __init__(self, hierarchy: Hierarchy) -> None:
        n_nodes, n_bottom = hierarchy.summing.shape
        aggregate = hierarchy.summing[n_bottom:].astype(np.float64)
        identity = sp.eye_array(n_nodes - n_bottom, format="csr")
        self.shape = (n_nodes, n_nodes)
        self.constraint = sp.hstack([aggregate, -identity], format="csr")
        self.weights = np.asarray(hierarchy.summing.sum(axis=1), dtype=np.float64)
        system = self.constraint @ sp.diags_array(self.weights) @ self.constraint.T
        self.factor = cho_factor(system.toarray(), overwrite_a=True)

    def __call__(self, base: np.ndarray) -> np.ndarray:
        base = np.asarray(base, dtype=np.float64)
        multiplier = cho_solve(self.factor, self.constraint @ base)
        return (base - self.weights[:, None] * (self.constraint.T @ multiplier)).astype(np.float32)
