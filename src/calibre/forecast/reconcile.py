"""Reconcilers: maps from forecast series to the nodes of a hierarchy."""

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict
from scipy.sparse.linalg import splu

from calibre.data.hierarchy import Hierarchy
from calibre.forecast.forecasts import Forecasts


class Reconciler(BaseModel, ABC):
    """Maps points `[M, S, H, O]` float32 to node points `[M, N, H, O]` float32."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    @abstractmethod
    def project(self, points: np.ndarray, hierarchy: Hierarchy) -> np.ndarray: ...


def _columns(points: np.ndarray) -> np.ndarray:
    """`[S, M * H * O]`: one row per series."""
    return np.moveaxis(points, 1, 0).reshape(points.shape[1], -1)


def _nodes(columns: np.ndarray, shape: tuple[int, int, int, int]) -> np.ndarray:
    """`[M, N, H, O]` from `[N, M * H * O]`."""
    n_models, _, horizon, n_origins = shape
    return np.moveaxis(columns.reshape(-1, n_models, horizon, n_origins), 0, 1)


def _bottom_up(points: np.ndarray, hierarchy: Hierarchy) -> np.ndarray:
    bottoms = _columns(points)
    missing = np.isnan(bottoms)
    summing = hierarchy.summing.astype(np.float64)
    nodes = summing @ np.where(missing, 0.0, bottoms.astype(np.float64))
    if missing.any():
        present = hierarchy.summing @ (~missing).astype(np.float32)
        nodes[present == 0] = np.nan
    return np.ascontiguousarray(_nodes(nodes.astype(np.float32), points.shape))


def _shape(points: np.ndarray, n_series: int) -> None:
    if points.ndim != 4 or points.shape[1] != n_series:
        raise ValueError(f"points must be [M, {n_series}, H, O], got shape {points.shape}")


class BottomUp(Reconciler):
    """Each node is the sum of its bottoms. A NaN bottom adds zero."""

    def project(self, points: np.ndarray, hierarchy: Hierarchy) -> np.ndarray:
        _shape(points, hierarchy.n_bottom)
        return _bottom_up(points, hierarchy)


class Identity(Reconciler):
    """The forecasts of every node, as they are. They are not coherent."""

    def project(self, points: np.ndarray, hierarchy: Hierarchy) -> np.ndarray:
        _shape(points, len(hierarchy.nodes))
        return np.array(points, dtype=np.float32)


class WlsStruct(Reconciler):
    """MinT with structural scaling: `S (S' W^-1 S)^-1 S' W^-1 y`, W the bottom counts."""

    def project(self, points: np.ndarray, hierarchy: Hierarchy) -> np.ndarray:
        _shape(points, len(hierarchy.nodes))
        if np.isnan(points).any():
            raise ValueError("WlsStruct needs points with no NaN")
        summing = hierarchy.summing.astype(np.float64)
        inverse = sp.diags_array(1.0 / np.asarray(summing.sum(axis=1)).ravel())
        weighted = summing.T @ inverse
        normal = (weighted @ summing).tocsc()
        rhs = weighted @ _columns(points).astype(np.float64)
        bottoms = splu(normal).solve(np.asarray(rhs))
        bottoms = bottoms.astype(np.float32)
        n_models, _, horizon, n_origins = points.shape
        shape = (n_models, hierarchy.n_bottom, horizon, n_origins)
        return _bottom_up(np.ascontiguousarray(_nodes(bottoms, shape)), hierarchy)


def reconcile(forecasts: Forecasts, hierarchy: Hierarchy, method: Reconciler) -> Forecasts:
    """Node points from forecasts by `method`. The result has the nodes as its series."""
    if not isinstance(forecasts, Forecasts):
        raise TypeError("reconcile takes a Forecasts")
    if not isinstance(hierarchy, Hierarchy):
        raise TypeError("hierarchy must be a Hierarchy")
    if not isinstance(method, Reconciler):
        raise TypeError("method must be a Reconciler")
    n_series = len(forecasts.series)
    if not np.array_equal(forecasts.series, hierarchy.nodes[:n_series]):
        raise ValueError("the forecast series must be the first nodes of the hierarchy, in order")
    points = method.project(forecasts.points, hierarchy)
    return Forecasts(points, forecasts.models, hierarchy.nodes, forecasts.origins)
