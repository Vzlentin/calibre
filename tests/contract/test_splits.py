"""A4, F7, R6, C8: calls split and joined again, and inputs loaded from read-only buffers."""

import dataclasses
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from calibre import (
    BottomUp,
    Forecasts,
    Hierarchy,
    Identity,
    Panel,
    Span,
    Step,
    WlsStruct,
    critical_ratio,
    order_up_to,
    reconcile,
    replay,
    rolling_forecasts,
    settle,
)

from . import _cases
from ._engines import engine
from ._expected import assert_bits
from ._panels import grid, make_dense, to_frame, to_panel
from .test_forecasts import FALLBACK, SIMPLE, WITH_FALLBACK, ragged_dense
from .test_reconcile import ATTRIBUTES, LEVELS, SERIES, hierarchy, large

ORIGINS = pd.date_range("2024-01-07", periods=9, freq="W-SUN")
MODELS = np.array(["m1", "m2", "m3"])
FIELDS = ("point", "truth", "score", "threshold", "lower", "upper", "censored")


def shared(obj: Any) -> Any:
    """`obj` pickled with protocol 5 and loaded from read-only buffers, as a process that
    shares memory with others receives it."""
    buffers: list[pickle.PickleBuffer] = []
    data = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
    return pickle.loads(data, buffers=[buffer.raw().toreadonly() for buffer in buffers])


def same(actual: Any, expected: Any) -> None:
    """The same type and values: float arrays bit for bit, through data classes."""
    assert type(actual) is type(expected), (type(actual), type(expected))
    if isinstance(expected, np.ndarray):
        if expected.dtype.kind == "f":
            assert_bits(actual, expected)
        else:
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, pd.DatetimeIndex):
        assert actual.equals(expected)
        assert actual.dtype == expected.dtype
    elif isinstance(expected, sp.csr_array):
        assert actual.shape == expected.shape
        for name in ("data", "indices", "indptr"):
            same(getattr(actual, name), getattr(expected, name))
    elif dataclasses.is_dataclass(expected):
        for field in dataclasses.fields(expected):
            same(getattr(actual, field.name), getattr(expected, field.name))
    else:
        assert actual == expected


def rolling_engines():
    return [engine(SIMPLE, "D"), engine(WITH_FALLBACK, "D", FALLBACK)]


@pytest.mark.parametrize("lookback", [None, 12])
def test_rolling_forecasts_split_by_series_and_origins(lookback):
    """F7: a call on `panel.take(index)` gives the rows `index` of the whole call, and a call
    on some of the origins gives the columns of those origins, bit for bit."""
    panel = to_panel(ragged_dense())
    origins = panel.periods[[8, 9, 15, 22, 23, 30, 41, 52, 60]]
    engines = rolling_engines()
    whole = rolling_forecasts(panel, engines, origins, 6, lookback=lookback)
    for index in ([0, 1, 2, 3], [4, 5, 6, 7], [1, 4, 6], [0, 2, 3, 5, 7], [6]):
        part = rolling_forecasts(
            panel.take(np.array(index)), engines, origins, 6, lookback=lookback
        )
        assert_bits(part.points, whole.points[:, index])
        assert list(part.series) == list(whole.series[index])
        assert list(part.models) == list(whole.models)
        assert part.origins.equals(whole.origins)
    for columns in ([0, 1, 2], [3, 4, 5, 6, 7, 8], [4], [8]):
        part = rolling_forecasts(panel, engines, origins[columns], 6, lookback=lookback)
        assert_bits(part.points, whole.points[..., columns])
        assert part.origins.equals(origins[columns])


@pytest.mark.parametrize(
    "method", [BottomUp(), Identity(), WlsStruct()], ids=lambda m: type(m).__name__
)
def test_reconcile_split_by_models_and_origins(method):
    """R6: a call on some of the models, or on some of the origins, gives that part of the
    whole call, bit for bit."""
    tree = hierarchy()
    series = tree.nodes[: tree.n_bottom] if isinstance(method, BottomUp) else tree.nodes
    rng = np.random.default_rng(91)
    # values that are not multiples of 1/64, over four orders of magnitude
    shape = (len(MODELS), len(series), 4, len(ORIGINS))
    points = rng.lognormal(4.0, 2.0, shape).astype(np.float32)
    if not isinstance(method, WlsStruct):
        points[rng.random(points.shape) < 0.1] = np.nan
    whole = reconcile(Forecasts(points, MODELS, series, ORIGINS), tree, method)
    for columns in ([0, 1, 2], [3, 4, 5, 6, 7, 8], [5], [8]):
        part = reconcile(
            Forecasts(points[..., columns], MODELS, series, ORIGINS[columns]), tree, method
        )
        assert_bits(part.points, whole.points[..., columns])
        assert part.origins.equals(ORIGINS[columns])
    for rows in ([0], [1], [2], [0, 2]):
        part = reconcile(Forecasts(points[rows], MODELS[rows], series, ORIGINS), tree, method)
        assert_bits(part.points, whole.points[rows])
        assert list(part.series) == list(whole.series)


def nodes_of(target: Step | Span, index: np.ndarray) -> Step | Span:
    """The target of the nodes at `index`."""
    if isinstance(target, Step):
        return target
    first = target.first[index] if isinstance(target.first, np.ndarray) else target.first
    last = target.last[index] if isinstance(target.last, np.ndarray) else target.last
    return Span(first=first, last=last)


def levels_of(level: float | np.ndarray, index: np.ndarray) -> float | np.ndarray:
    """The levels of the nodes at `index`."""
    if isinstance(level, np.ndarray) and level.ndim == 1:
        return level[index]
    return level


@pytest.mark.parametrize("name", _cases.NAMES)
def test_replay_split_by_nodes(name):
    """C8: a call on some of the nodes gives the rows of those nodes in each array of the
    whole call, bit for bit."""
    case = _cases.make(name)
    whole = case.run()
    forecasts, truth = case.forecasts(), case.truth()
    for rows in ([0, 2, 5], [1, 3, 4], [4], [0, 1, 2, 3, 5]):
        index = np.array(rows)
        part = replay(
            Forecasts(
                forecasts.points[:, index],
                forecasts.models,
                forecasts.series[index],
                forecasts.origins,
            ),
            truth.take(index),
            target=nodes_of(case.target, index),
            method=case.method,
            level=levels_of(case.level, index),
        )
        assert part.origins.equals(whole.origins)
        for field in FIELDS:
            same(getattr(part, field), getattr(whole, field)[index])


def settlement():
    on_hand = np.array([1.0, 2.0, 3.0], np.float32)
    in_transit = np.array([[1.0, 0.0], [0.0, 4.0], [2.0, 2.0]], np.float32)
    demand = np.array([2.0, 7.0, 1.0], np.float32)
    order = np.array([4.0, 0.0, 1.5], np.float32)
    return on_hand, in_transit, demand, order


@pytest.fixture(scope="module")
def public_objects() -> dict[str, Any]:
    case = _cases.make("irregular-span-signed")
    on_hand, in_transit, demand, order = settlement()
    _, _, settled = settle(on_hand, in_transit, demand, order, 0.2, 1.0)
    return {
        "Panel": to_panel(ragged_dense()),
        "Hierarchy": hierarchy(),
        "Forecasts": case.forecasts(),
        "Replay": case.run(),
        "Step": Step(),
        "Span": case.target,
        "Split": case.method,
        "BottomUp": BottomUp(),
        "Identity": Identity(),
        "WlsStruct": WlsStruct(),
        "Settlement": settled,
    }


NAMES = [
    "Panel",
    "Hierarchy",
    "Forecasts",
    "Replay",
    "Step",
    "Span",
    "Split",
    "BottomUp",
    "Identity",
    "WlsStruct",
    "Settlement",
]


@pytest.mark.parametrize("name", NAMES)
def test_public_objects_pickle(public_objects, name):
    """A4: each public class pickles with protocol 5 and loads to an object of the same type,
    with the same dtypes, shapes and bits in its arrays, and equal other fields."""
    obj = public_objects[name]
    same(shared(obj), obj)
    same(pickle.loads(pickle.dumps(obj, protocol=5)), obj)


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_loaded_panel_is_read_only(public_objects, protocol):
    """A4, P3: a panel loaded by pickle has read-only arrays, with any protocol."""
    loaded = pickle.loads(pickle.dumps(public_objects["Panel"], protocol=protocol))
    for name in ("series", "data", "indptr", "starts"):
        array = getattr(loaded, name)
        assert not array.flags.writeable, name
        with pytest.raises(ValueError):
            array[0] = array[-1]


def cv_frame() -> pd.DataFrame:
    """A cross-validation frame: two series, two cutoffs, two steps, one model."""
    cutoffs = pd.to_datetime(["2024-01-10", "2024-01-12"])
    rows = [(uid, cutoff, h) for uid in ("b", "a") for cutoff in cutoffs for h in (1, 2)]
    rng = np.random.default_rng(93)
    return pd.DataFrame(
        {
            "unique_id": [uid for uid, _, _ in rows],
            "ds": [cutoff + np.timedelta64(h, "D") for _, cutoff, h in rows],
            "cutoff": [cutoff for _, cutoff, _ in rows],
            "y": grid(rng, len(rows)).astype(np.float32),
            "m": grid(rng, len(rows)).astype(np.float32),
        }
    )


def test_functions_run_on_read_only_inputs():
    """A4: each public function gives the same result, bit for bit, on inputs loaded from
    read-only buffers."""
    dense = ragged_dense()
    panel = to_panel(dense)
    stored = shared(panel)
    assert not stored.data.flags.writeable
    engines = rolling_engines()
    origins = panel.periods[[8, 15, 30, 52, 60]]
    same(
        rolling_forecasts(stored, shared(engines), shared(origins), 6),
        rolling_forecasts(panel, engines, origins, 6),
    )
    pooled = [engine(SIMPLE, "D", n_jobs=2), engine(WITH_FALLBACK, "D", FALLBACK, n_jobs=2)]
    same(
        rolling_forecasts(stored, shared(pooled), shared(origins), 6, lookback=12),
        rolling_forecasts(panel, engines, origins, 6, lookback=12),
    )
    index = np.array([1, 4, 6])
    same(stored.take(shared(index)), panel.take(index))
    frame = to_frame(dense)
    same(Panel.from_frame(shared(frame), freq="D"), Panel.from_frame(frame, freq="D"))

    tree = hierarchy()
    same(Hierarchy.from_attributes(shared(SERIES), shared(ATTRIBUTES), LEVELS), tree)
    same(Hierarchy.flat(shared(SERIES)), Hierarchy.flat(SERIES))
    rng = np.random.default_rng(92)
    periods = pd.date_range("2024-01-01", periods=30, freq="D")
    spans = [(0, 30), (4, 30), (0, 22), (10, 30), (0, 30), (2, 26), (0, 30), (7, 30)]
    bottoms = to_panel(make_dense(rng, tuple(SERIES), periods, "D", spans))
    same(shared(tree).aggregate(shared(bottoms)), tree.aggregate(bottoms))
    for method in (BottomUp(), Identity(), WlsStruct()):
        series = tree.nodes[: tree.n_bottom] if isinstance(method, BottomUp) else tree.nodes
        forecasts = Forecasts(large(rng, (2, len(series), 3, 4)), MODELS[:2], series, ORIGINS[:4])
        same(
            reconcile(shared(forecasts), shared(tree), shared(method)),
            reconcile(forecasts, tree, method),
        )
        same(shared(forecasts).select("m2"), forecasts.select("m2"))
    cv = cv_frame()
    ids = np.array(["a", "b"])
    same(
        Forecasts.from_cv(shared(cv), series=shared(ids), horizon=2),
        Forecasts.from_cv(cv, series=ids, horizon=2),
    )

    for name in ("daily-step-absolute", "irregular-span-signed"):
        case = _cases.make(name)
        got = replay(
            shared(case.forecasts()),
            shared(case.truth()),
            target=shared(case.target),
            method=shared(case.method),
            level=shared(case.level),
        )
        same(got, case.run())
        horizon = case.points.shape[2]
        same(
            shared(case.target).cover(case.n_nodes, horizon),
            case.target.cover(case.n_nodes, horizon),
        )

    holding, shortage = np.array([0.2, 0.5, 1.0]), np.array([1.0, 1.0, 3.0])
    same(critical_ratio(shared(holding), shared(shortage)), critical_ratio(holding, shortage))
    on_hand, in_transit, demand, order = settlement()
    bound = np.array([5.0, 0.0, 12.5], np.float32)
    same(
        order_up_to(shared(bound), shared(on_hand), shared(in_transit)),
        order_up_to(bound, on_hand, in_transit),
    )
    got = settle(shared(on_hand), shared(in_transit), shared(demand), shared(order), 0.2, 1.0)
    want = settle(on_hand, in_transit, demand, order, 0.2, 1.0)
    for got_part, want_part in zip(got, want, strict=True):
        same(got_part, want_part)
