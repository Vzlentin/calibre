# Architecture

This document and [Semantics](semantics.md) are the contract of Calibre v2. Each guarantee
has an id, such as C3, and names the check that proves it. The checks are frozen with these
documents (see [Checks](#checks)). A milestone PR does not change them. A change of the
contract is its own PR, and the owners review it.

## Scope

Calibre is a library of array functions for conformal forecasting, on the Nixtla kernel
layer. It reads a panel, makes point forecasts at each origin, reconciles them over a
hierarchy, and issues calibrated bands and bounds at each origin. A bound gives an order and
its lost-sales cost.

Calibre owns what no Nixtla library has: when each value becomes known, its reconciler, and
the decision functions. The rest comes from Nixtla.

| Part | From | Used as |
|---|---|---|
| Frames to arrays | utilsforecast | inside `Panel.from_frame` |
| Kernels | coreforecast | `GroupedArray` holds each window. `replay` may use its rolling statistics |
| Models that Nixtla has | their engines, passed as they are | statsforecast from M1. mlforecast and neuralforecast from M3 |
| Models that Nixtla does not have | Calibre's `Forecaster`, on coreforecast | from M3 |
| Scores and the exact rank | conformalforecast | `AbsoluteResidual`, `SignedResidual`, `conformal_quantile` |
| Cross-validation frames | conformalforecast | `reshape_cv_many`, inside `Forecasts.from_cv` |
| Coverage, width, Winkler score | conformalforecast diagnostics | called by the user on `Replay` arrays moved to `[N, O, C]`. Calibre has no metrics function |
| Reconciliation | Calibre | checked against hierarchicalforecast |

Calibre uses a Nixtla object when its unit of work is Calibre's unit, all origins at once.
Otherwise it uses the primitive under that object. A statsforecast model fits one series at
a time anyway, so statsforecast runs as an engine. Calibration methods and their fixes go
upstream to conformalforecast. Calibre keeps the clock.

## Terms

N is the number of nodes, S of series, B of bottom series, M of models, T of periods, O of
origins, H of steps, and C of target columns.

| Term | Meaning |
|---|---|
| Period | One step of the panel calendar, at the panel frequency |
| Series | Finite values on consecutive periods, from its first period to its last |
| Node | A bottom series, or a sum of bottom series. Bottoms come first |
| Origin | A period at which a forecast is issued: the last period that the forecast reads. Origin o forecasts the H periods after o |
| Step | h in 1 to H: the period h periods after the origin |
| Target | The steps that each column sums: `Step` or `Span` |
| Column | One sum of steps. `Step` has H columns, one per step. `Span` has one |
| Cell | One node, column and origin |
| Point | The sum of the step points of a column |
| Truth | The sum of the observed values of a column |
| Score | conformalforecast's score of a point and its truth: the absolute or the signed residual |
| Known at o | A score of origin o0 is known at origin o when the last period of its column is at or before o |
| Score window | The scores that the threshold of a cell reads: `Split.window` origins long |
| Level | A probability strictly between 0 and 1, one per node: the coverage of a band, or the probability that the truth is at or below a bound. `Hierarchy.level` is another thing: the id of an aggregation level |
| Threshold | The score quantile issued for a cell |
| Band | `[point - threshold, point + threshold]`, from absolute scores |
| Bound | `point + threshold`, from signed scores: an upper bound |
| Ready | A cell that holds enough known scores for its level |
| Censored | A value that is a lower limit of demand, such as sales during a stockout |

## Milestones

Each milestone is one PR into the `v2` branch. A campaign builds it, and the owners review
it. Before each milestone after M1, a contract PR extends these documents and the checks, as
M0 does for M1. After 0.1 freezes, `v2` merges into `main`, and 0.1 stays as its release.

| Milestone | Delivers |
|---|---|
| M0 | These documents and `AGENTS.md`. The checks of [Checks](#checks). The baselines of G2, measured with 0.1 first. The M5 and VN2 loaders, rewritten as raw readers that import no `calibre`: they give arrays and frames, and the VN2 loader holds the VN2 protocol, with the decision files loaded by path. The dependencies of v2 in `pyproject.toml` and `uv.lock`. CI on the `v2` branch, with the frozen-path check. Its CI fails by design, because the v2 code does not exist yet, and the owners approve it by review. Before M0 merges, its author runs every frozen file once against a throwaway prototype of the M1 functions, to find defects in the checks. The prototype is not committed. A defect that M1 finds in a frozen file goes to a contract PR |
| M1 | The backtest on the new layout: every module in [Modules](#modules). Every check passes |
| M2 | Production: `step` and its state, and a ledger that records the order and the stock in force at each origin |
| M3 | The `Forecaster` contract and `Window` with covariates, `AR1` and `BootstrapEnsemble` on coreforecast, the mlforecast and neuralforecast engines |
| M4 | `combine` with inverse-MAE weights over models and origins, and `fitted_through`, which reuses a fitted model across runs |
| Later | ACI, the quantile tracker and PID, upstream or on Calibre's base. Pooled scores. Decisions that read more than one quantile. A correction for censored scores. `RiskControl` |

M1 deletes each 0.1 module that it does not rewrite, with the bench scripts that import it.
It deletes or rewrites each 0.1 test, so that `pytest` passes. A deleted part comes back
with its milestone. 0.1 keeps it as a release.

| 0.1 | v2 |
|---|---|
| `forecast/models/` | the statsforecast engine in M1, the rest in M3 |
| `conformal/scores.py` | conformalforecast's scores |
| `conformal/calibrators/` | `conformal/split.py` in M1, the online methods later |
| `conformal/losses.py` | with `RiskControl`, later |
| `online/` | M2 |
| `backtest/scorecasting.py` | with PID, later |
| `metrics.py` | conformalforecast diagnostics |
| `bench/compare.py`, `bench/vn2_conformal.py`, `bench/vn2_intervals.py`, `bench/selective_imagenet.py` | `bench/gate_vn2.py` takes the place of `bench/vn2_conformal.py` |

`decision/` stays as it is. The other bench loaders stay, and M1 changes each one to return
the v2 `Panel`, with the same values. A loader that gave a censored mask passes it as
`Panel.censored`. Other masks stay next to the panel in the dataclass of the loader, with
their 0.1 shape `[B, T]`.

## Modules

| Module | Responsibility |
|---|---|
| `data/panel.py` | `Panel`: the ragged grouped array on one calendar, validated once |
| `data/hierarchy.py` | `Hierarchy`: node labels, the summing matrix, `aggregate` |
| `forecast/window.py` | `Window`: what one origin lets a model read |
| `forecast/engines/statsforecast.py` | Runs the models of a `StatsForecast` object on windows |
| `forecast/forecasts.py` | `Forecasts`, `Forecasts.from_cv` |
| `forecast/reconcile.py` | `reconcile`, the `Reconciler` contract, `BottomUp`, `Identity`, `WlsStruct` |
| `backtest/rolling.py` | `rolling_forecasts` |
| `backtest/replay.py` | `replay`, `Replay`: the thresholds and their bands and bounds |
| `conformal/targets.py` | the `Target` contract, `Step`, `Span` |
| `conformal/split.py` | `Split`: the settings of split calibration |
| `decision/policy.py` | `critical_ratio`, `order_up_to` |
| `decision/lost_sales.py` | `settle`, `Settlement` |

| Package | Calibre packages it imports | Other packages it imports |
|---|---|---|
| `data` | none | numpy, pandas, scipy, utilsforecast |
| `forecast` | `data` | numpy, pandas, scipy, pydantic, coreforecast, conformalforecast, narwhals. statsforecast only in the engine file |
| `conformal` | none | numpy, pydantic, conformalforecast |
| `backtest` | `data`, `forecast`, `conformal` | numpy, pandas, coreforecast, conformalforecast |
| `decision` | none | numpy |

| Id | Guarantee | Check |
|---|---|---|
| A1 | `import calibre` loads neither statsforecast nor hierarchicalforecast. The engine file loads only when a `StatsForecast` object is passed | `test_imports.py` |
| A2 | The import statements of each module of `src/calibre` follow the table above, plus the standard library. `calibre/__init__.py` imports only Calibre packages. Imports in the `if` branch of `if TYPE_CHECKING:` do not count. No module uses `importlib`, `runpy`, `__import__`, `exec`, `eval`, `sys.modules` or `os._exit`, except `backtest/rolling.py`, which loads the engine file with `importlib.import_module("calibre.forecast.engines.statsforecast")`, the module name as a string literal, and uses `importlib` for nothing else. No line of `src/calibre` holds `# noqa`, `# ruff: noqa`, `# type: ignore`, `# ty: ignore` or `# pyright: ignore` | `test_imports.py`, by a scan of the source |
| A3 | `calibre.__all__` is exactly `Panel`, `Hierarchy`, `rolling_forecasts`, `Forecasts`, `reconcile`, `Reconciler`, `BottomUp`, `Identity`, `WlsStruct`, `Target`, `Step`, `Span`, `Split`, `replay`, `Replay`, `critical_ratio`, `order_up_to`, `settle` and `Settlement` | `test_imports.py` |

`Window` and the engine file stay internal until M3.

## Public functions

Each function checks the conditions that this section states for its inputs, once. It
raises `TypeError` when an input has the wrong type, and `ValueError` when another condition
fails. The type is that of the argument itself: a frame with a missing column or a column of
the wrong dtype, and a level that names no column, raise `ValueError`. A subclass of either
error is allowed, such as a named error of Calibre. Each data class checks, when it is built, the shapes, dtypes and conditions that its
comments and guarantees state, and converts arrays to the stated dtypes. Pydantic settings
use strict mode and forbid extra fields: they raise pydantic's `ValidationError`, which is a
`ValueError`.

### Data

#### Panel

```python
@dataclass(frozen=True)
class Panel:
    series: np.ndarray  # [S] str, unique
    periods: pd.DatetimeIndex  # [T] the calendar, complete at freq
    freq: str
    data: np.ndarray  # [n] float32
    indptr: np.ndarray  # [S + 1] int64
    starts: np.ndarray  # [S] int64: index of the first period of each series
    censored: np.ndarray | None = None  # [n] bool

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        freq: str,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        censored_col: str | None = None,
    ) -> "Panel": ...
```

The constructor keeps the order of the series. It copies its inputs and converts them to
the stated dtypes.

| Id | Guarantee | Check |
|---|---|---|
| P1 | The series ids are unique. The calendar is complete at `freq`, and runs from the earliest first period of the series to the latest last period. `indptr` has S + 1 values, starts at 0 and increases, and its last value is the length of `data`. `starts` has S values. Series s holds `data[indptr[s]:indptr[s + 1]]`: one finite value for each period from `periods[starts[s]]` on, with no gap. Each series has at least one value, and its last period is on the calendar. `censored`, when present, has the length of `data` | `test_panel.py` |
| P2 | `from_frame` sorts the series by id, and keeps the values of each series in time order. It rejects a duplicate id and time, a time off the calendar, a gap inside a series, a value that is not finite, and a censored column that is not boolean | `test_panel.py` |
| P3 | The constructor raises when P1 fails. A change of its input arrays after construction does not change the panel, and the arrays of the panel are read-only | `test_panel.py` |

#### Hierarchy

```python
@dataclass(frozen=True)
class Hierarchy:
    nodes: np.ndarray  # [N] str
    summing: sp.csr_array  # [N, B] float32
    level: np.ndarray  # [N] int32

    @classmethod
    def from_attributes(
        cls,
        series: np.ndarray,
        attributes: pd.DataFrame,
        levels: Sequence[str | Sequence[str]] | None = None,
    ) -> "Hierarchy": ...

    @classmethod
    def flat(cls, series: np.ndarray) -> "Hierarchy": ...

    @property
    def n_bottom(self) -> int: ...

    def aggregate(self, panel: Panel) -> Panel: ...
```

`series` holds at least two series. `attributes` is indexed by series id, with one column per
attribute. It holds every series, with no missing value. Its other rows are ignored. A level is one column, or a list of
columns to cross. None makes one level per column.

| Id | Guarantee | Check |
|---|---|---|
| H1 | `from_attributes` puts the bottoms first, labeled with their series ids, in the order of `series`. Then it adds one aggregate per distinct value of each level, level by level, with the values of a level in the order of pandas `groupby(sort=True)`, and the total last, labeled `total`. An aggregate is labeled `column=value`, with `str` of the value, and a crossed level joins these labels with `/`, as in `state_id=CA/cat_id=FOODS`. It is dropped when it has the same bottoms as a bottom, the total, or an earlier aggregate. `level` is 0 for bottoms, the position of the level from 1 for aggregates, and the number of levels plus 1 for the total. The bottom rows of `summing` are the identity. `flat` gives the bottoms only, with no aggregate | `test_hierarchy.py` |
| H2 | `aggregate` takes a panel whose series are the bottoms, in order. It gives a panel whose series are the nodes, in order, on the same calendar. A node runs from the earliest first period of its members to their latest last period. Its value at a period is the sum of the member values at that period, in float64, stored as float32. A member adds zero before its first period and after its last. When the panel has a censored array, a node value is censored when one member value at that period is censored. When it has none, the result has none | `test_hierarchy.py`, `test_censoring.py` |

### Forecast

#### Forecasts

```python
@dataclass(frozen=True)
class Forecasts:
    points: np.ndarray  # [M, S, H, O] float32
    models: np.ndarray  # [M] str
    series: np.ndarray  # [S] str: series, or node labels after reconcile
    origins: pd.DatetimeIndex  # [O] strictly increasing

    @classmethod
    def from_cv(
        cls,
        cv: pd.DataFrame,
        *,
        series: np.ndarray,
        horizon: int,
        models: Sequence[str] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> "Forecasts": ...

    def select(self, model: str) -> "Forecasts": ...
```

`from_cv` reads a Nixtla cross-validation frame: one row per series, cutoff and step, with a
datetime `cutoff` column and one column per model. Each series and cutoff in the frame has
exactly `horizon` rows. Each id of the frame is in `series`. `models=None` reads every
column other than `id_col`, `time_col`, `cutoff` and `target_col`, and other than the interval
columns, whose names hold `-lo-` or `-hi-`. A frame carries no time rule, so Calibre cannot
check that its points did not read the future. `select` returns one model, with M = 1, and
raises for an unknown model.

| Id | Guarantee | Check |
|---|---|---|
| F6 | `from_cv` puts the value of model m, series s, cutoff o and step h at `points[m, s, h - 1, o]`, as float32. The origins are the sorted cutoffs. The rows follow `series`. A series that is absent at a cutoff has NaN points there | `test_forecasts.py` |

#### rolling_forecasts

```python
def rolling_forecasts(
    panel: Panel,
    engines: Sequence[StatsForecast],
    origins: pd.DatetimeIndex,
    horizon: int,
    *,
    lookback: int | None = None,
) -> Forecasts: ...
```

`origins` are periods of the panel, strictly increasing, at any spacing. `horizon` is at
least 1. `lookback` is at least 1, or None for the whole history. The `freq` of each engine
is the `freq` of the panel, compared as pandas offsets. The window of origin o holds each
series whose first period is at or before o and whose last period is at or after o, with its
values through o, at most the last `lookback`.

`engines` holds at least one `StatsForecast` object: an object whose type is `StatsForecast`
from the `statsforecast` package. Another type raises `TypeError`. `rolling_forecasts` loads
the engine file when it meets the first such object.

The engine reads the models, the `fallback_model`, the `freq` and the `n_jobs` of each
`StatsForecast` object. For each series window, it calls `model.forecast(y=values, h=horizon)`
of each model, as statsforecast's cross-validation does with `refit=True`. When a model
raises, the engine calls the fallback model in the same way, or raises the exception of the
model again when there is none. With `n_jobs` other than 1, the engine keeps its own pool of
worker processes for the whole call, and ends it before the call returns. Each worker gets a chunk of series once and runs every origin for it: this is
the origin loop of `rolling_forecasts`. G1 binds its cost.

The result has one row per model: the engines in order, and the models of each engine in
order. A row is named by the alias of its model.

| Id | Guarantee | Check |
|---|---|---|
| F1 | Take a panel whose series end on the last period. Let `df` be the long frame that built the panel, with `y` as float32. Take `cv = sf.cross_validation(df=df, h=horizon, step_size=step_size, n_windows=n_windows, input_size=input_size)`, with `refit=True` and no `level`, `test_size` or prediction intervals, for any settings that statsforecast accepts. With the cutoffs of `cv` as `origins`, `rolling_forecasts(panel, [sf], origins, horizon, lookback=input_size)` equals `Forecasts.from_cv(cv, series=panel.series, horizon=horizon)`: the points bit for bit, NaN included, the models, the series, and the values of the origins | `test_forecasts.py` |
| F2 | The points are the same, bit for bit, for any `n_jobs`. No worker process outlives the call | `test_forecasts.py` |
| F3 | No point issued at origin o changes when a value after o changes, or when the data after o is removed | `test_visibility.py` |
| F4 | A series whose first period is after o, or whose last period is before o, has NaN points at o. A model that gives NaN gives NaN points | `test_forecasts.py` |
| F5 | Two models with the same alias in different engines raise. An engine whose `freq` is not the panel `freq` raises. A model that raises in an engine with no fallback model stops the call with its exception | `test_forecasts.py` |

#### reconcile

```python
def reconcile(forecasts: Forecasts, hierarchy: Hierarchy, method: Reconciler) -> Forecasts: ...


class Reconciler(BaseModel, ABC):
    # points [M, S, H, O] float32 to node points [M, N, H, O] float32
    @abstractmethod
    def project(self, points: np.ndarray, hierarchy: Hierarchy) -> np.ndarray: ...


class BottomUp(Reconciler): ...


class Identity(Reconciler): ...


class WlsStruct(Reconciler): ...
```

`BottomUp` takes forecasts whose series are the bottoms of the hierarchy, in order.
`Identity` and `WlsStruct` take forecasts whose series are all the nodes, in order: they come
from `rolling_forecasts` on `hierarchy.aggregate(panel)`. The result has the nodes as its
series. The reconcilers have no fields. In R3 and R4, the scale is the largest absolute value
of the whole input array of points.

| Id | Guarantee | Check |
|---|---|---|
| R1 | `BottomUp` gives the N nodes from the bottoms. A node is the sum of its bottoms, in float64, stored as float32. A NaN bottom adds zero. A node whose bottoms are all NaN is NaN | `test_reconcile.py` |
| R2 | `Identity` returns its input, bit for bit | `test_reconcile.py` |
| R3 | `WlsStruct` gives the projection of MinT with structural scaling, `S (S' W^-1 S)^-1 S' W^-1 y`, where W is diagonal with the number of bottoms under each node. Its bottoms are within 1e-5 times the scale of that projection, computed in float64. Its aggregates are `BottomUp` of its bottoms. It raises on a NaN point | `test_reconcile.py` |
| R4 | On points with no NaN, `BottomUp` equals hierarchicalforecast's `BottomUpSparse`, and `WlsStruct` equals `MinTrace(method="wls_struct")`, within 1e-5 times the scale. hierarchicalforecast takes the bottoms last, so the check moves them | `test_reconcile.py` |
| R5 | `reconcile` raises `TypeError` when its input is not a `Forecasts` | `test_totals.py` |

### Conformal

A conformal method in Calibre has three parts. The score and the statistic come from
conformalforecast: `AbsoluteResidual` or `SignedResidual`, and the exact rank of
`conformal_quantile`. The delay comes from Calibre's clock: a score enters calibration only
once its truth is known. A `Target` says which steps each column sums. `Split` holds the
settings of the method. `replay` runs the clock for all origins in one call: for each node,
column and origin, it issues the quantile of the score window, the latest scores known at
that origin, and the band or the bound around the point. [Semantics](semantics.md) gives
the rules. M1 has one method, `Split`. The other conformalforecast methods and the online
methods come later, as [Milestones](#milestones) says.

#### Step and Span

```python
class Target(ABC):
    @abstractmethod
    def cover(self, n_nodes: int, horizon: int) -> np.ndarray: ...  # [N, C, H] bool


@dataclass(frozen=True)
class Step(Target): ...


@dataclass(frozen=True)
class Span(Target):
    first: int | np.ndarray  # the first step, 1-based, or one per node: [N] int64
    last: int | np.ndarray  # the last step, 1-based, or one per node: [N] int64
```

Targets are frozen dataclasses, not Pydantic settings, because a `Span` per node holds
arrays. `cover` checks that 1 ≤ first ≤ last ≤ `horizon`, and that an array has one value
per node.

| Id | Guarantee | Check |
|---|---|---|
| T1 | `Step` has H columns: column c covers step c + 1. `Span(first, last)` has one column, which covers the steps from `first` to `last` of each node | `test_targets.py` |
| T2 | The point and the truth of a column are the sums over its steps, in float64, stored as float32. Either is NaN when one of its step values is NaN. The score is conformalforecast's score of the stored point and truth, in float32 | `test_targets.py` |

#### Split

```python
class Split(BaseModel):
    score: AbsoluteResidual | SignedResidual  # from conformalforecast.nonconformity
    window: int  # at least 1
```

`AbsoluteResidual` gives bands. `SignedResidual` gives bounds. `Split` holds settings only:
`replay` computes the thresholds. The level is not a setting of `Split`: it is data, passed
to `replay`.

#### replay

```python
def replay(
    forecasts: Forecasts,
    truth: Panel,
    *,
    target: Target,
    method: Split,
    level: float | np.ndarray,
) -> Replay: ...


@dataclass(frozen=True)
class Replay:
    origins: pd.DatetimeIndex  # [O]: the origins of the forecasts
    point: np.ndarray  # [N, C, O] float32
    truth: np.ndarray  # [N, C, O] float32
    score: np.ndarray  # [N, C, O] float32
    threshold: np.ndarray  # [N, C, O] float32
    lower: np.ndarray  # [N, C, O] float32
    upper: np.ndarray  # [N, C, O] float32
    censored: np.ndarray  # [N, C, O] bool
```

`forecasts` holds one model. Its series are the series of `truth`, in order, and each of its
origins is a period of `truth`. `truth` is usually `hierarchy.aggregate(panel)`. `level` is a
float, a 0-d array, or one value per node `[N]`, read as float64, each strictly between 0
and 1. `alpha` is `1 - level`, computed in float64. `Replay.truth` and `Replay.score` are NaN
where the truth is missing or is not known at the last period of `truth`. `threshold` is +inf
where the cell is not ready.

A cell is ready when its score window holds at least `ceil(1 / alpha)` finite scores, with
conformalforecast's ceiling, which rounds to 9 decimals first. That is 10 scores at level
0.9, and 6 at level 1/1.2.

| Id | Guarantee | Check |
|---|---|---|
| C1 | No `point`, `threshold`, `lower` or `upper` issued at origin o changes when a value of `truth` after o changes, or when the data after o is removed | `test_visibility.py` |
| C2 | The threshold of a cell reads the scores of its node and column at the `window` latest origins of `forecasts.origins` whose scores are known at its origin. A missing score keeps its place. Known is read from the dates, so one rule covers daily, weekly and irregular origins | `test_split.py` |
| C3 | The threshold of a ready cell equals conformalforecast's `conformal_quantile` of the finite scores of its score window, at `alpha`, bit for bit | `test_split.py` |
| C4 | The threshold of a cell that is not ready is +inf | `test_split.py` |
| C5 | A band is `lower = point - threshold`, `upper = point + threshold`. A bound is `upper = point + threshold`, `lower = -inf`. Where the point is NaN, `lower` and `upper` are NaN | `test_split.py` |
| C6 | Each threshold comes from the scores of its own node and column. A change of one point changes only thresholds of its node and column, at later origins. The bound of an aggregate is not the sum of the bounds of its members | `test_totals.py` |
| C7 | A cell is censored when one value of its truth is censored, and is not censored when its truth is NaN. Censored scores enter score windows as observed: the thresholds equal those of the same truth with no censored array | `test_censoring.py` |

### Decision

`critical_ratio(holding, shortage)`, `order_up_to(bound, on_hand, in_transit)` and
`settle(on_hand, in_transit, demand, order, holding, shortage)` stay as they are. An order at
the critical ratio reads the `upper` of `replay` with `Split(score=SignedResidual(), ...)`
on a `Span` that covers the lead time and the review period, at
`level=critical_ratio(holding, shortage)` for each node. M1 adds no other decision function.

## Dependencies

| Package | Pin | Reason |
|---|---|---|
| numpy | `>=2.0` | |
| pandas | any version that resolves with the others | |
| scipy | `>=1.13` | the sparse summing matrix and the projection of `WlsStruct` |
| pydantic | `>=2.7` | settings, as in conformalforecast |
| narwhals | `>=2.9` | the frame type of `reshape_cv_many` |
| coreforecast | `==0.0.20` | a 0.x release |
| utilsforecast | `==0.2.17` | a 0.x release |
| conformalforecast | `==0.1.0` | `conformal_quantile` and `reshape_cv_many` are not public |

The `stats` extra is `statsforecast>=2.1,<2.2`. The development group holds statsforecast,
`hierarchicalforecast>=1.5.3,<1.6`, psutil, pytest, ruff and ty. No package of v2 depends on
numba. A pin changes only in a contract PR, with the checks run again.

## Checks

| Check | Proves |
|---|---|
| `tests/contract/test_imports.py` | A1, A2, A3 |
| `tests/contract/test_panel.py` | P1, P2, P3 |
| `tests/contract/test_hierarchy.py` | H1, H2 |
| `tests/contract/test_forecasts.py` | F1, F2, F4, F5, F6 |
| `tests/contract/test_reconcile.py` | R1, R2, R3, R4 |
| `tests/contract/test_targets.py` | T1, T2 |
| `tests/contract/test_split.py` | C2, C3, C4, C5 |
| `tests/contract/test_visibility.py` | F3, C1 |
| `tests/contract/test_totals.py` | R5, C6 |
| `tests/contract/test_censoring.py` | H2 for censored values, C7 |
| `bench/gate_m5.py`, with its run `bench/gate_m5_run.py` | G1 |
| `bench/gate_vn2.py`, with its run `bench/gate_vn2_run.py` | G2 |

The checks catch mistakes and drift. They cannot stop deliberate sabotage: the owners' review
of each milestone diff is the last barrier.

The tests run in CI on seeded synthetic panels, with helpers only from `tests/contract/`.
The panels hold late launches, early ends, censored values, tied scores, daily, weekly and
irregular origins, and levels and spans that differ per node. Their values are multiples of
1/64 below 2^17, so float32 holds each of them exactly and every float64 sum is exact in any
order. Some sums exceed 2^18, where a float32 sum would round. Expected values come from statsforecast, conformalforecast and
hierarchicalforecast, computed in a process that does not import `calibre`, or from a direct
computation in the test. Each test file also checks that its function raises on each input
condition of this document. CI runs `tests/contract` in its own pytest process, apart from the
other tests, with the root `pyproject.toml` as its only configuration. A skip or an expected
failure there fails CI, and so does a test id of `tests/contract/ids.txt` that did not pass.

The visibility probes of F3 and C1 run at three origins: the second, the middle one and the
one before the last. C1 also probes the first origin that has a ready cell. One probe replaces every value after the origin with seeded noise for
points, and with 1e6 for `truth`. A second probe removes the data after the origin. In both,
every output named in F3 and C1 that is issued at or before the origin stays the same, bit
for bit. The probe panels start every series before the first probe origin.

`bench/gate.py` holds the harness of both gates. Each gate runs Calibre in a child process, in a new session, with a new temporary directory
as its `HOME`, `TMPDIR` and `XDG_CACHE_HOME`. A parent process that does not import `calibre` measures it. Another process
that does not import `calibre` checks its outputs and computes the cost: it reads the raw
data with the frozen readers of the M5 and VN2 loaders, and loads the frozen decision files by
path. Besides `calibre`, the gates import only frozen files. After the timed part,
the child writes its outputs to a directory that the parent gives it. A gate fails when a
process of a run is still alive 5 s after the run, or when a run writes in the repository or
leaves a file in its temporary directory.

**G1, speed and memory.** `uv run python bench/gate_m5.py data/m5`: one warm-up run, then
three timed runs. The owners run it for each milestone PR on the gate machine, an Apple M5
Pro with 18 cores and 48 GB, with no other load. A campaign runs it too.

- Data: M5 as 0.1 reads it, each bottom series over all 1,941 days, with the 12 standard
  levels. Before it times anything, the gate checks the shape of the raw data and the sum of
  its values. Its checker then checks the number of nodes, 42,840, and the sum of the values
  of the `total` node.
- Run: `StatsForecast(models=[SeasonalNaive(season_length=7)], freq="D", n_jobs=12)`, 64
  daily origins from 2016-02-21 to 2016-04-24, horizon 28, and `BottomUp`. Then `replay`
  with `Step()` and `Split(score=AbsoluteResidual(), window=28)` at level 0.9, and `replay`
  with `Span(1, 7)` and `Split(score=SignedResidual(), window=28)` at level 0.9.
- Wall time runs from the start of the load to the last array. Peak memory is the largest sum
  of the resident memory of the processes in the session of the run or under its child,
  sampled every 50 ms. Resident memory counts shared pages in each process, so the sum can only
  overstate.
- After the timed runs, the checker recomputes from the raw data every point, as the value of
  the same weekday in the last week of the window, every `BottomUp` sum, and the thresholds of
  10,000 cells of each replay, drawn with a seed that it picks at run time. A ready cell must
  equal `conformal_quantile` of its recomputed scores, bit for bit, and a cell that is not ready
  must be +inf. The expected shape, sums and node count are constants of the gate, which M0
  computes from the raw files.
- Pass: the median wall time is under 19.0 s, and each peak is under 11.9 GB. These are the
  limits of decision 17. 0.1 took 19.4 s and 11.9 GB on this run, in one process, with ACI in
  its calibration.

**G2, the cost of the decision bound.** `uv run python bench/gate_vn2.py data/vn2`.

- Data: VN2 as 0.1 reads it, with its hierarchy and its out-of-stock mask. The gate checks
  its shape and the sum of its values first.
- Run: `StatsForecast(models=[WindowAverage(window_size=13)], freq="W-MON")`, origins at the
  week indexes from 60 to the last of the six decisions, 0-based as in 0.1, horizon 3, and
  `BottomUp`. Then `replay` with `Span(1, 3)` and `Split(score=SignedResidual(), window=52)`
  at `level=critical_ratio(0.2, 1.0)`, which is 1/1.2. The upper bounds of the bottom series
  at the six decisions give the orders and the total cost, by the VN2 protocol of
  `bench/vn2.py`. Each of these bounds is ready, or G2 fails.
- Baseline: `bench/baselines/vn2-0.1.json` and `bench/baselines/vn2-0.1.npz`. M0 measures
  them once, with 0.1 at the commit where `v2` branches and before it rewrites the loaders:
  `SplitQuantile(level, window=52)`, not pooled, on the same forecasts. They hold the total
  cost, the coverage, the bottom points at every origin of the run, and the bounds of the
  bottom series at the six decisions, with their series ids and dates. M0 checks that the
  baseline itself passes the coverage condition below.
- Pass: the bottom points equal the baseline points, bit for bit, matched by series id and
  date. The decision bounds are within 1e-5 times the larger of 1 and the absolute baseline
  bound. The total cost is at most the baseline cost times 1.0001. The coverage is between
  τ - 0.01 and τ + 1/53 + 0.01, with τ = 1/1.2 and 53 = window + 1. The coverage is the
  share, over all nodes and origins, of the cells with a finite bound and a known truth,
  where the truth is at or below the bound.

**Frozen paths.** `docs/architecture.md`, `docs/semantics.md`, `AGENTS.md`,
`tests/__init__.py`, `tests/contract/`, `bench/m5.py`, `bench/vn2.py`, `bench/gate.py`,
`bench/gate_m5.py`, `bench/gate_m5_run.py`, `bench/gate_vn2.py`, `bench/gate_vn2_run.py`,
`bench/baselines/`, `src/calibre/decision/`, `pyproject.toml`,
`uv.lock`, `.python-version` and `.github/`. A milestone PR changes none of them. It adds no
file or directory at the root of the repository, in `bench/`, or in `src/` outside
`src/calibre/`. It adds no `conftest.py`, `sitecustomize.py`, `usercustomize.py`, `.pth` or
`pyproject.toml` file at any depth, and none of `pytest.ini`, `.pytest.ini`, `pytest.toml`,
`.pytest.toml`, `tox.ini`, `setup.cfg`, `ruff.toml`, `.ruff.toml` and `ty.toml`. CI fails a PR into `v2` that does any of this, unless
it has the label `contract`. The protection of the `v2` branch makes the owners approve each
PR, with or without the label.

**Done.** M1 and each later milestone are done when CI passes, with `ruff check`,
`ruff format --check`, `ty check`, `pytest` and the frozen-path check, and when the owners
have run G1 and G2 on the gate machine and both pass.
