# Architecture

## Scope

Calibre is an open-source library of array functions for conformal forecasting:
validated panel input, point forecasts, hierarchy reconciliation, calibrated bands
and bounds, and orders from those bounds with their lost-sales cost. [Semantics](semantics.md) defines the calibration rules.

There are two loops over origins, both in `backtest`. `rolling_forecasts` makes
points, and `replay` calibrates them. Neither stores anything. `online.step` is one
calibration origin: it takes a state and returns the next one, so a product can save
the state between origins and continue later.

## Modules

A concept is one module or one package. Its contract is an `abc.ABC` with only
abstract methods, and each implementation subclasses it, so a missing method fails when
the object is made. In a module, the contract comes first, then the implementations. In
a package, the contract is in `base.py` and each implementation is one file.
`__init__.py` only re-exports.

| Module | Responsibility |
|---|---|
| `data/panel.py` | `Panel`, validated once |
| `data/hierarchy.py` | `Hierarchy`, node labels and the summing matrix |
| `forecast/models/base.py` | `Window`, `Covariate`, `Forecaster` and `Fitted` contracts |
| `forecast/models/` | `naive.SeasonalNaive`, `statsforecast.StatsForecastModel`, `mlforecast.MLForecast`, `neuralforecast.NeuralForecast` |
| `forecast/models/frames.py` | Long frames for the mlforecast and neuralforecast adapters |
| `forecast/reconcile.py` | `Reconciler` contract, `BottomUp`, `Identity`, `WlsStruct` |
| `conformal/targets.py` | `Target` contract, `Step`, `LeadTime`, and `columns`: steps to target columns |
| `conformal/scores.py` | `Score` contract, `Absolute`, `Signed` |
| `conformal/calibrators/base.py` | `Calibrator` and `QuantileCalibrator` contracts, `Feedback`, `State`, `Level`, `check_level` |
| `conformal/calibrators/ranks.py` | `score_quantile`, `retained_quantile`: rank, window, pooling |
| `conformal/calibrators/` | `split.SplitQuantile`, `aci.ACI`, `tracker.QuantileTracker` |
| `online/ledger.py` | Issued points and thresholds that wait for their targets, released as `Feedback` |
| `online/step.py` | `initial_state`, `step`, `Issue` |
| `online/state.py` | `flatten`, `unflatten` for storage |
| `backtest/forecasts.py` | `rolling_forecasts`, `Forecasts.residuals` |
| `backtest/replay.py` | `replay`, `Replay` |
| `decision/policy.py` | `critical_ratio`: the newsvendor level from costs. `order_up_to`: an order from a bound |
| `decision/lost_sales.py` | `settle`, `Settlement`: one period of lost-sales inventory and its cost |
| `metrics.py` | coverage, width, interval score, pinball, newsvendor cost |

Each package imports only these Calibre packages.

| Package | Imports |
|---|---|
| `data` | nothing |
| `forecast` | `data` |
| `conformal` | nothing |
| `online` | `conformal` |
| `backtest` | `data`, `forecast`, `conformal`, `online` |
| `decision` | nothing |
| `metrics` | nothing |

A calibration method is one file in `conformal/calibrators/`. It imports only
`calibrators.base` and `calibrators.ranks`.

`conformal` and `online` do not depend on hierarchy code. They need points and actuals.

`bench/` is a folder of scripts at the repository root, not a package and not in the
wheel. Its scripts may name datasets and reproduce benchmark protocols. They import the
installed `calibre` and each other. `calibre` never imports from `bench/`.

| Script | Responsibility |
|---|---|
| `bench/vn2.py` | VN2 files to `Panel` with the out-of-stock mask, hierarchy, and stock. `play`: the six-decision protocol |
| `bench/m5.py` | M5 files to `Panel`, the 12-level hierarchy, and the price covariate |
| `bench/compare.py` | Many calibrators on the same forecasts, one metrics row per method |
| `bench/vn2_conformal.py` | The VN2 run: forecasts, four calibrators, and the played cost |

Forecast adapters can depend on vendor libraries, but not on calibration.
Each vendor library is an extra: `calibre[stats]`, `calibre[ml]`, `calibre[neural]`.
`import calibre` needs none of them.

## Array contracts

B is the number of bottom series, N all nodes with bottoms first, T periods,
O origins, and H forecast steps.

| Array | Shape | Meaning |
|---|---|---|
| `Panel.values` | `[B, T]` float32 | Finite bottom history on a complete calendar |
| `Hierarchy.summing` | `[N, B]` sparse | Rows follow `Hierarchy.nodes`. Bottom rows are the identity. No two rows are equal |
| `Covariate.values` | `[B, 1]`, `[1, L]`, `[B, L]` | Static, calendar, or dynamic input from the first period |
| `Window.y` | `[S, t]` read-only | History of the S forecast series through the origin |
| `Window.x` | `[S or 1, 1 or t or t + H]` read-only | Covariates. Only `known_ahead` ones reach t + H |
| `Fitted.predict` output | `[S, H]` | Base point forecasts |
| Reconciler | `shape = (N, S)`, called on `[S, H]` | `BottomUp` (S = B), `Identity` or `WlsStruct` (S = N) |
| Origins | `[O]` int | Increasing period indexes, each the last observed period |
| Points | `[O, N, H]` | Reconciled points, owned by the caller |
| Residuals | `[O, N, H]` | `actual - point`, NaN until the actual is known |
| `Panel.censored` | `[B, T]` bool, optional | True where a value is a lower bound of the target |
| `Target.cover` | `[C, H]` bool | Steps that each target column sums. C = H for `Step`, 1 for `LeadTime` |
| `Feedback.scores` | `[K, N]` | One row per newly known (issuing origin, column), in origin order |
| Level | scalar or `[N, C]` | Owned by the calibrator. `[N, 1]` is one level per node, for example from `critical_ratio` |
| Threshold | `[N, C]` float32 | Issued per origin. inf means not ready |
| `Replay` arrays | `[O, N, C]` | Points, thresholds, bounds, targets, scores, and censored flags per column |
| State | nested dict of arrays | `flatten` gives one named array per key |

`online.step` gives the periods since the last origin to the ledger, scores the
columns whose last covered step is now known, gives them to the calibrator with the
thresholds issued for them, and only then issues for the new origin. In the ledger,
origins wait in a ring of slots, one per step up to the last covered step, so the
ledger state has a fixed shape.

A call owns the state it receives: it can write into those arrays, and the caller
continues with the returned state. A calibrator keeps its state in numpy arrays only,
never fitted objects. Anything else it needs is recomputed from those arrays. Every
state array keeps the node axis, so a product can shard state by node ranges unless
`groups` pools nodes across shards.

A model sees data only through `Window`. `rolling_forecasts` builds each window by
slicing, so a value after the origin cannot reach a model, except a covariate marked
`known_ahead`. `fit` returns a new fitted model and does not change the forecaster.
The model is fitted at the first origin and again every `refit_every` origins.
Aggregate nodes get covariates by the declared `aggregate` rule, sum or mean of their
bottom series. There is no default rule.

Model adapters that need long frames build them inside `fit` and `predict`, from the
last periods the model reads. At full M5 a long frame of the whole history is 58
million rows, 1 GB, and 180 ms to build at each origin. The last 84 periods are 44 MB
and 7 ms. With LightGBM on lags up to 28 and a 28-day rolling mean, a lookback of 84
gives the same points as the full history and 3.2 GB less peak memory.

`Panel` validates once at construction. Functions downstream trust its arrays. Hot
paths work on typed arrays, with no per-cell objects, hashes, or serialization.

## Costs

| Operation | Cost |
|---|---|
| `SplitQuantile` threshold | One sort of a contiguous `[K, N]` slice per column, `O(C * N * K log K)` |
| `replay` at M5 | 42,840 nodes, 64 origins, 28 steps, window 28: 3.5 s, 7.6 GB peak, against 2.5 s and 9.9 GB for the earlier stateless functions, with identical thresholds |
| Saved state at M5 | `LeadTime(7)` with `SplitQuantile(window=28, capacity=40)`: 26 MB |
| Bottom-up | `O(nnz(S) * H)` per origin |
| `WlsStruct` | Dense Cholesky of the `A x A` aggregate system once per hierarchy, A = N - B. At M5 with 5 single-attribute levels this is 3,073 x 3,073 float64, 75 MB. With the 12 standard levels it is 12,350 x 12,350, 1.2 GB, factored in 2.3 s |
| `StatsForecastModel` | One Python call per series and origin |
| `rolling_forecasts` | Node history and covariates aggregated once per call, windows are views |
| `NeuralForecast` | Two frames per fit, one per predict. Predict reads `input_size` periods. M5 NHITS, 1,000 steps on CPU, 365 fit periods, price feature: one fit and 4 predicts in 25 s, 4.7 GB peak |
| `MLForecast` | Two frames per fit or predict. M5 with LightGBM, 365 fit periods, price feature: about 5 s per fit, 8 origins with 2 fits in 12 s |
| Dynamic covariate | `[B, T + H]` float32, 229 MB at M5. With every node for `WlsStruct`, 252 MB more |

## Not implemented

These are approved directions for the library. None of them exist yet:

- Issuance at the start of the first forecast period instead of a last-observed index.
- Exact decimal levels with integer or rational rank arithmetic.
- Declared target support and conservative projection of bounds onto it.
- Normalized, weighted, and adaptive calibration.
- Model lifecycle beyond fit and refit: incremental update, native quantiles, and fitted residuals.
- A target transform pair for adapters other than mlforecast.
- Diagonal residual-variance reconciliation and a memory preflight for `WlsStruct`.

Each one is a semantic change. It updates [semantics](semantics.md) in the same diff.
