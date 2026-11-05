# Semantics

[Architecture](architecture.md) defines the modules, the public functions and the checks.
This document defines the rules. Each rule names the guarantees that check it. A change of a
rule is a contract PR, named as such, with its checks.

## Rules

| Topic | Rule | Consequence | Guarantees |
|---|---|---|---|
| Visibility | A value is visible at origin o once its period is at or before o. A model reads only its window: the values of each series through o. A score of origin o0 is known at origin o when the last period of its column is at or before o. A threshold issued at o reads only scores known at o. | A later value cannot change a point or a threshold issued earlier. | F3, C1, C2 |
| Lags | Each input has a lag. An observation has lag 0. The score of a column has the lag of its last step, counted in periods after its origin. Known is read from dates, not counted in origins. | Daily, weekly and irregular origins follow one rule. Weekly origins on daily data need no special case. | C2 |
| Spans | A series has values only from its first to its last period. At an origin outside them, it has no point. Outside them, it adds zero to the values of each aggregate. A forecast issued inside the span can cover steps after its end: there, the truth of the series is missing. | A late launch is a short series, not a run of zeros that counts as demand. An aggregate forecast that still counts a series after its end shows as an error of the aggregate. | H2, F4, R1 |
| Missing values | Panel values are finite. In points, truths and scores, NaN means none: outside a span, after the last period of the data, or where a model gives no value. In truths and scores, NaN is never zero and never a score. `BottomUp` is the one exception: a bottom with no point adds zero. | A column sum needs every one of its steps. A model that gives no value for a series inside its span makes its aggregates smaller. | P1, T2, R1 |
| Window | The threshold of a cell reads the scores at the `window` latest origins of the run whose scores are known at the issuing origin. A missing score keeps its place. n counts the finite scores among them. | n can be smaller than `window`. A new series is ready only after enough of its own scores are known. | C2 |
| Level | A level is data: one float, or one float per node, strictly between 0 and 1, passed to `replay`. `Split` holds the score and the window, not the level. `critical_ratio(holding, shortage)` gives the newsvendor level of each node. | Nodes with different costs get different levels in one run. | C3 |
| Conformal rank | conformalforecast's rule, with alpha = 1 - level: the k-th smallest of the n finite scores, with k = min(ceil((1 - alpha) × (n + 1)), n), and no interpolation. A cell is ready when n ≥ ceil(1 / alpha). Both ceilings round to 9 decimals first, as conformalforecast does. | At level 0.9, a cell needs 10 scores. At the critical ratio 1/1.2, it needs 6. | C3, C4 |
| Bands | `AbsoluteResidual` scores \|truth - point\| per node and column. The band is [point - threshold, point + threshold]. | The upper end has no one-sided coverage at the level. | C5 |
| Bounds | `SignedResidual` scores truth - point. The bound is point + threshold, and the lower end is -inf, where the point exists. On a `Span`, the score is the residual of the total over the span. | A bound is not a sum of band ends. It is not projected onto the support of the target: the caller applies floors. | C5 |
| Totals | Each band or bound is computed on the total that it serves, at its node, from the scores of that total. No function adds, averages or reconciles thresholds, bands or bounds. Reconciled points are coherent. Bands and bounds are not. | Quantiles do not add: the bound of an aggregate is not the sum of the bounds of its members, and the bound of a span is not the sum of step bounds. | R5, C6 |
| Censoring | A censored value is a lower limit of demand. A node value is censored when one member value is. A cell is censored when one value of its truth is. Censored scores enter windows as observed. | A band from sales covers sales, not demand. The flag stays, so a later correction can read it. | H2, C7 |
| Orders | An order at the critical ratio reads the bound of `SignedResidual` on a `Span` that covers the lead time and the review period, at `critical_ratio(holding, shortage)`. `order_up_to` orders the bound minus the stock and the pipeline, never below zero. | The bound is the inventory position that the order targets. | G2 |
| Settlement | `settle` receives the first pipeline slot, sells the smaller of stock and demand, loses unmet demand, then puts the order at the end of the pipeline. Holding cost is on the stock at the end of the period. | The lead time is the pipeline length. | G2 |
| Nodes | `Hierarchy.from_attributes` makes one aggregate per distinct value of each level. A level is one attribute or crossed attributes. An aggregate with the same bottom series as a bottom, the total, or an earlier aggregate is dropped. | No two nodes have the same bottom series, so `WlsStruct` does not count one sum twice. The full M5 hierarchy gives its 42,840 standard series. | H1 |
| Same numbers | Points equal the engine's own cross-validation at the same settings. Thresholds equal conformalforecast's exact rank on the same scores. Reconciled points equal hierarchicalforecast's. | Calibre adds the clock, not new numbers. | F1, C3, R4 |
| Splits | A call split along its split axes and joined again gives the same bits. `rolling_forecasts` splits by series, through `Panel.take`, because its models fit each series alone, and by origins. `reconcile` splits by models and by origins. `replay` splits by nodes, and never by origins: a threshold reads the scores of earlier origins. A slice of a panel keeps the calendar of the panel. Calibre imports no cluster framework: the caller splits the work and runs the parts where it wants. | The numbers do not depend on where the work is cut, or on how many processes run the parts. A forecast or a reconciled point computed alone at one origin equals that origin in a full run. | A2, A4, P4, F2, F7, R6, C8 |
| Claims | A `Split` band or bound is a marginal, finite-sample claim under exchangeable scores, at its level, node and column. Overlapping span totals are not exchangeable. | It is not conditional coverage. A sales history gives sales coverage, also when values are flagged as censored. | |
| Tuning | A configuration chosen by its scores at the origins through o has seen those origins. Its scores at those origins never calibrate it: `replay` gets only its forecasts at the origins after o. | A cell is ready only when its window holds `ceil(1 / alpha)` known scores of origins after o. Coverage measured at the tuning origins looks better than it is. | |

## Benchmark input preprocessing

These policies live only in `bench/`. The rules above do not change for them. The loaders
of the additional datasets verify pinned raw-file hashes before parsing. Beijing, Bike and Favorita always reindex
their calendar, forward-fill causally, and trim the leading prefix until every series has an
observation. They return the original-target mask and the number of trimmed periods. There
is no global missing-data switch or pre-trimming series limit. ETTh1, LSTNet and
FreshRetailNet do not impute targets. The M5 and VN2 loaders read the data as 0.1 does:
each series spans the whole calendar, and VN2 counts a product as selling zero before it is
listed. Those zeros are data, not imputed values. Calibre has no mask for imputed values, so
their scores enter calibration as observed. Callers exclude imputed targets, including
aggregates and spans that contain one, from evaluation.

LSTNet dates are positional labels, not a recovered calendar. FreshRetailNet's train and
evaluation dates are module constants, and its stockout mask marks sales as lower bounds,
not demand. See the [benchmark guide](../bench/README.md) for the details of each dataset.

## Later

None of these is part of M1. Each one comes with the contract PR of its milestone.

| Topic | Rule when it comes |
|---|---|
| Production (on a measured need) | `step` takes a state and returns the next one. It issues for the last origin with the same kernels as `replay`, and its output equals the last origin of `replay`. The ledger keeps the order and the stock in force at each origin, with the forecast that waits for its truth. |
| Pools (M2) | Equal labels pool scores, and each member gets the pool threshold. A run that splits nodes into ranges cuts only at pool edges. Pool coverage is not node coverage. |
| Decision backtest (M2) | It runs the decision weeks in order from a starting stock: `order_up_to` on the bound issued at each decision, then `settle`. Its cost is the objective of a tuning run. |
| Global models (M3) | A global model fits on all series at once. A call that holds one splits by origins, never by series. |
| Random draws (M3) | A model that draws random numbers takes a seed. It draws for each series from the seed and the series id, never from the position of the series, the process or a global state. A series gets the same draws in any split. |
| Decisions on more than one quantile | They read `Replay.score` under the same visibility rule: the scores beyond the bound, or the whole window. |
| Censored scores | A correction for censored scores reads the flag and the record of orders and stock. |
| Imputed values | A mask of observed values. A value that is not observed is a missing truth. |
| Online methods (M4) | ACI, the quantile tracker and PID are scans over origins inside `replay`. Each one compares a score with the threshold issued for it. No object keeps state between calls. |

## Approved directions

These directions are approved and not in effect. Each one would change a rule above, through
a contract PR.

| Topic | Default | Consequence |
|---|---|---|
| Issuance | Issuance is the start of the first forecast period. Target completion and publication must both be strictly before it. | Origins stop being last-observed periods. Training views respect publication times. |
| Level precision | Levels are canonical decimals at input. Ranks use exact integer or rational arithmetic. | Float32 describes score storage, not probability. Adjacent levels stay distinct. No epsilon. |
| Target support | Callers declare support. Real-valued targets are allowed. Bounds are projected conservatively onto support. | Support is not inferred from samples. Clipping points breaks hierarchy coherence. |
| Normalization | A positive, causal issuance-time scale is stored with each forecast. Zero or invalid scales need a named rule. | No division by an epsilon. No scale recomputed with later data. |
| Weighted calibration | Declare weight origin, time decay, test-point mass, and weighted rank and readiness. | Weighting does not inherit the unweighted finite-sample claim. |
| Model generations | A changed model, feature transform, or calibration method resets calibration by default. | Reset warmup stays visible. No silent transfer of old scores. |
