# Semantics

[Architecture](architecture.md) defines modules and array contracts. Change a rule here
in the same diff as its code and tests. A semantic change is named, not hidden in a
refactor.

## Current rules

| Topic | Rule | Consequence |
|---|---|---|
| Nodes | `Hierarchy.from_attributes` makes one aggregate per distinct value of each level. A level is one attribute or crossed attributes. An aggregate with the same bottom series as a bottom, the total, or an earlier aggregate is dropped. | No two nodes have the same bottom series, so `WlsStruct` does not count one sum twice. The full M5 hierarchy gives its 42,840 standard series. |
| Level | A calibrator owns its target level, set at construction: a scalar, or `[N, C]` per node and column. A level must be strictly between zero and one. A vector `[N]` is rejected, because it broadcasts on the column axis. `ACI` targets the level of its base, which must be a `QuantileCalibrator`. `critical_ratio(holding, shortage)` gives the newsvendor level. | Cost-calibrated levels differ per node. `online.step` and `replay` take no level. |
| Rank | One-based `ceil((n+1)*q)`, no interpolation. Ready only when the rank is at most n. No separate minimum history. | q = 0.9 is ready with nine scores. Ranks use float arithmetic for now. |
| Bands | `Absolute` on a `Step` target scores absolute residuals of reconciled points, per node and step, for a two-sided band. Steps never share scores. | An upper endpoint does not have one-sided coverage at the nominal level. |
| Lead-time bounds | `Signed` on a `LeadTime(steps)` target scores signed residual sums over the first `steps` steps, one score per origin and node. The bound is the point total plus the threshold. | It is not a sum of band endpoints. It is not projected onto target support. The caller applies support and business floors. |
| Causality | `online.step` gives a calibrator a score only when the last step it covers is at or before the current origin, and before that origin issues. A model reads only its `Window`: history and covariates through the origin, and `known_ahead` covariates through the last step. | A later actual cannot change an issued threshold. A later value cannot change a point, except through a covariate declared known ahead. |
| Feedback | Each feedback row is one score column of one issuing origin, in origin order, with the threshold issued for it and a censored flag. A column is censored when any step it covers is censored. Online calibrators compare each score with its issued threshold. | ACI and quantile tracking react to the thresholds that were issued, not to thresholds recomputed later. |
| Missing values | `Panel` rejects NaN and infinity. In residuals, NaN means not yet known. It is never zero and never a score. | A lead-time score needs every step. A nonfinite constructed score, including an overflowed sum, is excluded before retention. |
| Rolling window | `window` keeps the last K resolved origin rows per pool and step, before pooling. A row is resolved for a pool when it has at least one finite score. | Holes do not consume rows. n counts finite retained scores, so n can differ from K. |
| Pooling | Equal integer labels in `groups` pool raw residual scores. Every member gets the pool quantile. A level per node stays per node within a pool. | Pool coverage does not establish per-node coverage. Pool only scores with comparable meaning. |
| Settlement | `settle` receives the first pipeline slot, sells the smaller of stock and demand, loses unmet demand, then puts the order at the end of the pipeline. Holding cost is on the stock at the end of the period. `order_up_to` orders the bound minus stock and pipeline, never below zero. | The lead time is the pipeline length. A bound for an order must cover the lead time and the review period, for example a `LeadTime` target of that length. |
| Loss | A `Loss` is the cost of issued bounds against a known target, any shape in the threshold, with a declared finite `maximum`. The caller passes only known targets. `Miss` is 1 outside the bounds, maximum 1. A loss does not see the censored flag, so a censored target is scored as observed. | A loss of sales targets is a risk on sales, not on demand. |
| Risk control | `RiskControl(loss, grid, alpha)` issues the smallest grid threshold with `(loss sum + B) / (n + 1) <= alpha`, where n counts known targets and B is the loss maximum, per node and column. No feasible threshold gives inf. `alpha` is between 0 and B. No window and no pooling. | With `Miss` it is the `SplitQuantile(1 - alpha)` threshold, rounded up to the grid. |
| Claims | A `SplitQuantile` band or bound is a marginal, finite-sample claim under exchangeable scores, at the stated level, column, and pool. `ACI` and `QuantileTracker` give only long-run marginal coverage, with no finite-sample claim. `RiskControl` with a loss nonincreasing in the threshold controls the expected loss at `alpha` under exchangeable targets (Angelopoulos et al. 2024). For other losses the excess over `alpha` is bounded by the stability of the chosen threshold (Angelopoulos 2026), which Calibre does not estimate. Overlapping lead-time targets are not exchangeable. | It is not conditional coverage. A sales history gives sales coverage, not demand coverage, also when targets are flagged as censored. |

## Approved, not implemented

| Topic | Default | Consequence |
|---|---|---|
| Issuance | Issuance is the start of the first forecast period. Target completion and publication must both be strictly before it. | Origins stop being last-observed indexes. Training views respect publication times. |
| Level precision | Levels are canonical decimals at input. Ranks use exact integer or rational arithmetic. | Float32 describes score storage, not probability. Adjacent levels stay distinct. No epsilon. |
| Target support | Callers declare support. Real-valued targets are allowed. Bounds are projected conservatively onto support. | Support is not inferred from samples. Clipping points breaks hierarchy coherence. |
| Normalization | A positive, causal issuance-time scale is stored with each forecast. Zero or invalid scales need a named rule. | No division by an epsilon. No scale recomputed with later data. |
| Weighted calibration | Declare weight origin, time decay, test-point mass, and weighted rank and readiness. | Weighting does not inherit the unweighted finite-sample claim. |
| Model generations | A changed model, feature transform, or calibration method resets calibration by default. | Reset warmup stays visible. No silent transfer of old scores. |
