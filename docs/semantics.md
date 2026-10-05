# Semantics

[Architecture](architecture.md) defines modules and array contracts. Change a rule here
in the same diff as its code and tests. A semantic change is named, not hidden in a
refactor.

## Current rules

| Topic | Rule | Consequence |
|---|---|---|
| Nodes | `Hierarchy.from_attributes` makes one aggregate per distinct value of each level. A level is one attribute or crossed attributes. An aggregate with the same bottom series as a bottom, the total, or an earlier aggregate is dropped. | No two nodes have the same bottom series, so `WlsStruct` does not count one sum twice. The full M5 hierarchy gives its 42,840 standard series. |
| Rank | One-based `ceil((n+1)*q)`, no interpolation. Ready only when the rank is at most n. No separate minimum history. | q = 0.9 is ready with nine scores. Ranks use float arithmetic for now. |
| Bands | `width` uses absolute residuals of reconciled points, per node and step, at a two-sided level. Steps never share scores. | An upper endpoint does not have one-sided coverage at the nominal level. |
| Window bounds | `window_bound` uses signed residual sums over the first `steps` steps, one score per origin and row. It adds the score quantile to the point total. | It is not a sum of band endpoints. It is not projected onto target support. The caller applies support and business floors. |
| Causality | A function reads only origins before `rows`. `resolved_rows` counts earlier origins whose target period is at or before the issuing origin. A model reads only its `Window`: history and covariates through the origin, and `known_ahead` covariates through the last step. | Later residuals in the array cannot change a band. A later value cannot change a point, except through a covariate declared known ahead. |
| Missing values | `Panel` rejects NaN and infinity. In residuals, NaN means not yet known. It is never zero and never a score. | A window score needs every step. A nonfinite constructed score, including an overflowed sum, is excluded before retention. |
| Rolling window | `window` keeps the last K resolved origin rows per pool and step, before pooling. A row is resolved for a pool when it has at least one finite score. | Holes do not consume rows. n counts finite retained scores, so n can differ from K. |
| Pooling | Equal integer labels in `groups` pool raw residual scores. Every member gets the pool quantile. `window_bound` keeps one level per row within a pool. | Pool coverage does not establish per-node coverage. Pool only scores with comparable meaning. |
| Claims | A band or bound is a marginal, finite-sample claim under exchangeable scores, at the stated level, step or window, and pool. | It is not conditional coverage. A sales history gives sales coverage, not demand coverage. |

## Approved, not implemented

| Topic | Default | Consequence |
|---|---|---|
| Issuance | Issuance is the start of the first forecast period. Target completion and publication must both be strictly before it. | Origins stop being last-observed indexes. Training views respect publication times. |
| Level precision | Levels are canonical decimals at input. Ranks use exact integer or rational arithmetic. | Float32 describes score storage, not probability. Adjacent levels stay distinct. No epsilon. |
| Target support | Callers declare support. Real-valued targets are allowed. Bounds are projected conservatively onto support. | Support is not inferred from samples. Clipping points breaks hierarchy coherence. |
| Normalization | A positive, causal issuance-time scale is stored with each forecast. Zero or invalid scales need a named rule. | No division by an epsilon. No scale recomputed with later data. |
| Weighted calibration | Declare weight origin, time decay, test-point mass, and weighted rank and readiness. | Weighting does not inherit the unweighted finite-sample claim. |
| Adaptive updates | Process feedback by availability time, then stable target, origin, node, and step order. Store the level used at issuance. | Today's level does not redefine a past event. |
| Model generations | A changed model, feature transform, or calibration method resets calibration by default. | Reset warmup stays visible. No silent transfer of old scores. |
