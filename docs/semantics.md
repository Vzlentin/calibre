# Semantics

[Architecture](architecture.md) defines modules and array contracts. Change a rule here
in the same diff as its code and tests. A semantic change is named, not hidden in a
refactor.

## Current rules

| Topic | Rule | Consequence |
|---|---|---|
| Nodes | `Hierarchy.from_attributes` makes one aggregate per distinct value of each level. A level is one attribute or crossed attributes. An aggregate with the same bottom series as a bottom, the total, or an earlier aggregate is dropped. | No two nodes have the same bottom series, so `WlsStruct` does not count one sum twice. The full M5 hierarchy gives its 42,840 standard series. |
| Level | A calibrator owns its target level, set at construction: a scalar, or `[N, C]` per node and column. A level must be strictly between zero and one. A vector `[N]` is rejected, because it broadcasts on the column axis. `ACI` targets the level of its base, which must be a `QuantileCalibrator`. `critical_ratio(holding, shortage)` gives the newsvendor level. | Cost-calibrated levels differ per node. `online.step` and `replay` take no level. |
| Conformal rank | `SplitQuantile` (including an ACI base) uses one-based `ceil((n+1)*q)`, no interpolation. Ready only when the rank is at most n. No separate minimum history. | q = 0.9 is ready with nine scores. Ranks use float arithmetic for now. This is not the empirical EnbPI rank. |
| Bands | `Absolute` on a `Step` target scores absolute residuals of reconciled points, per node and step, for a two-sided band. Steps never share scores. | An upper endpoint does not have one-sided coverage at the nominal level. |
| Lead-time bounds | `Signed` on a `LeadTime(steps)` target scores signed residual sums over the first `steps` steps, one score per origin and node. The bound is the point total plus the threshold. | It is not a sum of band endpoints. It is not projected onto target support. The caller applies support and business floors. |
| Causality | `online.step` gives a calibrator a score only when the last step it covers is at or before the current origin, and before that origin issues. A model reads only its `Window`: history and covariates through the origin, and `known_ahead` covariates through the last step. | A later actual cannot change an issued threshold. A later value cannot change a point, except through a covariate declared known ahead. |
| Feedback | Each feedback row is one score column of one issuing origin, in origin order, with its issued scalar or paired parameters and a censored flag. A column is censored when any step it covers is censored. Scalar scores are compared with their issued threshold; paired corrections use archived endpoints. | ACI and quantile tracking react to issued thresholds, not thresholds recomputed later. Paired corrections cannot be tested by comparing one score with one scalar threshold. |
| Missing values | `Panel` rejects NaN and infinity. In residuals, NaN means not yet known. It is never zero and never a score. | A lead-time score needs every step. A nonfinite constructed score, including an overflowed sum, is excluded before retention. |
| Rolling window | `window` keeps the last K resolved origin rows per pool and step, before pooling. A row is resolved for a pool when it has at least one finite score. | Holes do not consume rows. n counts finite retained scores, so n can differ from K. |
| Pooling | Equal integer labels in `groups` pool raw residual scores. Every member gets the pool quantile. A level per node stays per node within a pool. | Pool coverage does not establish per-node coverage. Pool only scores with comparable meaning. |
| Settlement | `settle` receives the first pipeline slot, sells the smaller of stock and demand, loses unmet demand, then puts the order at the end of the pipeline. Holding cost is on the stock at the end of the period. `order_up_to` orders the bound minus stock and pipeline, never below zero. | The lead time is the pipeline length. A bound for an order must cover the lead time and the review period, for example a `LeadTime` target of that length. |
| Loss | A `Loss` is the cost of issued bounds against a known target, any shape in the threshold, with a declared finite `maximum`. The caller passes only known targets. `Miss` is 1 outside the bounds, maximum 1. A loss does not see the censored flag, so a censored target is scored as observed. | A loss of sales targets is a risk on sales, not on demand. |
| Risk control | `RiskControl(loss, grid, alpha)` issues the smallest grid threshold with `(loss sum + B) / (n + 1) <= alpha`, where n counts known targets and B is the loss maximum, per node and column. No feasible threshold gives inf. `alpha` is between 0 and B. No window and no pooling. | With `Miss` it is the `SplitQuantile(1 - alpha)` threshold, rounded up to the grid. |
| Claims | A `SplitQuantile` band or bound is a marginal, finite-sample claim under exchangeable scores, at the stated level, column, and pool. `ACI` and `QuantileTracker` give only long-run marginal coverage, with no finite-sample claim. `RiskControl` with a loss nonincreasing in the threshold controls the expected loss at `alpha` under exchangeable targets (Angelopoulos et al. 2024). For other losses the excess over `alpha` is bounded by the stability of the chosen threshold (Angelopoulos 2026), which Calibre does not estimate. Overlapping lead-time targets are not exchangeable. | It is not conditional coverage. A sales history gives sales coverage, not demand coverage, also when targets are flagged as censored. |

## EnbPI: explicit semantic additions

- **Bootstrap point model.** `BootstrapEnsemble` is a local one-step lag regressor,
  not a calibrator. Each bag samples training rows with replacement and sorts their
  indexes; all series share the same resamples. Learners and their preprocessing
  are cloned and fitted inside each bag. Covariates and horizons other than one
  are rejected. Models stay frozen after fit; later windows supply only new lags.
- **OOB initialization and center.** Every training row must have an OOB learner.
  Train at least `n_estimators` bags, extending up to `max_estimators` (default
  max(4*n_estimators, n_estimators+20)); fail explicitly if a row remains uncovered.
  For row i, average only learners that excluded i. A future point is the mean
  of these initial leave-one-out predictors, with each learner weighted by
  mean_i(1{learner excludes i}/number of learners excluding i). It is not generally
  the unweighted ensemble mean. OOB errors use these row-specific OOB predictions.
- **Causality of seeds.** `FitResiduals.origins` indexes the last observed period of
  each lag row, not the model-fit date. OOB fits can use later rows within the
  initial training prefix: these are not chronological forecast errors. All
  initial targets must nevertheless be known at the first forecast origin.
  `rolling_forecasts(initial_residuals=True)` verifies that timing and reconciles
  aligned signed errors with the same linear map as points. It rejects refits.
  The caller seeds an empty `EnbPI` state explicitly; no artificial issued
  thresholds or misses are invented for the OOB period.
- **Signed empirical intervals.** `EnbPI(level, window)` retains the last `window`
  finite signed errors per node and column. There is no pooling. Each new error
  is actual minus the archived point, once its target is known. Missing or
  nonfinite constructed errors do not advance the ring. `window` is explicit;
  choose the initial OOB sample count for the original full-length residual window.
- **Asymmetric parameters and ranks.** `Asymmetric` scores signed errors and maps
  parameters [lo, hi] to point+[lo, hi]. `EnbPI` selects the shortest
  [Q(beta), Q(level+beta)] interval over `beta_grid_size` equally spaced beta values
  from 0 to 1-level inclusive. Ties choose the first beta. Q uses the empirical
  inverse CDF: ceil(n*q), clamped to ranks 1 through n, including finite endpoints
  at q=0 and q=1. There is no n+1 correction. Any finite sample makes a cell ready;
  an empty cell issues [-inf, inf]. Parameters are [N, C, 2] float32, while points
  and bounds remain [N, C]. Scalar quantile and risk calibrators are incompatible
  with this score and are rejected by the first online step.
- **State and continuation.** EnbPI state is a fixed-size NumPy residual ring,
  counts, and insertion positions, each keeping the node axis. Reconstruct
  quantiles from those arrays. The ledger archives both issued corrections;
  paired `Feedback.misses` and `Replay.covered` use endpoint membership. Scalar
  score comparisons are unchanged. A caller can pass
  a seeded or saved online state to `replay(state=...)`; already consumed periods
  are not fed again. The caller must retain the same model, target, score, and
  calibration configuration when resuming.
- **Claims and scope.** Row-wise OOB exclusion does not remove dependence between
  overlapping lag rows and does not certify chronological exclusion. These
  empirical asymmetric intervals do not inherit split conformal's exchangeable
  finite-sample guarantee. EnbPI validity requires assumptions on errors and model
  estimation (Xu and Xie, 2021, https://proceedings.mlr.press/v139/xu21h.html).
  Generic residual calibration accepts delayed target columns, but this bootstrap
  adapter implements no multi-horizon, simultaneous-path, or conditional-coverage
  guarantee. Reconciled residual intervals likewise carry no new such claim.

## SPCI

- **Scope and explicit horizon.** `SPCI(level, window, n_lags, ahead=...)`
  returns paired residual corrections, normally with `Asymmetric`. Each node and
  target column has its own history and forest, with no pooling. `ahead` is required:
  use `np.arange(1, H + 1)` for `Step()` or `steps` for `LeadTime(steps)`. It must
  equal each column's last covered step, in the same calendar-index units as
  `Feedback.origin`. The caller owns this configuration; the generic calibrator
  contract does not receive the target cover and cannot verify that match.
- **Causal direct-horizon pairs.** Let e[o,h] be actual minus the archived point
  for a forecast issued at o, revealed at o+h. The context of training label
  e[o,h] consists of the last `n_lags` finite errors e[u,h] with u+h <= o,
  in issuing-origin order. This is not the immediately preceding retained label
  when feedback is delayed. Fit the conditional distribution of e[o,h] from these
  contexts, then query it with the latest revealed `n_lags` errors. Historical
  contexts cannot use errors that were revealed after the historical emission,
  even if they are known when fitting today. Each column learns directly; there
  is no recursive substitution of predicted residuals or future observations.
- **Window, gaps, and missing errors.** Keep the last `window` finite signed
  errors and their original issuing indexes per cell. NaN and nonfinite constructed
  errors occupy no slots; censored feedback is treated as observed, not as latent
  demand. Even missing feedback advances the last-consumed origin, preventing
  duplicate consumption. Origins must increase within each column. Build pairs
  entirely from the retained window, excluding labels without a complete historical
  context. Gaps do not become zeros, equally spaced retained-score time, or extra
  training labels. Contexts use the last available finite errors, not necessarily
  consecutive calendar periods; gap length itself is not a predictor.
- **Quantile forest.** Fit scikit-learn `RandomForestRegressor` to lag rows and
  signed residual labels. For a query, each tree gives total mass one, shared
  equally among all training rows in its query leaf. Average these per-tree masses
  to obtain QRF weights as in equations 15–17 of Xu and Xie (2023). All training
  rows are evaluated in each fitted tree, including rows outside its bootstrap
  sample; these are quantile-distribution weights, not OOB calibration errors.
  Keep only positive-weight support and invert its weighted empirical CDF without
  interpolation or a split-conformal n+1 correction. Quantiles at 0 and 1 use the
  finite minimum and maximum of that support. No third-party quantile-estimator
  package, mean-of-tree-predictions quantile, or model fallback is used.
- **Tails.** Search `beta_grid_size` (default 21) equally spaced beta values on
  [0, 1-level], inclusive. Choose the shortest [Q(beta), Q(level+beta)]; width ties
  choose the first beta. Parameters are [N, C, 2] float32; level can be scalar or
  [N, C]. These are residual endpoints, not scalar nonconformity thresholds, and
  cannot be wrapped by `ACI` as a `QuantileCalibrator`.
- **Readiness.** Require at least `min_train_samples` complete pairs per cell,
  default 20, an operational warm-up choice rather than a coverage theorem.
  Otherwise issue [-inf, inf], without silently switching to empirical EnbPI.
  On consecutive origins with m retained errors and horizon h, the pair count is
  max(0, m - n_lags - h + 1). Thus choose a window of at least
  n_lags + h - 1 + min_train_samples to ever become ready on that schedule.
  Irregular schedules use actual origin comparisons instead of this count formula.
- **Initialization.** `seed(state, residuals, origins=...)` accepts [K, N, C]
  signed errors and strictly increasing nonnegative int64 issuing origins [K].
  All seed targets must already be known at the first issuing origin; this timing
  and the suitability of the historical point forecasts are caller obligations.
  Do not use in-sample fitted values as forecast errors. Seeds require an empty
  state, including no consumed missing feedback. No issued intervals or misses
  are fabricated. Without a seed, ordinary feedback accumulates the same history.
  `BootstrapEnsemble` is unchanged and remains limited to one-step prediction.
- **State, continuation, and cost.** State contains only fixed-size NumPy residual
  and origin rings, counts, insertion positions and last-consumed origins, all
  retaining the node axis. `threshold` changes neither arrays nor instance state.
  Forests are refitted per ready node/column and discarded. Defaults are 100 trees,
  depth 5, minimum leaf size 5, bootstrap sampling, `random_state=0`, and `n_jobs=1`.
  A fixed integer random state is required so saved arrays reproduce continuation
  with a new identically configured calibrator. Save the pending ledger as well,
  and preserve model generation, target, score, horizons and configuration.
  Model-generation changes require caller-managed fresh initialization. For Step,
  up to N × H forest fits are needed per origin; no full-panel scalability claim
  is made. The optional `spci` extra installs scikit-learn; importing `calibre`
  and evaluating unready cells load no forest library.
- **Claims and relation to the paper.** This is the conditional-residual-quantile
  construction of SPCI adapted to Calibre's direct per-column residual streams and
  caller-owned rolling or block origins, not a reproduction of Algorithm 3's
  bootstrap point ensembles, block-residual protocol or experimental results.
  Asymptotic conditional validity in the paper requires consistency and dependence
  assumptions, not merely selecting a QRF. This implementation does not establish
  those assumptions or offer distribution-free finite-sample conditional coverage,
  simultaneous-path coverage, or joint coverage across nodes. Reconciled points
  do not make the independently calibrated endpoints hierarchically coherent.
  Lead-time calibration is a direct interval for the total, not a sum of step
  endpoints. Reference: Xu and Xie,
  [Sequential Predictive Conformal Inference for Time Series](https://arxiv.org/abs/2212.03463v3),
  equations 10–17 and Appendix B.3.

## Conformal PID: explicit semantic additions

- **Scope and variant.** `ConformalPID(level, ...)` calibrates nonnegative scalar
  scores, normally `Absolute`, independently per node and column, without pooling.
  Negative finite scores and paired parameters are rejected on update. Its
  anticipation is a causal regularized AR(1), not the paper's Theta model or a
  finite-difference derivative. A signed-score extension is not implemented.
- **Explicit initialization.** `seed(state, scores)` takes [K, N, C] nonnegative
  scores in time order, with NaN holes; infinity is rejected. Every cell needs at
  least one finite score. Keep the last `window` finite scores per cell. Seeds
  must already be known at the first issuing origin, from predictions appropriate
  to the same model and target configuration. No historical thresholds or misses
  are invented. Seeding requires an empty state; seed every cell before feedback.
  An empty state issues inf, but update before seed raises instead of silently
  choosing a scale from warm-up feedback.
- **Initial rank and scale.** The initial quantile is the empirical inverse CDF
  at `level`: rank ceil(n * level), clamped to 1 through n, with no n+1 correction.
  Any nonempty seed suffices. Set
  `scale = max(std(seed), 0.1 * mean(seed), 1e-8)`, with population standard
  deviation, using only retained seed scores. The scale stays fixed. Initialize
  P to this quantile minus the initial scorecast, so the first threshold is the
  empirical quantile. These floors and the fixed rate scaling are explicit
  algorithm choices, not issuance-time normalization. Unlike the reference code's
  raw KI and P rate `lr * (max - min)` recomputed over a trailing `T_burnin` window,
  both gains here use the fixed seed scale, including `integral_gain * scale`.
- **Feedback and P.** A finite revealed score misses iff it is strictly above
  its archived issued threshold, including thresholds below zero or infinite.
  Equality is a hit. Let `e = miss - (1 - level)`. Update
  `P += learning_rate * scale * e`, `E += e`, and `t += 1`.
  Append the score to its cell's ring. Missing or nonfinite constructed scores
  change none of these arrays. Censored feedback is scored as observed, as for
  other controllers; this is not latent-demand calibration.
- **Integrator and issued threshold.** After t valid updates, set
  `z = E * log(t + 1) / (saturation_scale * (t + 1))`.
  I is zero at t=0 or when `integral_gain=0`; otherwise it is
  `integral_gain * scale * tan(z)` for `-pi/2 < z < pi/2`,
  and signed infinity outside that range, including the endpoints.
  Never multiply zero gain by infinity. Issue `P + I + scorecast`, as float32;
  P, E, scale, and score history are float64.
- **AR(1) scorecasting.** Use the last min(window, scorecast_window) retained
  scores in chronological order. Fewer than three scores give their mean.
  Otherwise regress successive scores with an intercept, adding
  `max(scale**2 * 1e-6, 1e-12)` to the centered sum-of-squares denominator.
  Clip the slope phi to [-0.99, 0.99]. With paired means mean_x and mean_y, set
  `mu = (mean_y - phi * mean_x) / (1 - phi)` and forecast
  `max(0, mu + phi**ahead * (last - mu))`, propagating the fitted intercept.
  The constructor's `ahead` is a positive integer scalar or [C] array, default 1.
  Set it to each target column's last covered step: `np.arange(1, H + 1)` for
  `Step()` and `steps` for `LeadTime(steps)`. It is fixed configuration, not a
  learned clock; delayed releases and gaps do not change it.
  Disabled scorecasting is exactly zero. Neither rate scales nor fitted AR objects
  are updated or hidden on the calibrator instance.
- **Time and continuation.** Feedback rows are consumed in ledger order,
  vectorized over nodes. The integrator clock counts valid updates per cell,
  not calendar periods or calls to step. Step and lead-time columns learn only
  when their last covered target is known, and use their own archived parameters.
  Threshold evaluation does not mutate state. Save the NumPy state and pending
  ledger together; resume with the same target, score, model, and configuration.
- **Empty and unbounded sets; evaluation limitation.** With `Absolute`, a
  negative threshold gives an empty set, +inf the whole line, and -inf an empty
  set. These are intentional controller decisions, not necessarily warm-up.
  `Feedback.misses`, `Replay.covered`, and `metrics.coverage` count them correctly.
  Do not mask saturation dates or clip thresholds to make intervals look finite.
  Generic `metrics.width` and `interval_score` currently assume ordered endpoints:
  they have no empty-set convention and can return negative widths or NaN.
  `bench/compare.py` treats every infinite threshold as unready and cannot
  evaluate PID without a separately specified evaluation change.
- **Claims.** This implementation variant makes no distribution-free
  finite-sample, conditional, or simultaneous-path coverage claim. Frequency
  control requires the controller's assumptions; it does not follow solely from
  these recursions, especially when feedback is delayed, targets overlap, or
  integral control is disabled. AR(1) does not reproduce the authors' numerical
  results. Reference: Angelopoulos, Candès and Tibshirani,
  [Conformal PID Control](https://arxiv.org/abs/2307.16895).

## Approved, not implemented

| Topic | Default | Consequence |
|---|---|---|
| Issuance | Issuance is the start of the first forecast period. Target completion and publication must both be strictly before it. | Origins stop being last-observed indexes. Training views respect publication times. |
| Level precision | Levels are canonical decimals at input. Ranks use exact integer or rational arithmetic. | Float32 describes score storage, not probability. Adjacent levels stay distinct. No epsilon. |
| Target support | Callers declare support. Real-valued targets are allowed. Bounds are projected conservatively onto support. | Support is not inferred from samples. Clipping points breaks hierarchy coherence. |
| Normalization | A positive, causal issuance-time scale is stored with each forecast. Zero or invalid scales need a named rule. | No division by an epsilon. No scale recomputed with later data. |
| Weighted calibration | Declare weight origin, time decay, test-point mass, and weighted rank and readiness. | Weighting does not inherit the unweighted finite-sample claim. |
| Model generations | A changed model, feature transform, or calibration method resets calibration by default. | Reset warmup stays visible. No silent transfer of old scores. |
