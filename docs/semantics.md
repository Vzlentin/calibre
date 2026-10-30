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

## Conformal PID

- **Scope.** `ConformalPID(level, ...)` calibrates nonnegative scalar
  scores, normally `Absolute`. P/I updates, initial ranks, and scales are per node
  and column, without score pooling. Negative finite scores and paired parameters
  are rejected on update. Its anticipation comes from an explicitly injected
  `Scorecaster`, not a finite-difference derivative. `scorecaster` is a required
  constructor choice. A signed-score extension is not implemented.
- **Explicit initialization.** `seed(state, scores)` takes nonnegative scores
  in time order, with NaN holes; infinity is rejected. Every cell needs at
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
  Never multiply zero gain by infinity. Issue `P + I + scorecast`.
- **Injected scorecasting.** The PID supplies the last
  min(window, scorecast_window) retained scores, without normalization.
  It validates the injected scorecaster's output shape and finiteness, then floors
  predictions at zero. None produces exactly zero. Overflow raises, separately
  from intentional infinity in the integrator and thresholds.
  `ahead` is fixed positive integer configuration, default 1: choose
  `np.arange(1, H + 1)` for `Step()` and `steps` for `LeadTime(steps)`.
  Delayed releases and gaps do not change this retained-score clock.
- **Forecaster adapter.** `ForecastScorecaster` groups cells by retained length
  across nodes and columns, fits once per length with horizon = ahead.max(),
  and selects each cell's ahead-th prediction. Local models must give the same
  h-th prediction regardless of the total horizon. A global forecaster may share
  parameters across both nodes and columns within a batch; controller updates
  remain per cell. The adapter validates only forecaster output shape and
  finiteness; fitting errors propagate without a silent model fallback.
  Its windows have no covariates and use positional daily dates in retained-score
  time, not calendar time. Calendar-aware
  features and calendar seasonality across gaps are not reconstructed.
- **Short histories.** The adapter owns the mean fallback: empty cells predict
  zero, and cells below `min_history` (default 3) use their retained mean.
  AR1 requires at least three values and has no short-history fallback.
  To ever fit a model, min_history must not exceed
  min(window, scorecast_window). StatsForecast Theta needs at least four values;
  for seasonal models choose suitable warm-up, for example min_history=14 for
  `Theta(season_length=7)`.
- **AR(1).** Regress successive scores by ordinary least squares with an
  intercept and no regularization. A zero centered variance gives slope zero.
  Clip slope phi to [-slope_limit, slope_limit], with slope_limit=0.99 by default, set
  `mu = (mean_y - phi * mean_x) / (1 - phi)`, and forecast
  `mu + phi**ahead * (last - mu)`. Generic forecasts may be negative.
- **Theta.** `StatsForecastModel(Theta(season_length=p))` uses StatsForecast
  Theta, not the authors' Statsmodels implementation, expanding training history,
  or precomputed scorecasts. Selecting it does not claim reproduction of the
  paper's experiments.
- **Time and continuation.** Feedback rows are consumed in ledger order,
  vectorized over nodes. The integrator clock counts valid updates per cell,
  not calendar periods or calls to step. Step and lead-time columns learn only
  when their last covered target is known, and use their own archived parameters.
  Threshold evaluation does not mutate state or retain fitted objects.
  Save the NumPy state and pending ledger together; resume with the same target,
  score, model, and configuration. Choose the scorecaster before seeding;
  changing it requires fresh initialization, since P subtracts its initial prediction.
- **Empty and unbounded sets; evaluation limitation.** With `Absolute`, a
  negative threshold gives an empty set, +inf the whole line, and -inf an empty
  set. These are intentional controller decisions, not necessarily warm-up.
  `Feedback.misses`, `Replay.covered`, and `metrics.coverage` count them correctly.
  Do not mask saturation dates or clip thresholds to make intervals look finite.
  Generic `metrics.width` and `interval_score` currently assume ordered endpoints:
  they have no empty-set convention and can return negative widths or NaN.
  `bench/compare.py` treats every infinite threshold as unready and cannot
  evaluate PID without a separately specified evaluation change.
- **Claims.** This controller makes no distribution-free
  finite-sample, conditional, or simultaneous-path coverage claim. Frequency
  control requires the controller's assumptions; it does not follow solely from
  these recursions, especially when feedback is delayed, targets overlap, or
  integral control is disabled. Choosing a scorecaster does not establish the
  assumptions or reproduce the authors' numerical results.
  Reference: Angelopoulos, Candès and Tibshirani,
  [Conformal PID Control](https://arxiv.org/abs/2307.16895).

## NexCP

- **Scope and score.** `NexCP(level, rho=0.99, window=None)` implements the split
  method in equation (11) of Barber, Candès, Ramdas and Tibshirani,
  [Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415).
  It is a `QuantileCalibrator` for scalar scores. `Absolute` gives symmetric bands;
  `Signed` gives one-sided bounds. Paired scores such as `Asymmetric` are rejected.
  This is calibration weighting, not weighted model training or the randomized
  full conformal algorithm for nonsymmetric learners.
- **Weight origin and clock.** Fix `0 < rho <= 1` before calibration. Each node
  and target column retains its own finite scores in issuing-origin order. For
  n retained scores, give score i the unnormalized weight `rho**(n+1-i)`, for
  i=1,...,n. The newest score has weight rho, not 1. Missing and nonfinite
  constructed scores are excluded and do not advance recency; calendar gaps and
  calls with no feedback do not age scores. This is retained-score time, not
  elapsed-calendar time or an extrapolation across unresolved forecast steps.
  `window=None` retains all finite scores. A positive integer keeps the last
  `window` finite scores per cell and assigns zero weight to discarded scores.
  There is no node pooling or sharing across horizons.
- **Test-point mass and rank.** Keep an additional unnormalized weight of 1 at
  +inf. With `S = sum(weights)`, select the first observed score in increasing
  score order whose cumulative weight is at least `level * (1 + S)`. Return
  +inf if no finite score reaches it. Do not interpolate, renormalize only the
  finite scores, clip an infeasible rank, or drop the test mass. Empty cells
  issue +inf. `rho=1` recovers rank `ceil((n+1)*level)` and ordinary split
  conformal for the same retained scores.
- **Readiness and precision.** A finite quantile requires
  `level <= S / (1 + S)`. For any finite geometric history with rho < 1, this
  ratio is strictly below rho. Levels at least rho therefore never become ready,
  regardless of history length; enforce that bound analytically even if floating
  accumulation rounds the limiting mass upward. Smaller windows can also prevent
  readiness. Scores and thresholds are float32, weights and accumulation float64.
  There is no epsilon or empirical-quantile fallback.
- **Initialization and state.** `seed(state, scores)` accepts already computed
  held-out scores `[K, N, C]` in chronological order, with NaN holes and no
  infinities, into an empty state. Scores must already be known at the first
  issuing origin, from the same model and score configuration. Feedback instead
  scores the archived point once the target is known. Every cell has NumPy score
  storage, a retained count and an insertion position; storage grows by doubling,
  up to window when bounded. Threshold evaluation is read-only. Save the ledger
  and calibrator arrays together and resume with the same model, score, target,
  level, rho and window. Changing them requires fresh initialization.
- **Claims.** The paper's split construction uses a predictor trained separately
  from calibration data and weights fixed independently of those data. Under the
  relevant exchangeability assumptions its marginal coverage is at least level,
  including fixed nonuniform weights. More generally, coverage is at least
  `level - Delta`, where `Delta = sum_i normalized_weight_i *
  d_TV(R(Z), R(Z_swapped_i))`. Here TV compares laws of residual vectors, not
  distances between realized residuals. It is not estimated by Calibre. The
  theorem does not require general independence or stationarity, but its bound
  can be uninformative. Under independent observations with TV drift bounded by
  `epsilon * age` on this same clock, the paper bounds Delta by
  `2 * epsilon / (1 - rho)` for rho < 1. No such drift bound is inferred from
  observed time-series scores.
  Causal online feedback alone, refits, score-dependent retention or tuning rho
  on calibration data do not establish these assumptions. Delayed or overlapping
  targets retain separate marginal columns, not conditional-on-history or
  simultaneous-path guarantees. Strong decay trades reduced influence of older
  data for less calibration mass and potentially infinite or wider intervals.

## Approved, not implemented

| Topic | Default | Consequence |
|---|---|---|
| Issuance | Issuance is the start of the first forecast period. Target completion and publication must both be strictly before it. | Origins stop being last-observed indexes. Training views respect publication times. |
| Level precision | Levels are canonical decimals at input. Ranks use exact integer or rational arithmetic. | Float32 describes score storage, not probability. Adjacent levels stay distinct. No epsilon. |
| Target support | Callers declare support. Real-valued targets are allowed. Bounds are projected conservatively onto support. | Support is not inferred from samples. Clipping points breaks hierarchy coherence. |
| Normalization | A positive, causal issuance-time scale is stored with each forecast. Zero or invalid scales need a named rule. | No division by an epsilon. No scale recomputed with later data. |
| Weighting beyond NexCP | Declare weight origin, time decay, test-point mass, and weighted rank and readiness for other weight schedules. | Other weighting does not inherit the unweighted finite-sample claim. |
| Model generations | A changed model, feature transform, or calibration method resets calibration by default. | Reset warmup stays visible. No silent transfer of old scores. |
