# Electricity forecasting with Calibre

[`01_electricity_application_oventi.ipynb`](01_electricity_application_oventi.ipynb)
compares five forecasters and four conformal methods: **20 pairs**, with point
errors reported once per forecaster and one final five-panel Oventi figure.

## Run locally

Download the pinned source **before** running the notebook, from the repository root:

```sh
uv run --no-dev python bench/download_benchmarks.py electricity
uv sync --group notebook
uv run --group notebook jupyter lab notebooks/01_electricity_application_oventi.ipynb
```

In VS Code, select **Python Environments → `.venv/bin/python`**
(`.venv/Scripts/python.exe` on Windows), then run all cells. Both the repository
root and `notebooks/` work as launch directories. No automatic download occurs.
Tracked outputs are empty; execution and exports stay local. No private paths are printed.

## Data, forecasting and calibration

- `lstnet.load(ROOT / "data/electricity")` verifies the existing matrix: 26,304
  observations × 321 anonymous clients, without source timestamps or verified
  physical units. Displays use observation indexes, not invented dates. Quality
  and descriptive tables cover every client; previews show the first five.
- Five adjacent, disjoint 168-target test blocks each follow 720 training and
  168 calibration observations. Lead is always one hour. Historical windows
  overlap, so folds are not independent replicates.
- Training-only AIC on the first fold chooses SARIMA's `(p,1,q)`, `p,q ∈ {0,1,2}`,
  with seasonal `(1,0,0,24)`. Moving average, MSTL and SARIMA use direct
  StatsForecast vendors on expanding past windows. `HistGradientBoostingRegressor`
  stays frozen after training. Online gradient descent replays one pass per past
  prefix with fixed training-only scaling. Calibre owns the two origin loops.
- Every calibrator reuses the same point forecasts: EnbPI (residual-only, not the
  complete paper algorithm), ACI with `SplitQuantile`, Conformal PID with an
  explicit AR(1) scorecaster, and SPCI.
- SPCI uses 24 residual lags, 100 trees, depth 5, minimum leaf size 5, minimum
  20 training pairs, `ahead=1`, `window=calibration_size`, seed 42 and one job.
  Signed chronological calibration errors and issuing origins seed each fold;
  every seed target is known before first test issuance. This QRF construction
  is not an end-to-end paper reproduction or a conditional-coverage guarantee.

## Results and exports

The **point fold table comes first**: 25 rows of `family, fold, H, rmse, mae, bias`,
without a method column. Signed error is forecast minus actual. The interval table
follows: 100 rows of `family, method, fold, H, ready, coverage, width, interval_score`.
`ready` is the finite-threshold share; standard `bench/compare.py` metrics use those
cells, not every saturation outcome. Width and interval score assume ordered
endpoints ([semantics](../docs/semantics.md)). All pairs are ready throughout the
checked default run.

Separate point and interval bootstrap summaries precede the **single final figure**:
three unique point panels (RMSE, MAE, bias), then two wider method-grouped panels
(coverage, width), with a shared color-and-hatch legend. Bars are observed arithmetic
fold means; caps are actual empirical endpoints and can exclude the bar. **Mean
fold RMSE is not pooled RMSE.** A cross marks an unavailable metric interval.
Nominal **90% prediction coverage** and **95% metric intervals** are different quantities.
Diagnostics and block-length sensitivity follow the chart, with points and intervals split.

`WRITE_RESULTS = True` writes lazily to ignored
`results/electricity_notebook/<UTC>/`; reruns use new folders. Set it to `False`
to disable writes. Exports include exploration, split and SARIMA tables, figures,
`point_metrics_by_fold.csv`, `interval_metrics_by_fold.csv`, and:

- `point_bootstrap_intervals_primary.csv` / `interval_bootstrap_intervals_primary.csv`;
- `point_bootstrap_intervals.csv` / `interval_bootstrap_intervals.csv` (all five lengths);
- `point_loss_diagnostics.csv` / `interval_loss_diagnostics.csv`;
- `bootstrap_draws.npz`: all draws, including invalid ones, source losses and readiness;
- `manifest.json`: source hash/URL, configuration, bootstrap settings, scope and axes.

NPZ point draws are `[length, replicate, family, point_metric]`; interval draws are
`[length, replicate, family, method, interval_metric]`. Labels, observation steps and
fold sizes come from the typed loss record, not method order in a metrics table.
The archive records settings, empirical ranks and Monte Carlo confidence; it has
no pickled objects or duplicated point rows. No raw prediction table is exported.

## Temporal bootstrap protocol

**Conditional on recorded losses, with fixed forecasts and no refitting.** The
bootstrap neither reruns forecasting/calibration nor estimates their training
uncertainty. Its pointwise 95% intervals are sensitivity summaries, **not guaranteed
95% frequentist confidence intervals**, simultaneous intervals, or prediction intervals.

1. Concatenate all 840 test hours (35 positional 24-hour cycles). Sample circular,
   phase-aligned blocks across the **whole record**, with replacement; wrap,
   concatenate and truncate to the original length. Every family, method and
   metric shares the same sampled hours. Never resample folds independently.
   Repartition each draw into the original fold-sized slots and recompute mean
   fold metrics, taking each fold's square root **before** averaging RMSE.
2. Defaults: **10,000 replicates, 48-hour blocks, period 24, confidence 0.95,
   seed 42** (`numpy.default_rng`, PCG64). Sensitivity lengths are **24/48/72/168/336**
   hours, with the same settings. Endpoints are inverse-empirical-CDF order
   statistics at one-based ranks **250/9750**: no normal approximation, clipping,
   recentering or assumption that the observed bar lies inside the interval.
3. Keep **all source hours and all replicates**. A nonfinite source loss disables
   that metric's interval, without finite-only resampling. Any unready hour for a
   family/method marks both interval metrics `unready_source`, with no CI: a
   full-outcome interval must not accompany a ready-only comparison bar. Point
   metrics remain unaffected. Other unavailable cases retain explicit statuses.
4. Pooling the record mixes regimes and moves fold/model/calibrator resets to new
   positions. Approximate stationarity and sufficiently weak dependence are
   assumptions, **not proved here**. A 48-hour block omits much weekly dependence;
   a 336-hour block has only **2.5 block lengths** in this record. Sensitivity is
   not validation or automatic block-length selection.
5. Summaries retain the bootstrap mean, `bootstrap_bias = bootstrap_mean − estimate`,
   finite-replicate counts, statuses and metadata. Bias is diagnostic, not a
   correction. The four endpoint Monte Carlo bracket columns use 95% binomial
   order-statistic brackets: **simulation noise only, not data uncertainty**.
6. Diagnostics report raw and phase-centered ACF at lags **24/168**, statuses and
   `n_pairs`. Raw ACF uses the global mean and the full squared-deviation sum
   (the biased convention). Phase means use the whole record for diagnosis only;
   forecasts and bootstrap losses are unchanged. These noisy ACFs prove no assumptions.

## Code

`bench/` is a scripts folder, not a package. `electricity_conformal.run(...)`
returns `(point_folds, interval_folds, losses)`; `electricity_bootstrap` owns temporal
resampling and diagnostics. The notebook calls the engine once; its presentation
helper owns `describe_data`, the exact Oventi palette, plotting and lazy figure export.

References: [LSTNet](https://arxiv.org/abs/1703.07015),
[SPCI](https://arxiv.org/abs/2212.03463v3), [Calibre contracts](../docs/architecture.md).
