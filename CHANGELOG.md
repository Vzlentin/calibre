# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version can break the API.

## Unreleased

### Added

- `combine` gives one row of points: the mean of the models, weighted by the losses
  that are known at each origin.
- `Forecasts.concat` joins forecasts along models, series or origins.
- `WindowAverage` and `SeasonalNaive` are `Forecaster` models that compute all origins
  at once on the arrays of the panel.
- `rolling_forecasts` runs `MLForecast` objects. The `ml` extra installs mlforecast.

### Changed

- The backtest runs on the v2 layout of `docs/architecture.md`. `Panel` is a ragged
  grouped array, and `rolling_forecasts` runs `StatsForecast` objects and `Forecaster`
  subclasses at each origin. `Forecasts`, `reconcile`, `replay` with `Step`, `Span` and
  `Split`, and the decision functions are the public API.
- The bench loaders return the v2 `Panel`. FreshRetailNet gives its censoring mask as
  `Panel.censored`. Bike and ETTh1 have one series and no `hierarchy`.

### Removed

- The 0.1 model adapters, scores, losses, calibrators, `calibre.online`,
  scorecasting and `calibre.metrics`. Scores and diagnostics come from
  conformalforecast.
- `bench/compare.py`, `bench/vn2_conformal.py`, `bench/vn2_intervals.py` and
  `bench/selective_imagenet.py`. `bench/gate_vn2.py` runs the VN2 path.

## 0.1.0

First release.
