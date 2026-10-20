# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version can break the API.

## Unreleased

### Added

- `ConformalPID` with explicit nonnegative-score seeding, per-node/column quantile
  tracking, a saturating tangent integrator, and causal rolling AR(1) scorecasting.
  Controller state is NumPy-only and resumes through the existing online/replay API.
- Local one-step `BootstrapEnsemble` with OOB-weighted points and a `bootstrap`
  extra for scikit-learn. Fit-time residuals have explicit `FitResiduals` and
  `FittedResiduals` contracts; rolling forecasts can collect reconciled initial
  errors from one frozen model.
- `EnbPI` signed empirical residual calibration with explicit seeding, finite
  per-node/column windows, and shortest-interval beta search.

### Changed

- The `Score` contract declares its parameter shape. `Asymmetric` uses two
  residual corrections; the ledger archives them and asymmetric feedback/replay
  coverage evaluates issued endpoints rather than assuming a scalar threshold.
- `replay` accepts a seeded or saved online state for continuation.
- Removed the nonfunctional mixed-case EnbPI scaffold.

## 0.1.0

First release.
