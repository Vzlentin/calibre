# Changelog

All notable changes to Calibre are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-03

First tagged release. Calibre is a demand-planning engine: probabilistic
forecasting, conformal prediction intervals, and ordering policies, exercised
through backtesting pipelines.

### Added

- **Forecasting** — adapter registry over `statsforecast`, `mlforecast`, and
  `neuralforecast`, with global-model fan-out (one full-panel adapter per
  distinct global config, parallelizable through Ray).
- **Conformal intervals** — per-partition conformal calibration with a stable
  pipeline-facing runtime interface and restart-safe pending observations.
- **Ordering** — newsvendor, reorder-point, and periodic-review policies over an
  inventory simulation (costs, rules, state); synthetic, snapshot, and
  client-ERP inventory adapters.
- **Execution** — backtesting pipeline orchestrated by `BackendEngine`, with
  ledger, dataset registry, task builder, and deterministic session ids.
- **API** — FastAPI service exposing the deployment path
  `/fit → /predict → /calibrate → /order → /observe`, plus promotion what-if
  prediction overrides.
- **Storage** — Postgres state store with Alembic migrations and a
  migration↔ORM parity gate.
- **Tuning** — joint model/conformal/ordering hyper-parameter search via Ray
  Tune + Optuna.
- **CLI** — `calibre run`, `validate`, `health`, and `run-sweep` over YAML
  configs.
- **Benchmarks** — VN2 inventory challenge and adaptive conformal inference
  (ACI) parity runs.
- **Project meta** — Apache-2.0 license, contributor guide, and a documented
  four-gate CI workflow.

[0.1.0]: https://github.com/Vzlentin/calibre/releases/tag/v0.1.0
