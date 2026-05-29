# CLAUDE.md

## Project

Calibre is a demand planning engine: probabilistic forecasting + conformal
intervals + ordering policies, exercised through backtesting pipelines.

See [`README.md`](README.md) for the full architecture, all API endpoints, and
benchmark/deployment detail. This file is the quick orientation; the README is
canonical.

## Architecture

Pipeline: load dataset → build `ForecastTask`s → fit model adapters → conformal
calibration → ordering policy → ledger + metrics. Orchestrated by
`execution/backend.py::BackendEngine`.

| Module             | Responsibility                                              |
| ------------------ | ----------------------------------------------------------- |
| `core/`            | `ForecastFrame`, `ForecastTask`, metrics, logging/tracing   |
| `forecasting/`     | Adapter registry + model adapters (`features/` for transforms) |
| `conformal/`       | Conformal calibration (see Gotchas re: stable interface)    |
| `ordering/`        | Order policies + inventory `simulation/`                    |
| `execution/`       | `BackendEngine`, ledger, dataset registry, I/O, task builder |
| `evaluation/`      | Scoring and metric computation                              |
| `tuning/`          | Ray Tune + Optuna hyper-parameter search                    |
| `storage/`         | Postgres state store + Alembic `migrations/`                |
| `api/`             | FastAPI routes and schemas                                  |
| `cli/`             | CLI entrypoint, YAML config loader, commands                |

Entry points: CLI `calibre.cli.main:app`, API `calibre.api.main:app`.

## Commands

Always prefix Python tooling with `uv run`. Never invoke `python`, `pytest`,
`ruff`, or `ty` directly.

| Task                   | Command                                                   |
| ---------------------- | --------------------------------------------------------- |
| Install deps           | `uv sync --extra dev --extra benchmarks`                  |
| Run tests       | `uv run pytest`                                 |
| Run single test | `uv run pytest path/to/test_file.py::test_name` |
| Lint            | `uv run ruff check .`                           |
| Format          | `uv run ruff format .`                          |
| Type check      | `uv run ty check calibre/`                      |
| Run a backtest  | `uv run calibre run --config <cfg.yaml>` (also `validate`, `health`, `run-sweep`) |
| Serve API       | `uv run uvicorn calibre.api.main:app`           |

## Core rules

- **Simplicity first.** Smallest change that solves the problem. No
  speculative abstractions.
- **Root cause, not symptoms.** No temporary patches or `try/except` to
  silence failures.
- **Plan before non-trivial work.** 3+ steps or any architectural decision
  goes through a plan first.
- **Verify before done.** Run the relevant tests / lints / type checks and
  show they pass before marking a task complete.
- **Clean cutovers, no backwards compatibility.** Nothing is in production
  yet. Don't preserve old code paths, add compatibility shims, deprecation
  aliases, or version-gated branches. Replace the old thing outright, update
  all callers, and delete the dead code.

## Gotchas

- `conformal/` top-level exports are experimental low-level building blocks;
  the stable pipeline-facing interface is `conformal/runtime.py`.
- VN2 winning-config regression baseline is `total_cost=4992.20` — don't drift it.

## Agent memory

Long-lived project memory lives in an Obsidian vault and is governed by the
`project-memory` skill at
[`.claude/skills/project-memory/SKILL.md`](.claude/skills/project-memory/SKILL.md).
Read it at the start of any non-trivial task.

The vault location is read from the `OBSIDIAN_VAULT_PATH` environment
variable. The project folder inside the vault is `Projects/calibre/`; durable
engineering memory lives under `Projects/calibre/engineering/`
(`architecture.md`, `lessons.md`, `vision.md`, and per-task `plans/`). If the
env var is unset, skip vault operations and proceed without persistent memory.
