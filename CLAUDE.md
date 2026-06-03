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
  This is the **x86_64/Linux CI** value, which is where the gate runs (no test
  asserts the literal number). On arm64/macOS the same config deterministically
  produces **~5011.20** — cross-arch LightGBM float divergence (SIMD/FMA/libm) plus
  Accelerate-vs-OpenBLAS, **not** a regression and **not** threading (single- and
  multi-thread agree bit-for-bit). Don't chase the macOS delta or loosen 4992.20.

## Worktrees

New worktrees auto-run `setup-worktree-unix` from `.cursor/worktrees.json`
(copies `.env` + `data/vn2`, then `uv sync --frozen`). For fast, correct setup:

- **uv's package cache is global and shared across worktrees**, so a *warm*
  `uv sync` rebuilds the full ~1.1G venv in a couple of seconds. The only heavy
  cost is the one-time *cold* download (~0.5–1G of wheels; `ray` alone is 200M).
  Don't copy/seed the main `.venv` — it's non-relocatable (absolute paths in
  `pyvenv.cfg`/scripts + an editable `.pth` pinned to the main checkout).
- **Don't set `UV_LINK_MODE`.** The default APFS clone is fastest (~1.5s);
  `hardlink`/`copy` are ~5x slower (~8s) here.
- **Don't run `pytest`/`--collect-only` during setup** — it's not needed to
  provision the env and costs ~30s of first-run bytecode compilation.
- **Use an explicit `cd <worktree> && …`** for shell commands targeting a
  worktree; a `working_directory` arg may not change cwd, silently running setup
  against the main repo and duplicating the work.

## Agent memory

Long-lived project memory lives in an Obsidian vault and is governed by the
`project-memory` skill at
[`.claude/skills/project-memory/SKILL.md`](.claude/skills/project-memory/SKILL.md).
Read it at the start of any non-trivial task.

The vault location is read from the `OBSIDIAN_VAULT_PATH` environment
variable. The project folder inside the vault is `Projects/Calibre/`; the canon
files live at its root (`architecture.md`, `lessons.md`, `vision.md`,
`ROADMAP.md`), alongside per-task `plans/`, an `archive/` for superseded notes,
and a `phd/` research track. If the env var is unset, skip vault operations and
proceed without persistent memory.

### Roadmap: GitHub for status, vault for rationale

The roadmap is a hybrid with one source of truth per fact-type — don't mirror one
into the other:

- **Live status + work orders = GitHub.** Each backlog item is an issue whose
  body holds the full symptom/fix/files spec, and a merged `closes #N` PR updates
  status for free. The current active milestone is named in `ROADMAP.md` (vault) —
  don't hard-code it here; a renamed milestone is exactly the drift #89 fixed.
  List open milestones and pull their issues with:

  ```bash
  gh api repos/Vzlentin/calibre/milestones --jq '.[] | select(.state=="open") | .title'
  gh issue list --milestone "<title>"
  ```

  Parked items carry `parked:phd` / `parked:saas` and are out of any milestone.
- **Durable rationale = `ROADMAP.md`** (vault): mission, how-we-work
  cadence/gates, root-issue analysis (R1–R5), dependency ordering, and parked
  decisions. Read it for the *why*; it deliberately carries **no** issue-status
  checklist.
