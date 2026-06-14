# CLAUDE.md

## Project

Calibre is a demand planning engine: probabilistic forecasting + conformal
intervals + ordering policies, exercised through backtesting pipelines.

See [`README.md`](README.md) for the full architecture, all API endpoints, and
benchmark/deployment detail. This file is the quick orientation; the README is
canonical.

## Architecture

Pipeline: load dataset → build `ForecastTask`s → fit model adapters →
reconciliation (when hierarchical) → conformal calibration → ordering policy →
ledger + metrics. Orchestrated by `execution/backend.py::BackendEngine`.

| Module             | Responsibility                                              |
| ------------------ | ----------------------------------------------------------- |
| `core/`            | `ForecastFrame`, `ForecastTask`, metrics, logging/tracing   |
| `forecasting/`     | Adapter registry + model adapters (`features/` for transforms) |
| `reconciliation/`  | `Reconciler` protocol + strategy registry + summing matrix (point reconcile between predict and calibrate) |
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

## Commenting & docstring convention

- **House style.** Imperative one-line summary ending in `.`/`?`/`!`. Add
  Google `Args:`/`Returns:`/`Raises:` sections only where a parameter or return
  needs explanation. Cross-reference siblings with reST `:class:`/`:func:`/
  `:meth:`. No NumPy-style `Parameters:`; no Sphinx/mkdocs build — docstrings are
  read in source.
- **Coverage bar.** Every public module, package, class, and function carries a
  docstring. Methods are documented by convention (not gated). Private helpers
  get one only when the intent is non-obvious.
- **Comments explain *why*, not *what*.** Drop redundant, duplicated, or stale
  comments. **No private references** in shipped prose — no vault pointers
  (`lessons.md`), dead roadmap phases (`P0.3`), plan/review thread IDs
  (`FIX #N`, `REVIEW #N`), or bare `#N`. Inline the load-bearing substance in
  1–2 lines; keep a public anchor (issue/PR/code path) only when it helps an
  outsider.
- **Enforced subset (ruff `D`).** `D100`, `D104` (module + package), `D101`,
  `D103` (class + function), `D205`/`D212`/`D415` (summary format). `D102`
  (methods) and `D107` (`__init__`) are convention-only, not gated. `tests/**`
  carries only `D100` — `D101`/`D103`/`D104` are exempt. `migrations/versions/`
  is exempt from all D rules (per-file-ignores); `scripts/databricks_notebook.py`
  is excluded from ruff entirely.

## Gotchas

- `conformal/` top-level exports are experimental low-level building blocks;
  the stable pipeline-facing interface is `conformal/runtime.py`.
- VN2 winning-config regression baseline is `total_cost=4992.20` — don't drift it.
  This is the **x86_64/Linux CI** value, which is where the gate runs (no test
  asserts the literal number). On arm64/macOS the same config deterministically
  produces **~5011.20** — cross-arch LightGBM float divergence (SIMD/FMA/libm) plus
  Accelerate-vs-OpenBLAS, **not** a regression and **not** threading (single- and
  multi-thread agree bit-for-bit). Don't chase the macOS delta or loosen 4992.20.
- Reconciliation strategy is an M5-coverage lever, not coverage-neutral: `wls_struct`
  lands population coverage on-target (~90.97%) where `bottom_up` over-covers (~94.92%).
  Weigh the reconciler choice, not just conformal knobs, on a coverage miss.

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

Long-lived project memory lives in an Obsidian vault and is governed by
`.claude/skills/project-memory/SKILL.md`. Read it at the start of any
non-trivial task. Hermes agents: use the `obsidian` skill instead.

The vault root is `$OBSIDIAN_VAULT_PATH`. The project folder is
`Projects/Calibre/`; canon files: `architecture.md`, `lessons.md`,
`vision.md`, `ROADMAP.md`, plus `plans/` and `archive/`. Reusable per-problem
learnings live in `Projects/Calibre/solutions/` (knowledge-track docs by category —
architecture-patterns, design-patterns, conventions, performance-issues, workflow,
… — with `module`/`tags`/`applies_when` frontmatter), the shared domain vocabulary
in `Projects/Calibre/CONCEPTS.md`, and postponed work in
`Projects/Calibre/deferred-findings-register.md`; consult them when implementing or
debugging in a documented area. If the env var is unset, plans fall back to
repo-local `docs/plans/` while durable memory is skipped (per the skill).

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
