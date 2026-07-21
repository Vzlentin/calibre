# AGENTS.md

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

The architecture spec for the greenfield rewrite lives at `docs/spec/`
(start at `docs/spec/00-overview.md`). It describes the successor engine,
not this codebase; this repo remains the behavior reference until cutover.
See *Agent memory* below for the editing rule that governs it.

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

- **House style.** Imperative one-line summary ending in `.`/`?`/`!`. Add Google
  `Args:`/`Returns:`/`Raises:` only where a parameter or return needs it.
  Cross-reference siblings with reST `:class:`/`:func:`/`:meth:`. No NumPy-style
  `Parameters:` — docstrings are read in source, not built by Sphinx/mkdocs.
- **Coverage bar.** Every public module, package, class, and function carries a
  docstring. Methods by convention (not gated). Private helpers only when the
  intent is non-obvious.
- **Comments explain *why*, not *what*.** Drop redundant, duplicated, or stale
  comments. **No private references** in shipped prose — no vault pointers, dead
  roadmap phases (`P0.3`), or plan/review thread IDs (`FIX #N`, `REVIEW #N`, bare
  `#N`). Inline the load-bearing substance in 1–2 lines;
  keep a public anchor (issue/PR/code path) only when it helps an outsider.
- **Enforced subset (ruff `D`).** Gated: `D100`/`D104` (module + package),
  `D101`/`D103` (class + function), `D205`/`D212`/`D415` (summary format). `D102`
  (methods) and `D107` (`__init__`) are convention-only. Per-file-ignores (in
  `pyproject.toml`) relax this: `tests/**` keeps `D100` + summary format,
  `migrations/versions/` is all-`D`-off, `scripts/databricks_notebook.py` is out
  of ruff entirely.

## Gotchas

- `conformal/` top-level exports are experimental low-level building blocks;
  the stable pipeline-facing interface is `conformal/runtime.py`.
- VN2 winning-config regression baseline is `total_cost=4992.20` — don't drift it.
  This is the **x86_64/Linux CI** value, where the gate runs:
  `tests/benchmarks/test_vn2_regression.py` hard-asserts it (`BASELINE_TOTAL_COST =
  4992.20`, holding `2488.20` / shortage `2504.00`, at `abs_tol 0.01`) on both model
  paths — `regression`-marked and skipped off x86_64 via `_x86_64_gate`. On
  arm64/macOS the same config deterministically produces **~5011.20** — cross-arch
  LightGBM float divergence (SIMD/FMA/libm) plus Accelerate-vs-OpenBLAS, **not** a
  regression and **not** threading (single- and multi-thread agree bit-for-bit).
  Don't chase the macOS delta or loosen 4992.20.
- Reconciliation strategy moves M5 sales-coverage diagnostics: `wls_struct`
  produced ~90.97% where `bottom_up` produced ~94.92% in one full-scale run.
  Always report the reconciler, but never tune or select the Gate C configuration
  by closeness to the diagnostic target; select it for architectural
  representativeness and scale.

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
**user-level** `project-memory` skill (`~/.agents/skills/project-memory/SKILL.md`,
from the public `Vzlentin/dotfiles` repo). Read it at the start of any
non-trivial task. Hermes agents: use the `obsidian` skill instead.

The vault root is `$OBSIDIAN_VAULT_PATH` — but **on this machine that variable
is defined in the repo's `.env`, which is not auto-exported into your shell.** A
bare `echo $OBSIDIAN_VAULT_PATH` (or `$env:OBSIDIAN_VAULT_PATH`) reads empty and
will fool you into thinking the vault is absent and degrading to fallback mode.
**Resolve the path from `.env` first** (e.g. `grep OBSIDIAN_VAULT_PATH .env`, or
source `.env`) and use that value; treat the vault as truly unavailable only if
`.env` has no such line. The project folder is
`Projects/Calibre/`; canon files: `architecture.md`, `vision.md`, `ROADMAP.md`, plus
`plans/` and `archive/`. Reusable per-problem learnings live in
`Projects/Calibre/solutions/` (knowledge-track docs by category —
architecture-patterns, design-patterns, conventions, performance-issues, workflow,
… — with `module`/`tags`/`applies_when` frontmatter), the shared domain vocabulary
in `Projects/Calibre/CONCEPTS.md`, and postponed work in
`Projects/Calibre/deferred-findings-register.md`. Read only the smallest relevant
solution notes when implementing or debugging; never bulk-load the store. If `.env`
carries no `OBSIDIAN_VAULT_PATH` line (truly unset — not merely absent from the live
shell), durable memory is skipped
(per the skill). `docs/plans/`, `docs/adr/`, and `CONTEXT.md` are public-safe
redirectors to the vault (see `docs/agents/domain.md`), not artifact stores — a
skill may write a temporary local plan under `docs/plans/` if the vault is
unreachable, then relocate it on vault return. **`docs/spec/` is the one
deliberate exception**: the repo's single durable public artifact store,
holding the public layer of the two-layer rewrite spec (private rationale
stays in the vault behind opaque `[ANNEX:*]` pointers — never resolve or
inline them in the repo). **Leak review before edit:** any change under
`docs/spec/` requires an owner leak-review stamp on the landing's tracking
issue *before* it lands; reviews are batched per landing, not per file (see
`docs/agents/domain.md`).

### Roadmap: GitHub for status, vault for rationale

Hybrid, one source of truth per fact-type — don't mirror one into the other:

- **Live status + work orders = GitHub.** Each backlog item is an issue whose
  body holds the full symptom/fix/files spec; a merged `closes #N` PR updates
  status for free. The active milestone is named in `ROADMAP.md` (vault) — don't
  hard-code it here (a renamed milestone is the drift #89 fixed). List open
  milestones and their issues:

  ```bash
  gh api repos/Vzlentin/calibre/milestones --jq '.[] | select(.state=="open") | .title'
  gh issue list --milestone "<title>"
  ```

  Parked items carry `parked:phd` / `parked:saas` and sit outside any milestone.
- **Durable rationale = `ROADMAP.md`** (vault): mission, how-we-work
  cadence/gates, root-issue analysis (R1–R5), dependency ordering, parked
  decisions. Read it for the *why*; it carries **no** issue-status checklist.

## Agent skills

### Shipping workflow (/go)

The implementation-orchestration pipeline (`/go`) and `project-memory` are
**user-level** skills from the public `Vzlentin/dotfiles` repo
(`~/.agents/skills/`); they are no longer tracked in this repo. There is no
per-project config layer: the skill discovers Calibre's quality gates
(ruff / ty / pytest, all `uv run`-prefixed) from this file's *Commands*
section directly.

### Issue tracker

GitHub issues in `Vzlentin/calibre` — hybrid with the vault `ROADMAP.md` for
rationale (status on GitHub, rationale in the vault); external PRs are a
triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles → `needs-triage`, `needs-info`, `ready-for-agent`,
`needs-decision` (the `ready-for-human` role, reusing the existing label),
`wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Domain docs live in the Obsidian vault, not the repo: glossary at
`Projects/Calibre/CONCEPTS.md`, ADRs at `Projects/Calibre/adr/`, plans at
`Projects/Calibre/plans/`. Repo `CONTEXT.md`, `docs/adr/`, `docs/plans/` are
public-safe redirectors. See `docs/agents/domain.md`.
