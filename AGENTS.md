# AGENTS.md

## Project

Calibre is a demand planning engine: probabilistic forecasting + conformal
intervals + ordering policies, exercised through backtesting pipelines.

See [`docs/spec/00-overview.md`](docs/spec/00-overview.md) for normative
successor design and [`newcalibre/README.md`](newcalibre/README.md) for the
implemented successor surface. The root [`README.md`](README.md) also documents
the frozen predecessor while both trees coexist.

## Architecture

Successor pipeline: load canonical domain inputs → resolve admissible actuals →
fit/predict → point reconciliation → conformal calibration → order → settle →
commit to the ledger. The I/O-free engine under
`newcalibre/src/newcalibre/engine/` is driven by both time-loop and event
frontiers.

| Successor module | Responsibility |
| --- | --- |
| `domain/` | Canonical identities, panels, tasks, frames, decisions, costs, and evidence |
| `engine/` | Closed verb surface, drivers, ports, ordering, settlement, and commit |
| `forecasting/` | Forecast adapter protocol, registry, and model adapters |
| `reconcile/` | Point-reconciliation protocol, strategies, summing structures, and preflight |
| `conformal/` | Method manifests, runtime, state, and registered families |
| `observe/` | Actual acceptance, pending-window resolution, delivery, and state updates |
| `ordering/` | Cost objectives and ordering-policy families |
| `protocols/` | External protocol bindings and evidence artifacts |
| `oracle/` | Manifest-checked behavior capture support used only before cutover |

The predecessor package under `calibre/` is the frozen oracle only at tag
`oracle-freeze-2026-07-06`. Later root-tree changes do not redefine the oracle.
Its CLI and API remain coexistence surfaces, not successor architecture.

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

Long-lived project memory is governed by the user-level `project-memory` skill.
Resolve `$OBSIDIAN_VAULT_PATH` from the environment or this repository's `.env`;
an empty shell variable alone does not prove the vault is unavailable.

In vault mode, start at `Projects/Calibre/index.md`. Read
`Projects/Calibre/CONTEXT.md` for domain language, follow only the relevant
entry under `Projects/Calibre/architecture/`, and read the matching active plan
under `Projects/Calibre/plans/`. Do not bulk-load the bundle or assume legacy
monolithic canon, solution, lesson, or deferred-register files are operational.

Authority is split deliberately:

- `docs/spec/` is normative successor design.
- Successor source is implementation fact.
- The vault architecture bundle is the navigational synthesis, including known
  deltas and private rationale.
- The `oracle-freeze-2026-07-06` tag is the frozen oracle; do not infer its
  structure from later root-engine changes.

`CONTEXT.md` and `docs/plans/README.md` are public-safe redirectors.
`docs/adr/README.md` points to `docs/spec/adr/`, the sole successor ADR series.
If the vault is unavailable, proceed with repository sources and public-safe
plan fallback; do not write private durable memory into the repository.

Public `[ANNEX:*]` references remain opaque contracts. Never resolve them,
identify their private storage, or inline their content. Any `docs/spec/` edit
requires an owner leak-review stamp on its landing issue before merge; see
`docs/agents/domain.md`.

### Program status

GitHub milestones, issues, and dependencies are the live status and work-order
surface. Do not mirror issue state into durable project-memory concepts.

## Agent skills

### Shipping workflow (/go)

The implementation-orchestration pipeline (`/go`) and `project-memory` are
**user-level** skills from the public `Vzlentin/dotfiles` repo
(`~/.agents/skills/`); they are no longer tracked in this repo. There is no
per-project config layer: the skill discovers Calibre's quality gates
(ruff / ty / pytest, all `uv run`-prefixed) from this file's *Commands*
section directly.

### Issue tracker

GitHub issues in `Vzlentin/calibre` are the live status and work-order surface;
external PRs are a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles → `needs-triage`, `needs-info`, `ready-for-agent`,
`needs-decision` (the `ready-for-human` role, reusing the existing label),
`wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Start project-memory discovery at the vault `Projects/Calibre/index.md`; the
glossary is `Projects/Calibre/CONTEXT.md`, modular architecture is under
`Projects/Calibre/architecture/`, and active plans are under
`Projects/Calibre/plans/`. Successor ADRs live only in `docs/spec/adr/`;
`CONTEXT.md`, `docs/adr/`, and `docs/plans/` are discovery redirectors. See
`docs/agents/domain.md`.
