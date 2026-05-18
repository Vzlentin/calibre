# /goal: Determine and document the perfect execution + tuning stack for Calibre

## Context (read before proposing anything)

Calibre is a demand-planning engine that combines probabilistic forecasting (`statsforecast`, `mlforecast`, `neuralforecast`), conformal prediction intervals, and ordering policies into a single backtestable pipeline. It targets retail clients with many SKUs/series (embarrassingly parallel workloads), is cloud-native (containers, K8s, ECS, Databricks), and values elegance over speculative abstraction.

**Current state:**
- `calibre/execution/backend.py` uses `fugue.api.transform` for per-series fan-out, forcing: parquet round-trips via `ForecastTask.to_uri`/`materialize`, base64-pickled `model_config` columns, dynamic schema string construction (`_collect_quantile_columns`), and a `_run_direct` single-node bypass because Fugue overhead is wasteful at small scale.
- `calibre/tuning/optimizer.py` runs `study.optimize(_objective, n_trials=...)` sequentially. Each trial spins up its own Fugue context. No parallel HPO, no early stopping, no resource sharing.
- The two distributed frameworks (Optuna + Fugue/Dask) cannot nest, so the system serializes at the trial level. VN2 (~600 series × ~50 trials × N origins ≈ 150k fit-predicts) runs sequentially today.
- Cloud MVP (PRs #26–31, May 17–18): slim/full Docker split, FastAPI surface, SQLAlchemy + Alembic state store, GHCR publishing, ECS Terraform, fsspec IO consolidation, `RunStore`/options dataclasses.

**Non-negotiables:**
- Model adapters in `calibre/forecasting/{stats,ml,neural}forecast_adapter.py` must remain backend-blind. No distributed-framework imports in adapters.
- VN2 cost gate: `benchmarks/vn2/config/winning.yaml` must reproduce `total_cost = 4992.20` on every milestone.
- The FastAPI surface, SQLAlchemy/Alembic state store, and fsspec IO consolidation from PR #31 stay untouched.

**Not constraints:**
- No deprecation cycles required — nothing is in production.
- Image size is not a concern.
- The current stack (Fugue + Optuna) is not sacred. Full rewrites of `backend.py` and `optimizer.py` are on the table.

**Existing plan (one option among many):** Read `~/.claude/plans/i-want-to-switch-polished-ocean.md`. It proposes replacing Fugue/Dask/Spark with Ray Core + Ray Tune, making `execute()` a generator, and deleting all Fugue code. Treat this as a candidate, not a mandate. Disagree with it where warranted.

**Vault context:** Read `~/obsidian-vault/vault/Val/Wiki/_wiki/Calibre Conformal Mission.md` and `Autoreason for Calibre.md` for mission context — Calibre's thesis is making conformal prediction competitive with direct quantile for inventory decisions (VN2 Silver: < EUR 5,000 achieved; Gold: < EUR 4,831; Diamond: < EUR 4,677).

**Real use cases driving the stack:**
1. **Per-series forecasting**: 100–10,000 SKUs, each fits its own model (SeasonalNaive, local LightGBM). Embarrassingly parallel.
2. **Global forecasting**: One model across all series (global LightGBM quantile regressor). Not fan-out parallel.
3. **Hyperparameter tuning**: Panel-level sweep over lag sets, quantile alphas, tree hyperparams. Needs trial parallelism + early stopping.
4. **Conformal calibration**: Sequential per-origin `observe → apply → observe`. Mutable state (`_issued_count`). Must stay on the driver or equivalent.
5. **Inventory simulation**: Walk-forward decision loop with R,S or newsvendor policies. Cost is the only metric that matters.
6. **Cloud deployment**: Stateless containers, object-store URIs, K8s Jobs, ECS Fargate, Databricks notebooks. No persistent local disk.

## Task

Determine and document the perfect execution + tuning stack for Calibre. Do not write implementation code. Produce a written decision document that answers:

### 1. Execution layer

Propose the best approach for fanning out per-series fit+predict work across cores/nodes. You may suggest anything — Ray, Dask, Spark, ProcessPoolExecutor, multiprocessing, a custom thread pool, or something else entirely. If the best answer is "keep Fugue but fix it," say so.

For your proposal, specify:
- How local-scope (per-uid) fan-out works
- How global-scope (single model on full panel) works
- How tasks are handed to workers (pickle, URI, ObjectRef, plasma, etc.)
- How `ExecutionOptions` and engine resolution change
- Cluster/worker lifecycle: who starts it, who stops it, lifetime
- The single-node fast path for small invocations (< 10 tasks)
- Whether the current `BackendEngine.execute()` signature survives or becomes something else

### 2. Tuning layer

Propose the best approach for parallel hyperparameter search. You may suggest anything — Ray Tune, Optuna + joblib, Optuna + Dask, Hyperopt, Ax, SMAC3, or something else. If the best answer is "keep sequential Optuna but add pruning," say so.

For your proposal, specify:
- How `TuningTask.search_space: Callable[[optuna.Trial], dict]` is preserved (conditional sampling is load-bearing)
- How trial-level parallelism works
- How early stopping / pruning hooks into the per-origin evaluation loop
- How MLflow experiment tracking works (replace `safe_log_metric` / `optuna_mlflow_callback` or keep them)
- How `RunStore` (PR #31) integrates with trial persistence and resumption
- Resource budget per trial and nested parallelism prevention

### 3. Scheduler depth: how deep does the chosen framework go?

Evaluate every layer that could be added and give a **do / defer / skip** verdict. Do not limit yourself to Ray. If Dask is your execution pick, evaluate Dask Bags, Dask Delayed, Dask Distributed, etc. If Spark, evaluate SparkSQL, Structured Streaming, etc.

| Layer | What it means | Verdict | Why |
|-------|--------------|---------|-----|
| 0 | Execution framework (fan-out) | ? | ? |
| 1 | Tuning framework (HPO) | ? | ? |
| 2 | Data loading (parallel IO, column pruning) | ? | ? |
| 3 | Stateful actors (caching, shared mutable state) | ? | ? |
| 4 | Model training (distributed LightGBM/XGBoost) | ? | ? |
| 5 | Serving (replace FastAPI or colocate with execution) | ? | ? |
| 6 | State store (replace SQLAlchemy/Alembic) | ? | ? |
| 7 | Orchestration (replace CLI / ECS / K8s Jobs) | ? | ? |

Justify each verdict with Calibre-specific reasoning.

### 4. Data layer

Propose the best approach for in-memory frames, serialization, and IO:
- `fsspec` for object-store IO: keep, extend, or replace?
- `pandas` as the in-memory frame format: keep or move to Arrow / Polars / Ray Data / something else?
- Parquet as the serialization format: keep or add IPC / Plasma / something else?
- `ForecastTaskRef` (URI-based materialization): keep, extend, or replace?

### 5. Observability

Propose the best approach for monitoring and debugging:
- Dashboard: unified (Ray/Dask/Spark) or separate surfaces?
- Experiment tracking: MLflow, Weights & Biases, Ray's native tracking, or something else?
- Metrics: Prometheus, built-in framework metrics, or both?
- Logging: structured JSON logs, distributed tracing, or both?

### 6. Packaging

Propose the best approach for dependencies and deployment:
- What extras should `pyproject.toml` have post-migration?
- Slim vs full image: what goes in each?
- Databricks compatibility: does the stack need a Databricks path, and if so, how?
- Version pinning strategy for the distributed framework

### 7. Staged migration sequence

If the decision is to migrate, define phases. If the decision is to keep the current stack, define improvement phases. Each phase needs:
- Entry criteria
- Exit criteria (tests + cost gate)
- Rollback plan
- Effort estimate in working days

### 8. Risk register

List the top 5 risks with likelihood, impact, and mitigation. Must include:
- Framework setup cost per CLI invocation
- Early stopping pruning good trials because origin ordering is non-stationary
- Conditional search spaces breaking under the chosen scheduler
- Memory pressure on large panels
- Dev experience on the platforms you actually develop on

## Output format

Write the decision document as `docs/stack-decision.md` in the repo. It must be self-contained: a reader with no knowledge of this conversation can understand why every choice was made.

Also append a one-page executive summary to `docs/stack-decision-summary.md` with:
- Before / after stack comparison table
- Effort estimate and timeline
- Go / no-go recommendation

## Rules

- **Do not write implementation code.** No edits to `backend.py`, `optimizer.py`, `pyproject.toml`, or tests. Research and document only.
- **Do not run tests or benchmarks.** Analysis only.
- **Read the existing canonical plan** at `~/.claude/plans/i-want-to-switch-polished-ocean.md` before forming opinions. Cite it where you agree or disagree.
- **Read the vault** at `~/obsidian-vault/vault/Val/Wiki/_wiki/Calibre Conformal Mission.md` and `Autoreason for Calibre.md` for mission context.
- **Read the code** in `calibre/execution/backend.py`, `calibre/tuning/optimizer.py`, `calibre/tuning/task.py`, `calibre/cli/commands.py`, `calibre/core/forecast_task.py`, and `pyproject.toml` to ground decisions in actual lines.
- **Show your reasoning.** Every verdict needs a "because" clause tied to Calibre's use cases.
- **Be direct.** No filler, no hedging. If something is wrong with the canonical plan, say so plainly.
- **Propose your own ideas.** The options listed above are starting points, not a closed set. If the best stack includes something not mentioned (e.g. Prefect for orchestration, Polars for frames, Modal for serverless GPU), include it and justify it.
