---
title: "feat: Committed cumulative-mode benchmark configs (M5 + VN2)"
type: feat
status: active
date: 2026-06-14
origin: "GitHub issue #198 — Add a committed cumulative-mode benchmark config"
---

# feat: Committed cumulative-mode benchmark configs (M5 + VN2)

## Summary

The engine-internal cumulative conformal mode — `SymmetricIntervalRuntime._apply_cumulative` /
`_observe_cumulative`, reachable from a CLI YAML config via `conformal.mode: cumulative` —
has **no committed benchmark config**. Its at-scale constant factor (a per-`(uid, model, origin)`
Python `groupby` loop with per-group `.loc` assignment, i.e. O(batch)) is therefore unmeasured by
the committed M5/VN2 gates. It was only probe-verified at M5 shape during AD2 (DF-26, PR #178 residual).

This plan commits cumulative-mode benchmark configs for **both** datasets so the path is measured:
- **M5** — adds a full-scale cumulative config (parse-asserted in CI, run locally via the M5 runbook
  for the real at-scale number) plus a fixture-backed smoke config that **executes the cumulative
  code path in CI**.
- **VN2** — adds the analogous pair, exercising the same engine-internal cumulative conformal path
  through `calibre run --config` at VN2 shape.

No new abstractions. The committed configs reuse the existing `ConformalConfig` schema
(`calibre/cli/config.py`) and the existing benchmark-gate pattern (`tests/benchmarks/test_m5_config.py`):
load each config, assert its fields, and execute the smoke variant end-to-end via `run_config()`.

---

## Problem Frame

- **What's missing:** A committed YAML config that selects `conformal.mode: cumulative` and is wired
  into the benchmark gates. Today the only committed conformal configs are `perhorizon`
  (`benchmarks/m5/config/full.yaml`, `full-wls-struct.yaml`).
- **Why it matters:** The cumulative `apply`/`observe` path is structurally different from the
  vectorized perhorizon path — it is a row-wise grouped loop. Without a committed config, no gate
  exercises it, so a regression in its correctness or its O(batch) constant factor would go unnoticed
  until a manual probe.
- **Scope of "cumulative" here:** This is specifically the **engine-internal** cumulative conformal
  mode in `calibre/conformal/runtime.py`. It is distinct from the VN2 HPO harness's order-conformal
  path (`calibre/conformal/cumulative_risk.py::CumulativeRiskRuntime`, driven by
  `benchmarks/vn2/run_benchmark.py`), which already has test coverage and is **out of scope**.

---

## Key Technical Decisions

### KTD1 — Both committed configs drive the same engine-internal path via `calibre run --config`

`conformal.mode: cumulative` flows: YAML → `ConformalConfig` → `ConformalConfig.to_runtime_config()`
→ `SymmetricIntervalConfig(mode="cumulative", …)` → `SymmetricIntervalRuntime._apply_cumulative`.
This entrypoint is dataset-agnostic, so a VN2 config and an M5 config exercise the **same** code path
at two different batch shapes. This is the minimal, consistent way to commit the mode for both
datasets without touching the VN2 HPO harness.

### KTD2 — Cumulative mode constraints are enforced by `SymmetricIntervalConfig.__post_init__`

Cumulative mode requires `method: mscp` and `protection_period >= 1`. Both committed configs must set
these. `protection_period` must be `<= horizon` for a terminal-H row to exist (the runtime writes the
bound only on the row where `H == protection_period`).

- **M5 full:** `protection_period: 28` (one cumulative bound over the full 28-day horizon — the natural
  "cumulative over the protection window" choice; the O(batch) loop cost is independent of this value).
- **VN2:** `protection_period: 3` (= lead_time 2 + review_period 1, matching the VN2 domain constant
  `HORIZON` in `benchmarks/vn2/config.py`).
- **Smoke variants:** `protection_period` set to the smoke horizon (`>= 2`) so the cumulative window
  and terminal-row logic actually run.

### KTD3 — The committed full config is NOT a coverage gate

Cumulative mode emits a single terminal-H bound per `(uid, model, origin)`, not per-horizon intervals,
so the M5 population-coverage gate (`score-m5-coverage`, the 90±3% statistical acceptance on
`full.yaml`/`full-wls-struct.yaml`) does not apply to it. The cumulative full config exists to **measure
the engine-internal cumulative path** (correctness + at-scale constant factor via the runbook), not to
assert coverage. The plan must not add the cumulative config to
`test_m5_full_origin_window_meets_mscp_horizon_invariant` (that invariant is about per-horizon quantile
readiness) and must not alter the VN2 `total_cost=4992.20` baseline — the cumulative configs are new,
separate files; `winning.yaml` is untouched.

### KTD4 — Smoke configs are the CI measurement surface; fixture/horizon may need a small adjustment

Becoming "measured in CI" means a smoke variant executes via `run_config()` against the committed
fixtures (`tests/fixtures/m5`, the VN2 fixture). The current M5 smoke runs `horizon: 1`; a cumulative
smoke needs `horizon >= protection_period >= 2`. Whether the existing fixture has enough history for a
2-step SeasonalNaive forecast is an execution-time check (see U2 verification). The VN2 smoke already
runs `horizon: 2`, so `protection_period: 2` fits its fixture without change.

---

## Implementation Units

### U1. M5 committed full cumulative config + parse-assert gate

**Goal:** Commit `benchmarks/m5/config/full-cumulative.yaml` and assert its fields parse, mirroring how
`full.yaml` / `full-wls-struct.yaml` are gated.

**Requirements:** Issue #198 (M5 half).

**Dependencies:** none.

**Files:**
- `benchmarks/m5/config/full-cumulative.yaml` (new)
- `tests/benchmarks/test_m5_config.py` (add parse-assert test)
- `benchmarks/m5/README.md` (document the new config under "Configs")

**Approach:**
- Base the config on `full.yaml`: same `dataset` (adapter `m5`, `path: data/m5`, `phase: evaluation`),
  same `tasks` (SeasonalNaive, `horizon: 28`, `scope: global`), same `reconciliation: bottom_up`,
  same `origins` window (64 daily origins).
- Conformal block: `method: mscp`, `mode: cumulative`, `coverage: 0.9`, `calibration_window: 10`,
  `partition: series`, `protection_period: 28`, `max_partitions: 1000000`.
- Distinct `output.ledger_path` (e.g. `results/m5/full-mscp-cumulative/forecast-ledger.parquet`),
  `streaming: true`, `execution.backend: auto`, `seed: 42`.
- README: add a "Configs" bullet describing `full-cumulative.yaml` as a local at-scale measurement run
  for the engine-internal cumulative conformal path (not a coverage-acceptance config). **Do not remove
  or reword** the phrases asserted by `test_m5_runbook_keeps_fixture_out_of_statistical_acceptance`.

**Patterns to follow:** `benchmarks/m5/config/full-wls-struct.yaml` (a sibling variant of `full.yaml`);
the field-assert tests `test_m5_full_config_uses_series_partition_handoff` and
`test_m5_full_wls_struct_config_uses_series_partition_handoff`.

**Test scenarios:**
- `load_config(full-cumulative.yaml)` parses and `config.conformal.method == "mscp"`,
  `config.conformal.mode == "cumulative"`, `config.conformal.protection_period == 28`,
  `config.conformal.partition == "series"`, `config.conformal.coverage == 0.9`,
  `config.conformal.calibration_window == 10`, `config.conformal.max_partitions == 1_000_000`.
- `config.reconciliation.strategy == "bottom_up"`, `config.tasks[0].horizon == 28`,
  `config.output.ledger_path` is the cumulative ledger path, `config.execution.backend == "auto"`.
- `config.conformal.to_runtime_config()` returns a `SymmetricIntervalConfig` that constructs without
  raising (proves the cumulative `mscp` + `protection_period` invariant is satisfied).

**Verification:** `uv run pytest tests/benchmarks/test_m5_config.py` passes; new parse-assert test green;
the README runbook test still green.

---

### U2. M5 cumulative smoke config + CI end-to-end execution

**Goal:** Commit a fixture-backed M5 smoke config that **runs the cumulative path in CI** via
`run_config()`, so `_apply_cumulative` / `_observe_cumulative` execute on every CI run.

**Requirements:** Issue #198 (M5 half, "so it is measured").

**Dependencies:** U1 (shares the conformal field shape).

**Files:**
- `benchmarks/m5/config/smoke-cumulative.yaml` (new)
- `tests/fixtures/m5/` (adjust only if the fixture lacks history for `horizon >= 2` — see verification)
- `tests/benchmarks/test_m5_config.py` (add parse-assert + execute test)

**Approach:**
- Base on `benchmarks/m5/config/smoke.yaml` (dataset `tests/fixtures/m5`, `phase: evaluation`,
  `execution.backend: local`, `seed: 42`), but set `horizon: 2`, `season_length: 2` (or a value the
  fixture supports), and a conformal block with `method: mscp`, `mode: cumulative`,
  `protection_period: 2`, `coverage: 0.9`, small `calibration_window`.
- Distinct `output.ledger_path` (e.g. `results/m5/smoke-cumulative/forecast-ledger.parquet`),
  `streaming: false`.
- Keep the existing `smoke.yaml` and its test untouched (clean separation; no back-compat shim needed).

**Patterns to follow:** `test_m5_smoke_config_executes_source_run_path` — copy its `run_config()` +
`tmp_path` ledger-redirect structure.

**Test scenarios:**
- Parse: `load_config(smoke-cumulative.yaml)` → `config.conformal.mode == "cumulative"`,
  `config.conformal.protection_period == 2`, `config.tasks[0].horizon == 2`.
- Execute (the load-bearing one): `run_config()` with a `tmp_path` ledger override produces a non-empty
  ledger; the ledger file is written; the cumulative interval columns are populated **only on the
  terminal-H row** (`H == protection_period`) for each `(uid, model, origin)` group, and NaN on
  earlier-H rows — directly asserting `_apply_cumulative`'s terminal-row contract.
- `conformal_mode` column on the resolved ledger equals `"cumulative"` (proves the cumulative branch,
  not perhorizon, executed).
- Edge: a group with fewer than `protection_period` resolved horizons yields no bound (NaN), matching
  the runtime's window-completeness rule.

**Execution note:** Start by confirming the fixture supports a 2-step horizon (run the new smoke locally);
only extend `tests/fixtures/m5` history if `run_config()` fails for lack of data — keep any fixture
change minimal.

**Verification:** `uv run pytest tests/benchmarks/test_m5_config.py` passes including the new execute
test; `uv run calibre run --config benchmarks/m5/config/smoke-cumulative.yaml` produces a ledger with
cumulative bounds on terminal-H rows.

---

### U3. VN2 committed cumulative config + parse-assert gate

**Goal:** Commit a VN2 cumulative config that drives the engine-internal cumulative conformal path via
`calibre run --config`, plus a parse-assert test.

**Requirements:** Issue #198 (VN2 half).

**Dependencies:** none (independent of U1/U2; shares the conformal shape).

**Files:**
- `benchmarks/vn2/config/cumulative.yaml` (new)
- `tests/benchmarks/test_vn2_benchmark.py` (or a sibling config test — see Approach) — add parse-assert
- `benchmarks/vn2/README.md` if present (document the new config; create no new docs file otherwise)

**Approach:**
- Base on `benchmarks/vn2/config/winning.yaml` (adapter `vn2`, `path: data/vn2`, `period: 8`,
  `origins` window, `seed: 42`) but add a conformal block: `method: mscp`, `mode: cumulative`,
  `protection_period: 3`, `coverage: 0.9`, small `calibration_window`. Keep `horizon: 3`.
- Distinct `output.ledger_path` (e.g. `results/vn2/cumulative-ledger.parquet`). **Do not modify
  `winning.yaml`** — the `total_cost=4992.20` regression baseline must stay bit-identical.
- Placement of the parse-assert test: `tests/benchmarks/test_vn2_benchmark.py` imports from the VN2
  harness; if a lighter-weight config-only test module is cleaner (mirroring `test_m5_config.py`),
  create `tests/benchmarks/test_vn2_config.py` for the parse + execute asserts. Implementer's call at
  execution time; either keeps the gate config-driven.

**Patterns to follow:** `tests/benchmarks/test_m5_config.py` field-assert structure; `winning.yaml`
shape; `load_config` from `calibre/cli/config.py`.

**Test scenarios:**
- `load_config(cumulative.yaml)` parses; `config.conformal.method == "mscp"`,
  `config.conformal.mode == "cumulative"`, `config.conformal.protection_period == 3`,
  `config.tasks[0].horizon == 3`, `config.dataset.adapter == "vn2"`.
- `config.conformal.to_runtime_config()` constructs a `SymmetricIntervalConfig` without raising.
- Guard test (cheap regression insurance): `winning.yaml` still has **no** `conformal` section / is
  unchanged — assert its loaded shape is untouched so the cumulative addition can't silently perturb
  the baseline config.

**Verification:** `uv run pytest tests/benchmarks/` passes; new VN2 parse-assert test green; VN2
regression test (`test_vn2_regression.py`) still green and the 4992.20 baseline untouched.

---

### U4. VN2 cumulative smoke config + CI end-to-end execution

**Goal:** Commit a fixture-backed VN2 smoke config that runs the cumulative path in CI via
`run_config()`.

**Requirements:** Issue #198 (VN2 half, "so it is measured").

**Dependencies:** U3.

**Files:**
- `benchmarks/vn2/config/smoke-cumulative.yaml` (new)
- `tests/benchmarks/test_vn2_config.py` (or `test_vn2_benchmark.py`) — add execute test

**Approach:**
- Base on `benchmarks/vn2/config/smoke.yaml` (adapter `vn2`, `path: benchmarks/vn2/fixture`,
  `period: 0`, SeasonalNaive, `horizon: 2`, `season_length: 2`, `execution.backend: local`,
  `seed: 42`). Add a conformal block: `method: mscp`, `mode: cumulative`, `protection_period: 2`,
  `coverage: 0.9`, small `calibration_window`.
- Distinct `output.ledger_path` (e.g. `results/vn2/smoke-cumulative-ledger.parquet`). The VN2 smoke
  fixture already supports `horizon: 2`, so no fixture change is expected.

**Patterns to follow:** `test_m5_smoke_config_executes_source_run_path` (run_config + tmp_path);
existing VN2 smoke usage in the benchmark tests.

**Test scenarios:**
- Parse: `config.conformal.mode == "cumulative"`, `protection_period == 2`, `horizon == 2`.
- Execute: `run_config()` with a `tmp_path` ledger override produces a non-empty ledger; cumulative
  interval columns are populated only on the terminal-H row (`H == 2`) per group; `conformal_mode`
  column equals `"cumulative"`.
- Edge: incomplete window (group with < `protection_period` resolved horizons) → no bound (NaN).

**Verification:** `uv run pytest tests/benchmarks/` passes including the new VN2 execute test;
`uv run calibre run --config benchmarks/vn2/config/smoke-cumulative.yaml` produces a ledger with
cumulative bounds on terminal-H rows.

---

## Scope Boundaries

**In scope:**
- Four new committed YAML configs (M5 full + smoke, VN2 full + smoke) selecting `conformal.mode: cumulative`.
- Parse-assert tests for the full configs; parse-assert + CI execute tests for the smoke configs.
- README documentation for the new M5 config (and VN2 README if one exists).
- Minimal `tests/fixtures/m5` adjustment **only if** the cumulative smoke cannot run on the current fixture.

**Out of scope (non-goals):**
- The VN2 HPO order-conformal path (`CumulativeRiskRuntime` in `calibre/conformal/cumulative_risk.py`,
  driven by `benchmarks/vn2/run_benchmark.py`) — already tested; not the engine-internal mode #198 targets.
- Any change to `winning.yaml`, the `total_cost=4992.20` baseline, or the M5 population-coverage gate.
- Optimizing the O(batch) cumulative loop. This plan **measures** the constant factor by committing a
  config that runs the path; it does not change the path.
- Adding the cumulative config to `test_m5_full_origin_window_meets_mscp_horizon_invariant` (per-horizon
  readiness invariant; not applicable to a single terminal-H bound).

### Deferred to Follow-Up Work
- A perf assertion / timing budget on the cumulative path's constant factor (this plan makes it *run* in
  CI and *measurable* locally; a quantitative budget is a separate hardening item if AD2 wants one).

---

## Risks & Dependencies

- **R1 — M5 smoke fixture may lack history for a 2-step horizon.** Mitigation: U2 verifies by running
  the smoke locally first; extend the fixture minimally only if needed. Low likelihood (SeasonalNaive
  with `season_length: 2` needs little history).
- **R2 — VN2 cumulative config via `calibre run --config` might require an order policy to produce a
  meaningful ledger.** The conformal `apply` itself does not require ordering (it only emits interval
  columns), so a forecast+conformal run is sufficient to exercise the path. If the VN2 CLI run path
  asserts an order-ledger output, mirror `winning.yaml`'s `order_ledger_path` without an active policy,
  or omit it. Resolve at execution time in U3/U4.
- **R3 — Accidentally perturbing the 4992.20 baseline.** Mitigation: new files only; U3 adds a guard
  test asserting `winning.yaml` is unchanged; the VN2 regression test stays green.

---

## Verification Strategy

Per CLAUDE.md ("verify before done", `uv run` prefix):
- `uv run pytest tests/benchmarks/` (full benchmark gate, including new tests and the VN2 regression).
- `uv run ruff check .` and `uv run ruff format .` (lint/format clean).
- `uv run ty check calibre/` (type check — though changes are mostly YAML + tests).
- Local sanity: `uv run calibre run --config benchmarks/m5/config/smoke-cumulative.yaml` and the VN2
  equivalent each produce a ledger with cumulative bounds on terminal-H rows.
- Confirm the VN2 `total_cost=4992.20` regression baseline is untouched.

---

## Sources & Research

- Issue #198 (milestone: Architecture deepening II); source DF-26 (PR #178 residual,
  "fix: defer incomplete cumulative conformal windows").
- `calibre/conformal/runtime.py` — `SymmetricIntervalConfig`, `_apply_cumulative`, `_observe_cumulative`,
  the cumulative-mode invariants (`mscp` + `protection_period >= 1`).
- `calibre/cli/config.py` — `ConformalConfig` schema (`mode`, `protection_period`) and
  `to_runtime_config()`.
- `tests/benchmarks/test_m5_config.py` — the existing committed-config gate pattern (parse-assert +
  smoke execute via `run_config()`); `test_m5_runbook_keeps_fixture_out_of_statistical_acceptance`.
- `benchmarks/m5/config/{full,full-wls-struct,smoke}.yaml`, `benchmarks/vn2/config/{winning,smoke}.yaml`,
  `benchmarks/m5/README.md`.
- Out-of-scope reference (the *other* cumulative path): `calibre/conformal/cumulative_risk.py`,
  `benchmarks/vn2/config.py` (`CUMULATIVE_BEST_CONFIG`), `benchmarks/vn2/run_benchmark.py`.
