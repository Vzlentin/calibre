---
# Calibre Improvement Wave 1 — Progress

## Phase 1 · Backend Stabilization ✅

```yaml
phase: 1
last_completed_task: "1.d replace getattr chains with PartitionedConformalRuntime Protocol"
next_task: "2.a SQL-back LifecycleStore"
last_commit: "da041d47c192c5597b0c39ddf76db5e921fd8d08"
notes: |
  1.a already applied (96348b9 cherry-pick). 1.b routes _run_observe_job through
  decision_loop dispatch (observe_per_horizon / observe_cumulative) instead of
  dropna+runtime.observe. 1.c extracts _cap_threaded_config/_thread_budget to
  calibre/execution/threading.py. 1.d adds PartitionedConformalRuntime Protocol
  (@runtime_checkable) and replaces both getattr chains in backend.py.
  DoD green: 7 new tests pass; grep getattr.*partition returns nothing;
  _cap_threaded_config shows only import in backend.py and optimizer.py;
  mypy (96 files) + ruff clean.
type_ignore_baseline: 47  # grep "type: ignore\|: Any" calibre/ — used as Phase 6 baseline
```

---

## Phase 2 · API Lifecycle Correctness — IN PROGRESS

```yaml
phase: 2
last_completed_task: ""
next_task: "2.a SQL-back LifecycleStore"
last_commit: ""
notes: ""
```
