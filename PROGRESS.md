```yaml
phase: 1
last_completed_task: "1.a cherry-pick observe silent-return fix"
next_task: "1.b fix observe cumulative dispatch"
last_commit: "33732f9"
notes: "Verified _run_observe_job logs a warning before returning when last_calibrated is empty."
```

```yaml
phase: 1
last_completed_task: "1.b fix observe cumulative dispatch"
next_task: "1.c extract _cap_threaded_config"
last_commit: "08c6cb4"
notes: "Routed /observe through observe_cumulative or observe_per_horizon; uv run pytest tests/api/test_observe.py passed."
```

```yaml
phase: 1
last_completed_task: "1.c extract _cap_threaded_config"
next_task: "1.d add PartitionedConformalRuntime Protocol"
last_commit: "8c12edb"
notes: "Moved threaded config capping to calibre.execution.threading; focused threading and tuning tests passed."
```

```yaml
phase: 1
last_completed_task: "1.d add PartitionedConformalRuntime Protocol"
next_task: "phase 1 DoD and cross-phase regression gate"
last_commit: "e02708b"
notes: "Added typed partitioned runtime/store access; focused conformal and state-resume tests passed."
```

```yaml
phase: 1
last_completed_task: "phase 1 DoD and cross-phase regression gate"
next_task: "2.a SQL-back LifecycleStore"
last_commit: "0fbe817"
notes: "Phase 1 gate green: ruff check ., mypy calibre/, pytest; baseline type_ignore_count=16 any_count=158."
```

```yaml
phase: 2
last_completed_task: "2.a SQL-back LifecycleStore"
next_task: "2.b make /fit actually fit"
last_commit: "54f9979"
notes: "Added SqlLifecycleStore, lifecycle SQL tables/migration, and SQL API mode; tests/api passed with memory and LIFECYCLE_STORE=sql."
```

```yaml
phase: 2
last_completed_task: "2.b make /fit actually fit"
next_task: "phase 2 DoD and cross-phase regression gate"
last_commit: "2b7f9ac"
notes: "Fit jobs now validate config, train adapters, persist ModelArtifactCache artifacts, and /predict reuses cache hits."
```

```yaml
phase: 2
last_completed_task: "phase 2 DoD and cross-phase regression gate"
next_task: "3.a replace manual YAML parsing with pydantic models"
last_commit: "678e795"
notes: "Phase 2 gate green: tests/api, LIFECYCLE_STORE=sql tests/api, ruff check ., mypy calibre/, pytest."
```

```yaml
phase: 3
last_completed_task: "3.a replace manual YAML parsing with pydantic models"
next_task: "phase 3 DoD and cross-phase regression gate"
last_commit: "1921634"
notes: "Replaced CLI config parsing helpers with pydantic section models; tests/cli, mypy calibre/cli, and ruff config checks passed."
```
