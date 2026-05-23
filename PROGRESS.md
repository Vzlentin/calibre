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
