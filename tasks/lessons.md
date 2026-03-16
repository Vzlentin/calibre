# Lessons

## Review Loop Control

- Start every non-trivial task by writing and maintaining `tasks/todo.md`.
- For a scoped implementation task, do not stack serial reviewer subagents after every small fix. Use at most one spec review and one quality review before parent verification unless a reviewer surfaces a clearly new high-severity issue.
- When a reviewer finds a concrete issue, prefer reusing the same implementer thread instead of spawning another approval reviewer immediately.
- Treat repeated "final approval" or "re-review" passes as a smell. Consolidate pending fixes, verify locally, then run one final approval review.
- If review feedback conflicts with the approved task guidance, resolve the ambiguity once in the parent context instead of bouncing between multiple reviewers.
- Liberal subagent use does not justify review churn. Extra review passes must buy new information, not just more confidence theater.
