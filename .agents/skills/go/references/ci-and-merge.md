# Stage 5 — CI verdict, merge, and cleanup policy

The mechanics live in the skill scripts: `ci_verdict.py` (verdict over the
typed check-runs API, failure signature, failed-log pull) and
`merge_cleanup.py` (`closes #N` verification, squash-merge, merge-gated
cleanup by mode). SKILL.md's Stage 5 owns the loop shape — the max-3-iterations
cap, the repeated-signature stop, the on-green merge decision, and the GATE.
This file keeps the policy behind them.

## Verdict policy

A verdict needs `status` **and** `conclusion` together: a still-pending run has
no conclusion yet, so any filter that only looks for failures reads "pending"
as "no failures → green" and merges early. Equally, an empty check set, a
malformed payload, or a failed `gh` call is a **non-verdict** — never green.
`ci_verdict.py` encodes exactly this; trust its exit code over any ad-hoc
re-derivation.

## Cleanup policy (merge-gated)

Preserving is the default; cleanup is the exception, run only after
`gh pr merge --squash` confirms. A squash-merged branch never shows as
"merged" to git, so the local branch is force-deleted (`git branch -D`, not
`-d`).

- **Direct mode** (the main checkout is on the PR branch from Stage 1): return
  to `main`, fast-forward, drop the branch.
- **Worktree mode** (the main checkout never left the user's branch/dirty
  tree): remove the worktree and drop the branch **without** `git checkout
  main`/`git pull` in the main checkout — preserving the user's branch and
  dirty tree is the whole point. The local `main` ref is fast-forwarded via
  `git fetch origin main:main`, skipped when the user is sitting on `main`
  (a checked-out branch cannot be moved by fetch).

**Preserve path.** `--no-merge`, a refused merge (no `closes #N` handle), a
failed merge command, or any failed cleanup step leaves the branch, worktree,
and PR intact for resume/debug.
