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
as "no failures → green" and merges early. `skipped` and `neutral` conclusions
are non-blocking (conditional jobs produce them on runs where they don't
apply) and must not read as red. Equally, an empty check set, a malformed or
truncated payload (`total_count` disagreeing with the returned page), or a
failed `gh` call is a **non-verdict** — never green. `ci_verdict.py` encodes
exactly this; trust its exit code over any ad-hoc re-derivation.

## Cleanup policy (merge-gated)

Preserving is the default; cleanup is the exception, run only after the merge
is confirmed against the PR's actual state (`gh pr merge` can exit non-zero
after the API merge succeeded, and an already-merged PR on retry skips
straight to cleanup). The merge is pinned with `--match-head-commit` to the
head SHA the green verdict was computed for, so a branch that moved after the
verdict refuses to merge. A squash-merged branch never shows as "merged" to
git, so the local branch is force-deleted (`git branch -D`, not `-d`) — but
the worktree removal is **not** forced: uncommitted work inside the worktree
refuses the removal (exit 2, merged-but-cleanup-incomplete) instead of being
destroyed.

- **Direct mode** (the main checkout is on the PR branch from Stage 1): return
  to `main`, fast-forward, drop the branch.
- **Worktree mode** (the main checkout never left the user's branch/dirty
  tree): remove the worktree and drop the branch **without** `git checkout
  main`/`git pull` in the main checkout — preserving the user's branch and
  dirty tree is the whole point. The local `main` ref is fast-forwarded via
  `git fetch origin main:main`, skipped when the user is sitting on `main`
  (a checked-out branch cannot be moved by fetch).

Both modes finish by deleting the remote branch (`gh pr merge
--delete-branch` is deliberately not used: its local-delete half fails on
worktree checkouts and misreports a successful merge as a failure). A failed
remote delete only warns — a leftover remote ref is litter, not a safety
problem.

**Preserve path.** `--no-merge`, a refused merge (no `closes #N` handle), or a
failed merge command leaves the branch, worktree, and PR intact for
resume/debug (exit 1 — nothing was deleted). A cleanup step failing **after**
the merge is exit 2: the PR is merged; finish the printed step manually rather
than treating the run as unmerged.
