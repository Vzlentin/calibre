# Stage 5 — CI log-pull and cleanup-by-mode bash

Stage 5 in `SKILL.md` owns the CI verdict logic itself — the pending/green/failure
read of `check-runs`, the autofix-loop shape, the max-3-iterations cap, the PR #38
repeated-signature stop, the on-green merge decision, the merge-gated cleanup
decision, and the GATE. This file holds only the supporting bash: the failed-run
log-pull and the cleanup-by-mode commands.

CI status is read via the canonical `gh api .../check-runs` (Stage 5 specifies the
pending/green/failure verdict over its output); the broken local check-status
wrapper and the other host-tooling traps are documented in
`references/environment.md`. In Stage 5 the SHA is
`HEAD_SHA=$(cd "$WORKDIR" && git rev-parse HEAD)`.

## Pull logs for failed runs

```bash
gh api repos/Vzlentin/calibre/commits/$HEAD_SHA/check-runs \
  --jq '.check_runs[] | select(.conclusion=="failure") | .details_url'
# For each failed run, get the workflow run ID from the URL, then:
gh run view <run-id> --log-failed
```

## Cleanup-by-mode (merge-gated — run only after `gh pr merge --squash` confirms)

A squash-merged branch never shows up as "merged" to git, so the local branch
must be force-deleted with `git branch -D` (not `-d`).

- **Direct mode** (the main checkout is on the PR branch from Stage 1): return to
  `main`, fast-forward, then drop the local branch:

```bash
git checkout main && git pull --ff-only
git branch -D <type>/<slug>
```

- **Worktree mode** (the main checkout never left the user's branch/dirty tree):
  from the main checkout, remove the worktree and drop the branch without
  touching the user's working tree:

```bash
cd "$MAIN"
git worktree remove .worktrees/<slug>    # add --force if the tree is dirty
git branch -D <type>/<slug>              # squash-merge => force-delete
git worktree prune
git fetch origin main:main               # fast-forward local main; skip if the user is sitting on main
```

  Do **not** `git checkout main` or `git pull` in the main checkout in worktree
  mode — preserving the user's branch and dirty tree is the whole point.
