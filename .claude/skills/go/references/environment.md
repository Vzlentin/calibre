# Environment traps and CI-status recipes

The single home for the two local-tooling traps and the canonical `gh api`
recipes for reading CI status. SKILL.md's top "Project rules" pointer and
`ci-and-merge.md` both point here — **do not restate these recipes elsewhere.**

## Trap 1 — `git status --porcelain` emptiness is unreliable

This environment wraps `git`: on a clean tree `git status --porcelain` may emit
`ok` instead of empty output. **Never** decide clean/dirty from `--porcelain`
emptiness. Parse the full `git status` text instead:

```bash
git status 2>&1 | grep -qE '(nothing to commit|clean)'
# match (exit 0) => clean tree; no match (exit non-zero) => dirty
```

## Trap 2 — `gh pr checks --json` is broken

The local `gh` wrapper breaks `--json` with `unknown flag`. **Never** use
`gh pr checks --json`. Query the GitHub REST API directly (recipes below).

## Canonical CI-status recipes (single source)

Given a commit SHA, read CI status straight from the API:

```bash
# Combined status (overall green/red):
gh api repos/Vzlentin/calibre/commits/<sha>/status --jq '.state'
# Individual check runs (name / status / conclusion):
gh api repos/Vzlentin/calibre/commits/<sha>/check-runs \
  --jq '.check_runs[] | {name, status, conclusion}'
```

In Stage 5 the SHA is the PR head:
`HEAD_SHA=$(cd "$WORKDIR" && git rev-parse HEAD)`.
