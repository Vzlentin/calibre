# Stage 0.7 — git-state read and worktree provisioning bash

Read git state in the **main checkout** to drive the Stage 0.7 mode decision,
and (worktree mode only) provision an isolated worktree. SKILL.md keeps the
mode decision and the GATE; this file holds the bash.

## Read git state (main checkout)

```bash
MAIN=$(git rev-parse --show-toplevel)        # main checkout path; capture before any cd
CURRENT_BRANCH=$(git branch --show-current)  # empty => detached HEAD
# Clean/dirty via full status text, NOT --porcelain emptiness (see environment.md, Trap 1):
git status 2>&1 | grep -qE '(nothing to commit|clean)'
CLEAN=$?   # 0 => clean tree, non-zero => dirty
```

Mode (decided in SKILL.md): on `main` **and** `CLEAN -eq 0` => DIRECT mode
(`WORKDIR="$MAIN"`, no provisioning); any other branch, detached HEAD, or
`CLEAN -ne 0` => WORKTREE mode (provision below).

## Provision (worktree mode only)

`<type>` is the kind `ce-work` would choose (`feat|fix|refactor|chore`);
`<slug>` is the Stage 0 slug. Run from the main checkout. The setup steps are
**defined in `.cursor/worktrees.json`** under `setup-worktree-unix` — read them
dynamically (never hardcode) so a change to the worktree config is picked up,
substituting `$ROOT_WORKTREE_PATH` -> `$MAIN` (Cursor injects that var so it
resolves to the main checkout; `/go` runs the steps itself, so it must supply
the value explicitly):

```bash
git fetch origin
git worktree add .worktrees/<slug> -b <type>/<slug> origin/main
cd .worktrees/<slug>
# Read setup-worktree-unix from worktrees.json (gh-consistent jq/sed, not python3),
# substituting $ROOT_WORKTREE_PATH -> $MAIN:
STEPS=$(jq -r '."setup-worktree-unix"[]' "$MAIN/.cursor/worktrees.json" \
  | sed "s|\$ROOT_WORKTREE_PATH|$MAIN|g")
# Process substitution (NOT a pipe) so a failed step actually aborts this shell:
while read -r step; do
    echo "  -> $step"
    eval "$step" || { echo "FAILED: $step"; exit 1; }
done < <(printf '%s\n' "$STEPS")
WORKDIR=$(pwd)                               # absolute worktree path
```

Per CLAUDE.md "Worktrees": a *warm* `uv sync` takes seconds; **never** copy the
main `.venv` (it is non-relocatable), and **never** set `UV_LINK_MODE`.

**Data provisioning.** `setup-worktree-unix` **copies** `data/vn2` (~4 MB) but
**links** `data/m5` as an NTFS junction (`mklink /J`) — `data/m5` is ~466 MB
(~116× vn2), so copying it into every worktree would bloat disk for no gain. The
link is **read-only-safe**: M5 reads `data/m5` and writes the separate
`results/m5`, so the worktree never mutates the linked source. Both data steps
carry `test -d … || true`, which **swallows a failed link** — a data-less
worktree then provisions "successfully", so the real catch is the **conditional
`data/m5` presence check in SKILL.md's Stage 0d GATE** for M5-dependent items
(the `import calibre` check passes without data).

The real provisioning check is the venv import in SKILL.md's Stage 0.7 GATE
(`cd "$WORKDIR" && uv run python -c "import calibre"`), plus that conditional
`data/m5` presence check for M5-dependent items.
