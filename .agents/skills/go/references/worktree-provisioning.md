# Stage 0d — mode and provisioning rationale

The git-state read, mode decision, and provisioning mechanics live in
`.agents/skills/go/scripts/provision_worktree.py` (`decide` / `provision`);
SKILL.md's Stage 0d owns the GATE. This file keeps the judgment behind them.

## Why the mode split

DIRECT mode (on `main` and clean) lets `ce-work` branch inside the main
checkout — cheapest path, nothing to preserve. Any other state — another
branch, detached HEAD, or a dirty tree — means the user's checkout carries
context that must not move, so WORKTREE mode cuts an isolated worktree on a
fresh branch from `origin/main`. The script reads clean/dirty via
`git status --porcelain` through Python's subprocess, which bypasses the MSYS
wrapper trap that motivated the old full-text detector (see
`references/environment.md`).

The setup steps are **defined in `.cursor/worktrees.json`** under
`setup-worktree-unix` — the script reads them dynamically (never hardcoded) so
a config change is picked up, substituting `$ROOT_WORKTREE_PATH` with the main
checkout path (Cursor injects that var; the script supplies it itself).

Per CLAUDE.md "Worktrees": a *warm* `uv sync` takes seconds; **never** copy the
main `.venv` (it is non-relocatable), and **never** set `UV_LINK_MODE`.

## Data provisioning

`setup-worktree-unix` **copies** `data/vn2` (~4 MB) but **links** `data/m5` as
an NTFS junction (`mklink /J`) — `data/m5` is ~466 MB (~116× vn2), so copying
it into every worktree would bloat disk for no gain. The link is
**read-only-safe**: M5 reads `data/m5` and writes the separate `results/m5`,
so the worktree never mutates the linked source. Both data steps carry
`test -d … || true`, which **swallows a failed link** — a data-less worktree
then provisions "successfully". The real catch is the script's `--require-m5`
gate for M5-dependent items (the `import calibre` venv gate passes without
data), per SKILL.md's Stage 0d data-presence GATE.
