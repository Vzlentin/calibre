---
name: project-memory
description: Project-agnostic agent memory backed by an Obsidian vault. Read at the start of non-trivial tasks; write durable architecture, vision, lessons, and per-task plans.
---

# project-memory

Long-lived memory for any project the agent works on. The vault is the
shared notebook between sessions: read it before acting on non-trivial work
and write to it whenever something durable is learned or decided.

## Vault resolution

The vault root is read from the `OBSIDIAN_VAULT_PATH` environment variable.

- bash / zsh: `$OBSIDIAN_VAULT_PATH`
- PowerShell: `$env:OBSIDIAN_VAULT_PATH`

Rules:

- **Optional but preferred.** If the variable is set, use the vault as the
  source of truth for project memory.
- **Degrade gracefully, by surface.** If the vault is unset, empty, or absent,
  the fallback differs by what is being stored:
  - **Plans** fall back to the repo-relative `docs/plans/` store (see
    [Plan store location](#plan-store-location)). Plans are work orders, not
    private context, so they may live in the repo.
  - **Durable memory** (`architecture.md`, `vision.md`, `lessons.md`) is
    skipped entirely — do not invent a path, fall back to a hardcoded
    location, or write it into the repo. These carry private context that must
    stay out of a (possibly public) repo.
- **No personal paths in the repo.** Never commit absolute vault paths into
  the codebase or documentation.

## Project folder

Each project gets exactly one folder inside the vault, named after the
repository directory:

```
$OBSIDIAN_VAULT_PATH/Projects/<project>/
├── architecture.md
├── vision.md
├── lessons.md
└── plans/
```

Do not create any other permanent files or folders in the project memory
surface. `plans/` may contain many transient files; the three top-level
markdown files are the only durable ones.

## Plan store location

Plans resolve to one of two stores, depending on vault availability. This is
the single home for plan-location logic — callers (e.g. `/go`) delegate here
rather than hardcoding a path.

- **Vault mode** — `OBSIDIAN_VAULT_PATH` set and the project folder reachable:
  `$OBSIDIAN_VAULT_PATH/Projects/<project>/plans/`. Resolve `<project>` by
  matching the repository directory name **case-insensitively** against
  existing `Projects/*` entries (so a repo dir `calibre` resolves to an
  existing `Projects/Calibre/`); if none matches, use the repo dir name
  verbatim, created on first write.
- **Fallback mode** — vault unset, empty, or absent: the repo-relative
  `docs/plans/` store. No vault paths are touched.

### Plan placement and relocation

`/ce-plan` writes a fresh plan to the repo at
`docs/plans/YYYY-MM-DD-NNN-<type>-<name>-plan.md`. Place it at the resolved
store:

- **Vault mode:** relocate it into the vault — rewrite its frontmatter to the
  vault convention (`title`, `type` (`feat|fix|refactor|chore`),
  `status: active`, `date`, and `origin` when known), write to
  `$OBSIDIAN_VAULT_PATH/Projects/<project>/plans/<YYYY-MM-DD-slug>.md`, verify
  the write, then delete the `docs/plans/` copy. One source of truth, no
  private context in a public repo.
- **Fallback mode:** leave the plan in `docs/plans/` as-is — it is already at
  the resolved store.

### Plan-status persistence

When a task ships, flip the plan's `status` in the resolved store:

- **Vault mode:** set `status` on the vault plan, update `architecture.md` /
  `lessons.md` where a durable decision or lesson warrants it, and commit +
  push the vault (it is its own git repo).
- **Fallback mode:** set `status` on the `docs/plans/` plan — that file is the
  record. Do **not** write durable vault memory; there is no vault.

## File roles

- **`architecture.md`** — durable system design: module boundaries, data
  contracts, key invariants, deliberate trade-offs. Edited when a design
  decision lands.
- **`vision.md`** — product intent: goals, non-goals, north-star, scope.
  Edited only when product direction shifts.
- **`lessons.md`** — append-only log of corrections, recurring pitfalls, and
  rules-for-self that prevent repeating mistakes. Append after every user
  correction.
- **`plans/<slug>.md`** — one file per non-trivial task. Working memory:
  goal, plan, progress, outcomes. Update as the task evolves.

## When to read

- At the start of any non-trivial task (3+ steps or architectural impact),
  read `architecture.md`, `vision.md`, and recent `lessons.md` entries.
- Before any architectural decision, re-read `architecture.md` and check
  `plans/` for a related in-flight plan.
- After a user correction, scan `lessons.md` for the relevant pattern
  before responding.

## When to write

- **`architecture.md`** — when a design decision is made or an invariant
  changes. Keep entries terse; cite the change in the codebase.
- **`vision.md`** — only when goals, non-goals, or scope move.
- **`lessons.md`** — append immediately after any user correction or any
  mistake that could recur. Each entry: the pattern, the rule for next time.
- **`plans/<slug>.md`** — create at the start of any multi-step task,
  update through the task, leave behind as the record when complete.

## CLI usage

The vault is a directory of Markdown files. Use standard filesystem tools
(replace `<project>` with the repo name):

```bash
cat "$OBSIDIAN_VAULT_PATH/Projects/<project>/architecture.md"
printf '\n## <pattern>\n- rule\n' >> "$OBSIDIAN_VAULT_PATH/Projects/<project>/lessons.md"
cat > "$OBSIDIAN_VAULT_PATH/Projects/<project>/plans/<slug>.md" << 'EOF'
...
EOF
grep -r "<term>" "$OBSIDIAN_VAULT_PATH/Projects/<project>/" --include="*.md" -l
```

If `OBSIDIAN_VAULT_PATH` is unset, skip durable vault operations; plans still
resolve to the `docs/plans/` fallback (see [Plan store location](#plan-store-location)).
