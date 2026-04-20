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
- **Degrade gracefully.** If unset or empty, skip all vault reads/writes.
  Do not invent a path, do not fall back to a hardcoded location, do not
  create files outside the vault.
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

The `obsidian` CLI is parameterized on the resolved vault path. Examples
(replace `<project>` with the repo name):

```bash
obsidian vault="$OBSIDIAN_VAULT_PATH" read    path="<project>/architecture.md"
obsidian vault="$OBSIDIAN_VAULT_PATH" read    path="<project>/lessons.md"
obsidian vault="$OBSIDIAN_VAULT_PATH" append  path="<project>/lessons.md" content="## <pattern>\n- rule"
obsidian vault="$OBSIDIAN_VAULT_PATH" write   path="<project>/plans/<slug>.md" content="..."
obsidian vault="$OBSIDIAN_VAULT_PATH" search  query="<term>" limit=20
```

PowerShell uses `$env:OBSIDIAN_VAULT_PATH` in place of `$OBSIDIAN_VAULT_PATH`.

If `OBSIDIAN_VAULT_PATH` is unset, skip these commands entirely and tell
the user once that persistent memory is disabled for this session.
