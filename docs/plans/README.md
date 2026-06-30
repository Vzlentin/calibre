# Plans

Calibre's plans live in the Obsidian vault, not the repo:

`$OBSIDIAN_VAULT_PATH/Projects/Calibre/plans/`

One file per plan (`YYYY-MM-DD-NNN-<slug>-plan.md`), plus `reviews/`. The vault is
the single source of truth — read the plan for any work item before implementing.

This directory is a **public-safe redirector**: it names *where* the plans live,
never a plan body. New plans are written to the vault; if the vault is
unreachable a skill may write a temporary local plan here, then relocate it to
the vault on return (see CLAUDE.md → *Agent memory*). See
`docs/agents/domain.md` for the full consumer rules.
