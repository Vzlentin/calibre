# Context

Calibre's domain glossary lives in the Obsidian vault, not the repo:

`$OBSIDIAN_VAULT_PATH/Projects/Calibre/CONCEPTS.md`

It defines the shared vocabulary — terms, preferred names, and *Avoid* aliases —
that tickets, code, and docs use without re-explaining. Read it before working in
a documented area and use its terms in any output that names a domain concept.

This file is a **public-safe redirector**: it names *where* the glossary lives,
never the body. See `docs/agents/domain.md` for the full consumer rules, and
CLAUDE.md → *Agent memory* for resolving the vault path (check `.env` for
`OBSIDIAN_VAULT_PATH` — a bare `$OBSIDIAN_VAULT_PATH` in the shell reads empty on
this machine).
