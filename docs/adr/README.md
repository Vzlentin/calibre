# Architecture Decision Records

Calibre's ADRs live in the Obsidian vault, not the repo:

`$OBSIDIAN_VAULT_PATH/Projects/Calibre/adr/`

Numbered `NNNN-<slug>.md` with a `README.md` index. Read the ADRs that touch the
area you're about to work in, and flag any contradiction explicitly rather than
silently overriding.

This directory is a **public-safe redirector**: it names *where* the ADRs live,
never an ADR body. New ADRs land in the vault at the next number in sequence (see
the vault `adr/README.md`). See `docs/agents/domain.md` for the full consumer
rules.
