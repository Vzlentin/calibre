# Domain Docs

How the engineering skills should consume this project's domain documentation.
**The canonical home is the Obsidian vault; the repo carries only public-safe
redirectors.** Nothing durable — glossary, ADRs, plans — lives in the code repo.

## Before exploring, read these (in the vault)

Resolve the vault root from the repo's `.env` first (`grep OBSIDIAN_VAULT_PATH
.env`) — do **not** trust a bare `$OBSIDIAN_VAULT_PATH` in the shell, which
reads empty on this machine (see CLAUDE.md → *Agent memory*). The project folder
is `Projects/Calibre/`.

- **Glossary** — `<vault>/Projects/Calibre/CONCEPTS.md`. The shared domain
  vocabulary: terms, preferred names, and *Avoid* aliases.
- **ADRs** — `<vault>/Projects/Calibre/adr/` (numbered `NNNN-<slug>.md` plus a
  `README.md` index). Read the ADRs that touch the area you're about to work in.
- **Plans** — `<vault>/Projects/Calibre/plans/` (`YYYY-MM-DD-NNN-<slug>-plan.md`
  plus `reviews/`). Read the plan for any work item before implementing.
- **Broader context** — the vault also holds `architecture.md`, `lessons.md`,
  `vision.md`, `ROADMAP.md`, `STRATEGY.md`, `solutions/` (per-problem learnings).
  See CLAUDE.md → *Agent memory* for the full map and consumer rules; this file
  covers only the glossary + ADRs + plans the Matt Pocock skills look for.

## Repo paths are redirectors, not bodies

The repo keeps **public-safe redirector** entrypoints at the conventional paths
— they name *where* the vault store is, never an artifact body. Treat them as
pointers and resolve into the vault:

- `CONTEXT.md` → vault `Projects/Calibre/CONCEPTS.md` (glossary)
- `docs/adr/README.md` → vault `Projects/Calibre/adr/` (ADRs)
- `docs/plans/README.md` → vault `Projects/Calibre/plans/` (plans)

If a skill reads `CONTEXT.md` expecting a glossary body, it will find a pointer
instead — resolve to the vault `CONCEPTS.md` as above. Do **not** write ADRs or
plans into the repo; new ones land in the vault at the next number/slug.

## If the vault is unavailable

If `.env` carries no `OBSIDIAN_VAULT_PATH` line (truly unset — not merely absent
from the live shell), the vault is unavailable: **proceed silently**. Don't flag
the absence; don't suggest creating the vault. The glossary and ADRs won't be
available. A skill may write a **temporary** local plan under `docs/plans/` to
keep work moving, then relocate it to the vault on return (see CLAUDE.md →
*Agent memory*); never write durable memory (ADRs, glossary, solutions) to a
guessed repo path.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal,
a hypothesis, a test name), use the term as defined in `CONCEPTS.md`. Don't drift
to synonyms the glossary explicitly lists under *Avoid*.

If the concept you need isn't in the glossary yet, that's a signal — either
you're inventing language the project doesn't use (reconsider) or there's a real
gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than
silently overriding:

> _Contradicts ADR-0007 (M5 coverage-neutrality gate runs the real pipeline) —
> but worth reopening because…_
