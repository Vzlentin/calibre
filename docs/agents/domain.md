# Domain Docs

How engineering skills consume Calibre's domain documentation. Project memory is
an OKF bundle in the configured Obsidian vault; the successor specification and
its ADRs are canonical public artifacts in this repository.

## Progressive project-memory discovery

Resolve `OBSIDIAN_VAULT_PATH` from the environment or the repository's `.env`.
When the vault is available:

1. Start at `Projects/Calibre/index.md`.
2. Read `Projects/Calibre/CONTEXT.md` when domain vocabulary matters.
3. Follow only the relevant entry under `Projects/Calibre/architecture/`.
4. Read the matching active work order under `Projects/Calibre/plans/`.

Indexes provide progressive disclosure. Do not bulk-load the bundle or assume a
legacy monolithic architecture, solutions, lessons, or deferred-register file
exists.

## Authority

- `docs/spec/` is normative successor design.
- Successor implementation is implementation fact.
- The vault architecture bundle is the navigational synthesis, including
  implementation state, known deltas, and private rationale.
- On disagreement, the public specification wins as the intended contract.

Use stable prose names **Calibre** or **successor engine**, and **frozen oracle**.
Temporary source-directory names are code anchors, not product vocabulary.

## Conventional redirectors

- Root `CONTEXT.md` points to the vault glossary.
- `docs/plans/README.md` points to active vault plans. If the vault is
  unavailable, temporary public-safe plans may live under `docs/plans/` and move
  to the vault on return.
- `docs/adr/README.md` points directly to `docs/spec/adr/`, the only successor
  ADR series. Never create or copy successor ADR bodies under `docs/adr/`.

## Specification editing

Start at `docs/spec/00-overview.md`. Every change under `docs/spec/` requires an
owner leak-review stamp on the landing issue before it lands. Reviews are
batched per landing.

Public `[ANNEX:*]` references remain opaque contracts. Do not resolve them,
identify their private storage, or inline their material into repository prose.
Every pointer in use must remain registered in `docs/spec/90-annex-registry.md`.

## Vault-unavailable behavior

Proceed with repository sources and public-safe plan fallback. Do not invent a
vault path or write private durable memory into the repository.

## Vocabulary and decisions

Use terms from the canonical `CONTEXT.md`. A missing term is a prompt to
reconsider or resolve the language, not to invent a competing synonym.

Read relevant decisions from `docs/spec/adr/`. Surface a contradiction
explicitly rather than silently overriding a ratified decision.
