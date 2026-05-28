# Development workflow

This is the workflow Calibre uses to ship the improvement-wave roadmap. It
exists because PR #38 (a single 6-phase, 77-file branch) passed CI green yet
failed on first deployment: the suite built its schema with
`Base.metadata.create_all()` and never ran `alembic upgrade head`, so a
migration↔ORM mismatch (the ORM expected `*_ref` columns and a
`fit_frame_artifacts` table that no migration created) shipped undetected.
See `REVIEW-PR-38.md` / `FIX.md` on the abandoned `cardinal-improvements`
branch for the full post-mortem.

## One small PR per roadmap item

- One roadmap item (or a tightly-cohesive pair) per PR.
- Each PR is independently mergeable and **independently deployable**, and is
  **merged to `main` before the next starts**. No mega-branches — slop must not
  be allowed to compound across phases.
- `main` stays deployable at every merge.

## Definition of Done (per PR)

DoD is proven by a **behavioral test that asserts the contract / production
path** — never by `grep` or `wc -l`. Specifically:

- A test exercises the real behavior the change promises (the production code
  path, not a shape check).
- **If the change touches the schema:** the migration is written; the
  migration↔ORM parity test (`tests/test_storage_migrations.py`) is confirmed
  **red first**, then green; a repository round-trip runs against the
  migration-built database.
- All four gates are green (below).
- A fresh, cold-context reviewer has cleared the PR with the target *"what
  breaks on first deploy / under concurrency / against untrusted input?"* — not
  "is CI green?". Any fix round is re-reviewed for **newly introduced** issues,
  not only whether the original list was addressed.

## The four gates

1. **Migration↔ORM parity** — `alembic upgrade head`, then
   `compare_metadata` against `Base.metadata`; any missing/extra table or
   column fails. This is the gate that would have caught PR #38.
2. **Postgres in CI** — the `test` job runs against a real Postgres service
   (`CALIBRE_TEST_DATABASE_URL`), not just SQLite `create_all()`.
3. **Deploy smoke** — boot the app on a migrated database and exercise
   `/fit` → `/predict` → `/calibrate` → `/order` → `/observe`
   (`tests/integration/test_deploy_smoke.py`).
4. **Behavioral DoD** — contract tests, not `grep`/`wc -l`. Add targeted tests
   for the recurring failure classes: concurrency, serialization trust
   boundaries, cache-key correctness, and no-silent-fallback.

"Green" means *deployable* — boots on a migrated DB and serves the core
endpoints — not merely "lint + unit pass".
