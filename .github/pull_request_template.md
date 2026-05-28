## Summary

<!-- One roadmap item (or a tightly-cohesive pair). Link the item. -->

## Definition of Done

See [docs/development-workflow.md](../docs/development-workflow.md).

- [ ] Scope is a single roadmap item; PR is independently deployable.
- [ ] Behavioral test asserts the production path (not `grep` / `wc -l`).
- [ ] **Schema changes:** migration written; parity test confirmed **red first**, then green; repo round-trip on the migrated DB.
- [ ] All four gates green: migration↔ORM parity · Postgres in CI · deploy smoke · behavioral DoD.
- [ ] Folds in the relevant `REVIEW-PR-38.md` finding so the bug is not rebuilt.
- [ ] Fresh adversarial review cleared (target: breaks on first deploy / under concurrency / against untrusted input).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
