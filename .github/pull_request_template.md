## Summary

<!-- One item per PR (or a tightly-cohesive pair). Link the issue. -->

## Definition of Done

- [ ] Scope is a single item; the PR is independently mergeable and revertible.
- [ ] A behavioral test asserts the production path (not `grep` / `wc -l`).
- [ ] **Frozen surfaces (`calibre/` + root):** change is a sanctioned carve-out; schema changes show the migration↔ORM parity test red first, then green.
- [ ] **Successor (`newcalibre/`):** conformance items landed as failing tests before the implementation; required successor checks green.
- [ ] Fresh adversarial review cleared (target: what breaks on first deploy / under concurrency / against untrusted input).
