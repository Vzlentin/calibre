# Contributing to Calibre

Thanks for your interest in Calibre. This guide is the human entry point to the
project. It links to the canonical docs rather than restating them, so there is a
single source of truth for each topic.

## Getting set up

You need Python `>=3.11` and [`uv`](https://docs.astral.sh/uv/). Clone the repo,
then:

```bash
uv sync --extra dev --extra benchmarks
```

The [README](README.md) covers what Calibre is, how to run a backtest, serve the
API, and run the benchmarks.

Always prefix Python tooling with `uv run` — never invoke `python`, `pytest`,
`ruff`, or `ty` directly:

| Task         | Command                              |
| ------------ | ------------------------------------ |
| Run tests    | `uv run pytest`                      |
| Lint         | `uv run ruff check .`                |
| Format       | `uv run ruff format .`               |
| Type check   | `uv run ty check calibre/`           |

## How we work

Calibre ships **one small PR per item**, each independently mergeable and
revertible, merged to `main` before the next starts. No mega-branches. This
cadence exists because a single 77-file branch once passed CI green yet broke
on first deploy.

Two tracks coexist during the rewrite:

- **`calibre/` (frozen engine)** — maintenance-only; it serves as the behavior
  oracle for the rewrite. Changes are limited to sanctioned carve-outs. Its CI
  enforces the deployability gates directly: migration↔ORM parity, tests
  against real Postgres, and a deploy smoke run — schema changes must show the
  parity test red first, then green.
- **`newcalibre/` (successor)** — built to `docs/spec/`
  (start at `docs/spec/00-overview.md`), conformance-first: a chapter's
  conformance items land as failing tests before the implementation. It has
  its own lockfile, tooling, and required CI lanes.

The key rule that surprises people: **Done is proven by a behavioral test that
asserts the production path**, never by `grep` or `wc -l`. A green lint+unit run
is not "done".

## Opening a pull request

1. Branch from `main` with a meaningful name (e.g. `feat/...`, `fix/...`,
   `chore/...`).
2. Scope the PR to a single roadmap item (or a tightly-cohesive pair).
3. Fill in the [pull request template](.github/pull_request_template.md) — it is
   the DoD checklist, including the four gates.
4. Make sure lint, type check, and the test suite pass locally.

## Roadmap and issues

Live status is on **GitHub** — the active milestone and its issues are the source
of truth (`gh issue list --milestone "<active milestone>"`). Durable rationale
(mission, root-issue analysis, dependency ordering) lives in the project's
Obsidian vault, not in the repo. Don't mirror status into prose; link to the
milestone.

## License

Calibre is licensed under the [Apache License 2.0](LICENSE). By contributing, you
agree that your contributions are licensed under the same terms (Apache-2.0
§5). Commits authored with AI assistance carry a `Co-Authored-By` trailer.
