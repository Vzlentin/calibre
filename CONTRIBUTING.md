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

Calibre ships its roadmap as **one small PR per item**, each independently
mergeable and **independently deployable**, merged to `main` before the next
starts. No mega-branches. `main` stays deployable at every merge.

The full cadence, the Definition of Done, and the **four CI gates**
(migration↔ORM parity · Postgres in CI · deploy smoke · behavioral DoD) are
documented in [docs/development-workflow.md](docs/development-workflow.md). Read
it before opening your first PR — it exists because a single 77-file branch
passed CI green yet broke on first deploy.

The key rule that surprises people: **Done is proven by a behavioral test that
asserts the production path**, never by `grep` or `wc -l`. A green lint+unit run
is not "done"; deployable is.

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
