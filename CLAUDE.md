# CLAUDE.md

## Project

Calibre is a demand planning engine: probabilistic forecasting + conformal
intervals + ordering policies, exercised through backtesting pipelines.

## Commands

Always prefix Python tooling with `uv run`. Never invoke `python`, `pytest`,
`ruff`, or `mypy` directly.

| Task            | Command                                         |
| --------------- | ----------------------------------------------- |
| Install deps    | `uv sync --extra dev`                           |
| Run tests       | `uv run pytest`                                 |
| Run single test | `uv run pytest path/to/test_file.py::test_name` |
| Lint            | `uv run ruff check .`                           |
| Format          | `uv run ruff format .`                          |
| Type check      | `uv run mypy calibre/`                          |

## Core rules

- **Simplicity first.** Smallest change that solves the problem. No
  speculative abstractions.
- **Root cause, not symptoms.** No temporary patches or `try/except` to
  silence failures.
- **Plan before non-trivial work.** 3+ steps or any architectural decision
  goes through a plan first.
- **Verify before done.** Run the relevant tests / lints / type checks and
  show they pass before marking a task complete.

## Agent memory

Long-lived project memory lives in an Obsidian vault and is governed by the
`project-memory` skill at
[`.claude/skills/project-memory/SKILL.md`](.claude/skills/project-memory/SKILL.md).
Read it at the start of any non-trivial task.

The vault location is read from the `OBSIDIAN_VAULT_PATH` environment
variable. The project folder inside the vault is `calibre/`. If the env var
is unset, skip vault operations and proceed without persistent memory.
