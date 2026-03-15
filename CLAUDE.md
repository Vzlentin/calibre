# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calibre is a Python project. The repository is in early stages of development.

## Development Setup

This is a Python project. Update this section as tooling is added (e.g., pyproject.toml, requirements.txt, Makefile).

## Commands




| Task            | Command                                                   |
| --------------- | --------------------------------------------------------- |
| Install deps    | `pip install -e .` (or `uv pip install -e .` if using uv) |
| Run tests       | `pytest`                                                  |
| Run single test | `pytest path/to/test_file.py::test_name`                  |
| Lint            | `ruff check .`                                            |
| Format          | `ruff format .`                                           |
| Type check      | `mypy .`                                                  |


## Obsidian Vault

This project has a dedicated Obsidian vault at `C:\Users\a933186\Vault`.

**Agents working in this codebase should:**

- Use the `obsidian` CLI (via Bash) to read/write project notes in the vault
- Target the vault explicitly: `obsidian vault="Vault" <command>`
- Store project decisions, architecture notes, research, and task context in the vault under a `calibre/` folder
- Check the vault for existing context before starting non-trivial tasks: `obsidian vault="Vault" search query="calibre" limit=20`

**Common patterns:**

```bash
obsidian vault="Vault" read path="calibre/architecture.md"
obsidian vault="Vault" create name="My Note" content="..." silent
obsidian vault="Vault" append file="My Note" content="..."
obsidian vault="Vault" search query="search term" limit=10
```

## Behaviour

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy

- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons. md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo. md`
6. **Capture Lessons**: Update `tasks/lessons. md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.

