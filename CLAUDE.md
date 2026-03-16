# [CLAUDE.md](http://CLAUDE.md)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calibre is a demand planning project. The repository is in early stages of development.

## Commands

> **Always use `uv` to run Python tools.** Never invoke `python`, `pytest`, `ruff`, `mypy`, or any other Python tool directly — always prefix with `uv run`.


| Task            | Command                                         |
| --------------- | ----------------------------------------------- |
| Install deps    | `uv pip install`                                |
| Run tests       | `uv run pytest`                                 |
| Run single test | `uv run pytest path/to/test_file.py::test_name` |
| Lint            | `uv run ruff check .`                           |
| Format          | `uv run ruff format .`                          |
| Type check      | `uv run mypy .`                                 |


## Obsidian Vault

This project has a dedicated Obsidian vault at `C:\Users\a933186\Vault`.
It has a dedicated sub-folder for agents, structured like this:

```
calibre/
└── agents/
    ├── home.md
    ├── knowledge/
    │   ├── architecture.md
    │   ├── lessons.md
    │   └── product.md
    └── tasks/
        └── todo.md
```

**Agents working in this codebase should:**

- Use the `obsidian` CLI to read/write project notes in the vault
- Target the vault explicitly: `obsidian vault="Vault" <command>`
- Store project decisions, architecture notes, research and task context in the vault under `calibre/agents/`
- Check the vault for existing context before starting non-trivial tasks: `obsidian vault="Vault" search query="calibre" limit=20`
- Use Obsidian as the shared agent-memory harness for this project

**Common patterns:**

```bash
obsidian vault="Vault" read path="calibre/agents/knowledge/architecture.md"
obsidian vault="Vault" append path="calibre/agents/knowledge/lessons.md" content="## <lesson>\n- pattern"
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

- After ANY correction from the user: append the pattern to `calibre/agents/knowledge/lessons`
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

1. **Plan First**: Write the plan to `calibre/agents/tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add the review section to `calibre/agents/tasks/todo.md`
6. **Capture Lessons**: Update `calibre/agents/knowledge/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.

