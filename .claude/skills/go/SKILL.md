---
name: go
description: Calibre implementation-orchestration pipeline. Given an already-specced work item (a GitHub issue), drive it end-to-end — implement → simplify → review → resolve feedback → babysit CI → squash-merge. Invoke with /go <issue> when a backlog item is ready to build hands-off.
argument-hint: "[issue number, #N, or roadmap work-item code like U3]"
---

You are running the Calibre implementation-orchestration pipeline for the work
item in `$ARGUMENTS`. Execute the stages **in order**. Each stage ends in a
**GATE**: if the stage did not do its job, stop and report — do not paper over a
failed stage to reach the next one.

This pipeline assumes the work is **already specced**: the GitHub issue body is
the work order (Calibre roadmap convention). `/go` does not plan. If the issue is
missing or its body is empty, stop and tell the user to spec it (or run
`/ce-plan`) first.

## Project rules that bind every stage

- **`uv run` prefix.** All Python tooling runs as `uv run pytest`,
  `uv run ruff check .`, `uv run ruff format .`, `uv run ty check calibre/`.
  Never invoke `python`/`pytest`/`ruff`/`ty` bare — here or inside spawned agents.
- **Never loosen the VN2 gate.** The winning-config baseline is `total_cost=4992.20`
  (x86_64/Linux CI). If CI goes red here, fix the root cause — never edit the
  baseline, weaken an assertion, or skip the test to make CI pass.
- **Public repo, private context stays out.** This repo is public. No client
  names, partners, or private commercial context in commit messages, the PR
  title/body, or issue comments.
- **Squash + `closes #N`.** Merge is squash-only; the PR body must carry
  `closes #N` so the issue closes and roadmap status updates for free.

---

## Stage 0 — Resolve the work item

Resolve `$ARGUMENTS` to a concrete GitHub issue:

```bash
# Accepts "123", "#123", or a roadmap code like "U3" (search open issues by title/body)
gh issue view <N> --json number,title,body,state,milestone,labels   # if numeric
gh issue list --search "<code> in:title,body" --state open --json number,title  # if a code
```

Read the full issue body — it is the spec. Note the issue number `N`, the title,
and the acceptance/verification criteria. Derive a short slug (e.g. `u3-parser-coverage`)
for branch naming and memory.

**GATE:** an open issue with a non-empty body exists. If not, stop and report.

---

## Stage 1 — Implement + open the PR (`ce-work` as a spawned agent)

Spawn **one** agent (foreground, `subagent_type: general-purpose`, no model
override — inherit) to implement the issue and open the PR. A fresh context
window keeps the heavy implementation stage clean. Give it this brief:

> Invoke the `ce-work` skill to implement GitHub issue #N. Treat the issue body
> below as the complete spec/plan — do not re-plan, do not ask to narrow scope.
>
> Setup, non-interactively (do not stop to ask which branch): from an up-to-date
> `main`, create a feature branch named `feat/<slug>` (or `fix/<slug>`). Do not
> commit to `main`.
>
> Follow repo conventions (AGENTS.md / CLAUDE.md). Run quality gates with the
> `uv run` prefix only: `uv run pytest`, `uv run ruff check .`,
> `uv run ty check calibre/`. Do NOT touch the VN2 baseline `4992.20`. Keep all
> private/commercial context out of commits and the PR.
>
> Finish by pushing the branch and opening a PR whose body includes `closes #N`,
> a ≤70-char title, a 3-bullet summary, and a test-plan checklist.
>
> Report back: PR number, PR URL, and the branch name.
>
> --- ISSUE #N SPEC ---
> <paste title + full body from Stage 0>

When the agent returns, sync the main checkout onto the PR branch (works whether
the agent ran in this checkout or an isolated one):

```bash
gh pr checkout <PR>
git pull --ff-only
```

**GATE:** a PR exists for this branch (`gh pr view --json number,url,state`) and
real code changed (`git diff main...HEAD --stat` is non-empty). If either is
missing, stop and report — do not hand-write the implementation yourself.

---

## Stage 2 — Simplify the diff (`/ce-simplify-code`, inline)

Invoke the `ce-simplify-code` skill (inline — it spawns its own three reviewers).
Scope is the branch diff vs `main`, which is correct here.

If it changes anything, it re-runs typecheck/lint/scoped tests itself. Commit and
push any resulting changes before moving on:

```bash
git add -A && git commit -m "refactor: simplify <slug>" && git push
```

**GATE:** working tree clean (committed + pushed) before Stage 3.

---

## Stage 3 — Review the PR (`/ce-code-review`, inline)

Invoke the `ce-code-review` skill (inline — it spawns its persona tiers) against
this PR. Ensure its **actionable findings land as inline PR review comments**
(resolvable threads) so Stage 4 has something to resolve — pass the PR and have
it post comments rather than only printing a report. If it applies any safe fixes
inline and commits them, push those:

```bash
git push
```

Findings that become PR review threads are handed to Stage 4. Zero findings is a
valid outcome — Stage 4 then no-ops.

**GATE:** review completed and any inline-applied fixes are pushed.

---

## Stage 4 — Resolve review feedback (`/ce-resolve-pr-feedback`, inline)

Invoke the `ce-resolve-pr-feedback` skill (inline — it spawns per-thread agents)
for this PR. It evaluates every unresolved thread (Stage 3's findings plus any
human/bot comments that arrived), fixes the valid ones, commits + pushes, then
replies and resolves each thread.

**GATE:** no unresolved review threads remain except ones it explicitly tagged
`needs-human`. If `needs-human` threads exist, surface them in the final report;
they do not block the merge unless they flag a correctness risk — use judgment.

---

## Stage 5 — Babysit CI, then squash-merge on green

Wait for CI, then enter an autofix loop (max **3** iterations):

```bash
gh pr checks --watch    # exit 0 = all green → break and merge
```

If checks fail:

1. Enumerate failures: `gh pr checks --json name,state,conclusion,link`.
2. Pull logs: `gh run view <run-id> --log-failed`.
3. Fix the **root cause** in the working tree. Never weaken an assertion, skip a
   test, or touch the VN2 `4992.20` baseline to turn CI green. If the failure is
   a genuinely flaky test with no code fix, record it rather than retrying blindly.
4. `git add <changed> && git commit -m "fix(ci): <what broke>" && git push`.
5. Re-watch.

After **3** failed cycles, stop looping: append a `## CI Failures Unresolved`
section to the PR body (`gh pr edit <PR> --body-file <tmp>`) and report. Do not merge red.

**On green** (and Stage 4 gate satisfied), confirm the PR body carries `closes #N`,
then merge:

```bash
gh pr merge <PR> --squash --delete-branch
git checkout main && git pull --ff-only
```

**GATE:** PR merged and `main` synced, OR a clear report of why it stopped short.

---

## Stage 6 — Persist outcome (vault-gated)

If `OBSIDIAN_VAULT_PATH` is set, invoke the `project-memory` skill and record the
result for the `calibre` project:

- **`plans/<slug>.md`** — if a plan/issue plan exists, append the outcome (PR URL,
  merged SHA, key decisions, status: shipped). If none exists and the change was
  non-trivial, write a brief retrospective.
- **`architecture.md`** — only if this introduced a durable design decision,
  module boundary, or invariant change. Terse; cite the PR.
- **`lessons.md`** — append any rule-for-self from a correction or pitfall hit.

Do not touch `vision.md` unless product scope actually moved. The vault is a git
repo — commit and push the vault edits. If `OBSIDIAN_VAULT_PATH` is unset, skip
this stage.

---

## Done

Report, in order: issue #N, PR URL, merged (yes + SHA / no + reason), CI result,
any `needs-human` review threads, and memory updates (or "skipped — no vault").
