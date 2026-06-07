---
name: go
description: Calibre implementation-orchestration pipeline. Given a plain idea, a GitHub issue (#N / number / roadmap code), or a path to a plan file, drive it end-to-end — resolve it to a plan (invoking /ce-plan when none exists), back it with a GitHub issue, then implement → simplify → review → resolve feedback → babysit CI → squash-merge → persist. Invoke with /go <idea | issue | plan-file> to build a backlog item hands-off.
argument-hint: "[idea text, issue (#N / code), or path to a plan file]"
---

You are running the Calibre implementation-orchestration pipeline for the work
item in `$ARGUMENTS`. Execute the stages **in order**. Every run ends in exactly
one **terminal outcome** — `shipped` (PR squash-merged) or `failed` (a stage
stopped short) — persisted to the plan store by Stage 6.

Each stage ends in a **GATE**: if the stage did not do its job, stop — do not
paper over a failed stage to reach the next one. A short-stopping GATE **is** the
`failed` outcome: it routes to Stage 6 to persist `failed` (when a plan already
exists), then reports.

`/go` accepts three input kinds — a **plain idea**, a **GitHub issue** (`#N`,
number, or roadmap code like `U3`), or a **path to a plan file**. It resolves the
input to a concrete plan in the store resolved by `/project-memory` (the vault,
or the `docs/plans/` fallback; invoking `/ce-plan` when none exists), guarantees
a backing issue so `closes #N` keeps working, then implements. The plan is the
work order; the issue is the close-handle.

## Project rules that bind every stage

- **Storage is delegated.** The plan store and outcome persistence are resolved by
`/project-memory`.
- **Public repo, private context stays out.** This repo is public. No client
  names, partners, or private commercial context in commit messages, the PR
  title/body, or issue comments.
- **Squash + `closes #N`.** Merge is squash-only; the PR body must carry
  `closes #N` so the issue closes and roadmap status updates for free.
- **Two local-tooling traps.** Don't trust `git status --porcelain` emptiness
  (the local `git` wrapper may emit `ok` on a clean tree) and don't use
  `gh pr checks --json` (the `gh` wrapper breaks `--json`). Parse the full
  `git status` text for clean/dirty, and read CI via the GitHub API. See
  `.claude/skills/go/references/environment.md` for both traps in full and the
  canonical `gh api` check-runs/status recipes — its single home.

---

## Invocation model

Each stage delegates to a skill-primitive; how it's invoked depends on who owns
the fan-out:

- **Subagent (fresh context window):** `ce-work` — a heavy, self-contained
  stage whose context should stay out of the orchestrator.
- **Inline (the `/go` agent runs it directly):** `ce-plan`, `ce-simplify-code`,
  `ce-code-review`, `ce-resolve-pr-feedback`. `ce-plan` runs inline so its
  clarifying gates can reach you (a subagent can't ask questions); the others
  each fan out their own subagents, so `/go` runs them itself to own that
  fan-out rather than nest it inside another subagent.

---

## Stage 0a— Classify the input

Classify `$ARGUMENTS` into one of three kinds, in this order — first match
wins:

1. **Plan-file** — `$ARGUMENTS` resolves to an existing `.md` file (e.g.
   `2026-06-06-001-feat-foo-plan.md`). Open it and check it is an
   *executable plan* — it has implementation units / phases (`### U1`,
   `## Implementation units`, etc.). A brainstorm/ideation doc with no
   implementation units is **not** executable: carry it as a planning seed for
   Stage 0b to turn into a plan via `/ce-plan`.
2. **Issue** — else if `$ARGUMENTS` matches `^#?\d+$` (number or `#N`) or a
   roadmap code `^[A-Za-z]+\d+$` (e.g. `U3`), resolve it to a GitHub issue:

   ```bash
   gh issue view <N> --json number,title,body,state,milestone,labels   # if numeric
   gh issue list --search "<code> in:title,body" --state open --json number,title  # if a code
   ```

   Read the full issue body — it seeds planning. Note the number `N` and title.
3. **Idea** — else treat `$ARGUMENTS` as free text describing the work to do.

Derive a short slug (e.g. `u3-parser-coverage`) from the chosen artifact for
branch naming and memory.

**GATE:** the input resolved to exactly one of {plan-file, issue, idea}. If it is
ambiguous — a `.md` path that does not exist, or a number/code that resolves to
no issue — stop and ask.

---

## Stage 0b — Ensure a plan exists

- **Plan-file (executable):** that file is the plan — skip to Stage 0c.
- **Issue / Idea / brainstorm (or non-executable plan-file):** find an existing
  plan in the resolved store, else create one via `/ce-plan`:
  - **Issue:** search the plan store for a plan whose `origin:` or body
    references `#N`. If found, use it; else seed `/ce-plan` with the issue title
    + body.
  - **Idea / brainstorm:** keyword/slug search plan titles + filenames. On a
    *plausible* match, **confirm with the user before reusing** (a wrong reuse is
    worse than a fresh plan); on no match, seed `/ce-plan` with the idea text /
    the brainstorm's full contents. A brainstorm is **never** executable on its
    own and is **never** fed to Stage 1 as the spec — it must become a plan first.

**Invoking `/ce-plan`.** Always run `/ce-plan` **inline** so its
scoping/clarifying gates reach you (a spawned subagent can't ask questions). At
its post-generation menu pick **"Done for now"** so `/go` keeps control of issue
creation + implementation — do **not** let it start `ce-work` or create the
issue itself.

`/ce-plan` writes to `docs/plans/YYYY-MM-DD-NNN-<type>-<name>-plan.md`;
**`/project-memory` places it at the resolved store** — relocated into the vault
in vault mode, left in `docs/plans/` in fallback mode. Delegate placement there
rather than moving files here.

**GATE:** a plan file for this work item exists in the resolved store. If
`/ce-plan` produced nothing, stop and report.

---

## Stage 0c — Ensure a backing GitHub issue

Every run merges with `closes #N`, so guarantee an issue exists:

- **Reuse** when the work item already has one — the **Issue** input's `#N`, or a
  plan carrying `origin: "GitHub issue #N — …"`. Do not create a duplicate.
- **Create** otherwise. Open an issue with a ≤70-char title and a **public-safe**
  body summarizing the plan — no client names, partners, or private commercial
  context (the full plan stays in the plan store; only the public-safe summary
  becomes the issue body). Capture the new number `N`:

  ```bash
  gh issue create --title "<≤70-char title>" --body-file <public-safe summary>
  ```

Record `origin: "GitHub issue #N — <short>"` in the plan's frontmatter.

**GATE:** a usable `#N` exists and is recorded on the plan. If issue creation
failed, stop and report.

---

## Stage 0d — Choose execution location and provision

Stages 1–5 run against a working directory `WORKDIR`, picked with a smart-worktree
gate so the user's current checkout — their branch *and* any uncommitted work —
is never disturbed. Read git state in the main checkout and decide the mode — see
`.claude/skills/go/references/worktree-provisioning.md` for the git-state read,
the provisioning bash, and the worktree caveats:

- **On `main` AND clean** → **DIRECT mode**: `WORKTREE_MODE=false`,
  `WORKDIR="$MAIN"`. No provisioning — `ce-work` branches inside this checkout.
- **Any other branch, detached HEAD, OR dirty tree** → **WORKTREE mode**:
  `WORKTREE_MODE=true`. Provision an isolated worktree on a fresh branch cut from
  `origin/main`, so neither the user's branch nor their dirty tree moves. The
  setup steps are read dynamically from `.cursor/worktrees.json` (per the
  reference) so config changes are picked up; `<type>` is the kind `ce-work`
  would choose, `<slug>` is the Stage 0a slug.

`WORKDIR` is where Stages 1–5 operate. From here on, every shell command for those
stages uses an explicit `cd "$WORKDIR" && …` in worktree mode (a
`working_directory` arg can silently target the main repo); in direct mode the
`cd` is a harmless no-op. Stage 6 is the exception — it always runs from `$MAIN`.

**GATE (worktree mode):** the worktree exists (`git worktree list` shows
`.worktrees/<slug>`) and its venv is usable
(`cd "$WORKDIR" && uv run python -c "import calibre"`). If provisioning failed,
stop and report — do **not** fall back to mutating the user's dirty checkout. In
direct mode this gate is automatically satisfied.

---

## Stage 1 — Implement + open the PR (`ce-work` as a spawned agent)

Spawn **one** agent (foreground, no model override — inherit) to implement the
plan and open the PR — a subagent per the Invocation model. Give it the brief in
`.claude/skills/go/references/ce-work-brief.md` (mode-specific setup clauses,
`uv run` quality gates, the private-context guard, the `closes #N` PR finish),
filling in `#N`, `<type>/<slug>`, `<WORKDIR>`, and the **pasted** plan text.

When the agent returns, sync by mode — never move the main checkout onto the PR
branch in worktree mode:

- **Direct mode:** the agent branched in this checkout — check out the PR branch
  here and fast-forward:

```bash
gh pr checkout <PR>
git pull --ff-only
```

- **Worktree mode:** leave the main checkout where the user left it (their branch
  / dirty tree). The worktree is already on `<type>/<slug>`; just fast-forward it:

```bash
cd "$WORKDIR" && git pull --ff-only
```

**GATE:** a PR exists for this branch (`gh pr view --json number,url,state`) and
real code changed (`cd "$WORKDIR" && git diff main...HEAD --stat` is non-empty).
If either is missing, stop and report — do not hand-write the implementation
yourself.

---

## Stage 2 — Simplify the diff (`ce-simplify-code`, inline)

Invoke the `ce-simplify-code` skill from `WORKDIR` (inline — see Invocation
model; it spawns its own subagents). Scope is the branch diff vs `main`. If it
changes anything, it re-runs typecheck/lint/scoped tests itself; commit and push
the result before moving on:

```bash
cd "$WORKDIR" && git add -A && git commit -m "refactor: simplify <slug>" && git push
```

**GATE:** working tree clean (committed + pushed) before Stage 3.

---

## Stage 3 — Review the PR (`ce-code-review`, inline)

Invoke the `ce-code-review` skill from `WORKDIR` (inline — see Invocation model;
it spawns its own subagents) against this PR. Ensure its **actionable findings
land as inline PR review comments** (resolvable threads) so Stage 4 has something
to resolve — pass the PR and have it post comments rather than only printing a
report. Push any safe fixes it commits inline:

```bash
cd "$WORKDIR" && git push
```

Findings that become PR review threads are handed to Stage 4. Zero findings is a
valid outcome — Stage 4 then no-ops.

**GATE:** review completed and any inline-applied fixes are pushed.

---

## Stage 4 — Resolve review feedback (`ce-resolve-pr-feedback`, inline)

Invoke the `ce-resolve-pr-feedback` skill from `WORKDIR` (inline — see Invocation
model; it spawns its own subagents) for this PR. It evaluates every unresolved
thread (Stage 3's findings plus any human/bot comments that arrived), fixes the
valid ones in `WORKDIR`, commits + pushes, then replies and resolves each thread.

**GATE:** no unresolved review threads remain except ones it explicitly tagged
`needs-human`. Surface any `needs-human` threads in the final report; they do not
block the merge unless they flag a correctness risk — use judgment.

---

## Stage 5 — Babysit CI, then squash-merge on green

Wait for CI, then enter an autofix loop (max **3** iterations). Poll the PR's head
SHA with the canonical `gh api` check-runs/status recipes in
`.claude/skills/go/references/environment.md`; the CI log-pull and the
cleanup-by-mode bash live in `.claude/skills/go/references/ci-and-merge.md`.

If checks fail:

1. Enumerate failures from the check-runs output.
2. Pull logs for the failed runs (recipe in `ci-and-merge.md`).
3. Fix the **root cause** in `WORKDIR`. Never weaken an assertion, skip a test,
   or touch the VN2 `4992.20` baseline to turn CI green. If a failure is a
   genuinely flaky test with no code fix, record it rather than retrying blindly.
   **If the same failure signature appears across 2+ iterations, stop** — the fix
   round is introducing its own failures (the PR #38 pattern); don't burn the last
   cycle.
4. `cd "$WORKDIR" && git add <changed> && git commit -m "fix(ci): <what broke>" && git push`.
5. Re-check.

After **3** failed cycles, stop looping: append a `## CI Failures Unresolved`
section to the PR body (`gh pr edit <PR> --body-file <tmp>`) and report. Do not
merge red — take the **preserve path** below.

**On green** (and Stage 4 gate satisfied), confirm the PR body carries `closes #N`
— Stage 0c guarantees the issue exists, so verify only that the line is present
— then squash-merge (this also deletes the remote branch):

```bash
gh pr merge <PR> --squash --delete-branch
```

**Cleanup is merge-gated** — run it only after that command confirms the
squash-merge, using the cleanup-by-mode bash in `ci-and-merge.md`. A squash-merged
branch never shows as "merged" to git, so the local branch must be force-deleted
with `git branch -D` (not `-d`). In **direct mode** return to `main`,
fast-forward, drop the branch; in **worktree mode** remove the worktree and drop
the branch **without** `git checkout main`/`git pull` in the main checkout —
preserving the user's branch and dirty tree is the whole point.

**Preserve path (failure / any short-stop).** If CI is still red after 3 cycles,
or any stage stopped short of a confirmed merge, do **not** clean up: leave the
local `<type>/<slug>` branch and — in worktree mode — the `.worktrees/<slug>`
working tree intact so the user can resume/debug, and surface the worktree path +
branch in the final report. This short-stop **is** the `failed` terminal outcome
— proceed to Stage 6 to persist `failed` before reporting.

**GATE:** either the PR is squash-merged **and** cleanup ran (local branch deleted
in both modes; worktree removed in worktree mode) with `main` fast-forwarded; OR a
clear report of why it stopped short, naming the preserved branch (and worktree
path, if any).

---

## Stage 6 — Persist terminal outcome

Stage 6 always runs from the main checkout (`$MAIN`), never from `WORKDIR` — by now
the worktree may be removed by Stage 5 cleanup, and persistence is independent of
execution mode. **Delegate persistence to `/project-memory`** and flip the plan to
exactly one terminal status:

- **shipped** — the PR squash-merged: flip the plan's `status: active → shipped` in
  the resolved store and append the outcome — PR URL, merged SHA, key decisions.
- **failed** — a GATE stopped short: flip the plan's `status: active → failed` in
  the resolved store and append the failing stage, the reason, and the preserved
  branch / worktree path.
- **Edge case — failure before a plan exists.** A short-stop in Stage 0a/0b has no
  plan to flip — just report `failed`.

**GATE:** the work item's plan reads exactly one of `status: shipped` (with the
PR/SHA recorded) or `status: failed` (with the failing stage + reason) in the
resolved store — or, when the run failed before a plan existed, the report states
`failed`.

---

## Done

Lead with the **terminal outcome** — `shipped` or `failed`; on `failed`, name the
failing stage + reason. Then report, in order: the resolved **input kind** (idea /
issue / plan-file) and the **plan path** in the resolved store; the **execution
mode** (direct / worktree, and on the preserve path the retained branch and — in
worktree mode — the worktree path); issue #N; PR URL; merged (yes + SHA / no +
reason); CI result; any `needs-human` review threads; and memory updates (plan
status flip, plus architecture/lessons or "skipped").
