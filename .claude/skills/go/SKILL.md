---
name: go
description: Calibre implementation-orchestration pipeline. Given a plain idea, a GitHub issue (#N / number / roadmap code), or a path to a plan file, drive it end-to-end — resolve it to a plan (invoking /ce-plan when none exists), back it with a GitHub issue, then implement → simplify → review → resolve feedback → babysit CI → squash-merge → persist. Invoke with /go <idea | issue | plan-file> to build a backlog item hands-off.
argument-hint: "[idea text, issue (#N / code), or path to a plan file]"
---

You are running the Calibre implementation-orchestration pipeline for the work
item in `$ARGUMENTS`. Execute the stages **in order**. Each stage ends in a
**GATE**: if the stage did not do its job, stop and report — do not paper over a
failed stage to reach the next one.

`/go` accepts three input kinds: a **plain idea**, a **GitHub issue** (`#N`, a
number, or a roadmap code like `U3`), or a **path to a plan file**. Before
implementing, it resolves the input to a concrete plan in the active plan store
(the Obsidian vault when `OBSIDIAN_VAULT_PATH` is set, else `docs/plans/`),
invoking `/ce-plan` on demand when no plan exists yet. It then guarantees a
backing GitHub issue so the `closes #N` merge convention keeps working, and only
then implements. The plan is the work order; the issue is the close-handle.

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

## Stage 0 — Classify the input

`$ARGUMENTS` is one of three kinds. Classify it in this order — first match wins:

1. **Plan-file** — `$ARGUMENTS` resolves to an existing `.md` file (e.g.
   `docs/plans/2026-06-06-001-feat-foo-plan.md`). Open it and check it is an
   *executable plan* — it has implementation units / phases (`### U1`,
   `## Implementation units`, etc.). If it is instead a brainstorm/ideation doc
   with no implementation units, it is **not** executable: carry it as a planning
   seed and let Stage 0.5 turn it into a plan via `/ce-plan`.
2. **Issue** — else if `$ARGUMENTS` matches `^#?\d+$` (a number or `#N`) or a
   roadmap code `^[A-Za-z]+\d+$` (e.g. `U3`), resolve it to a GitHub issue:

   ```bash
   gh issue view <N> --json number,title,body,state,milestone,labels   # if numeric
   gh issue list --search "<code> in:title,body" --state open --json number,title  # if a code
   ```

   Read the full issue body — it seeds planning. Note the issue number `N` and
   the title.
3. **Idea** — else treat `$ARGUMENTS` as free text describing the work to do.

Derive a short slug (e.g. `u3-parser-coverage`) from the chosen artifact for
branch naming and memory.

**GATE:** the input resolved to exactly one of {plan-file, issue, idea}. If it is
ambiguous — a `.md` path that does not exist, or a number/code that resolves to
no issue — stop and ask.

---

## Stage 0.5 — Ensure a plan exists in the active store

**Resolve the plan store first.** When `OBSIDIAN_VAULT_PATH` is set, the store is
the vault: `$OBSIDIAN_VAULT_PATH/Projects/Calibre/plans/`. When it is unset, the
store is repo `docs/plans/` — there is no private store, so nothing private can
enter the plan, the public repo is the canonical store, and the pipeline still
runs end-to-end. Every search and write below targets the active store.

Get a plan in hand, by input kind:

- **Plan-file (executable):** that file is the plan. Skip to Stage 0.6.
- **Plan-file (brainstorm) / Issue / Idea:** find an existing plan, else create one.
  - **Issue:** search the active store for a plan whose frontmatter `origin:` or
    body references `#N`:

    ```bash
    grep -rl "#<N>" "$OBSIDIAN_VAULT_PATH/Projects/Calibre/plans"   # vault store
    grep -rl "#<N>" docs/plans                                       # degraded store
    ```

    If found, use it. Else invoke `/ce-plan` inline, seeded with the issue title + body.
  - **Idea / brainstorm doc:** keyword/slug search the store's plan titles and
    filenames. On a *plausible* match, confirm with the user before reusing it (a
    wrong reuse is worse than a fresh plan). On no match, invoke `/ce-plan`
    inline, seeded with the idea text (or the brainstorm doc's contents).

**Invoking `/ce-plan`.** Run it **inline** (not as a spawned agent) — planning is
interactive and its scoping/clarifying gates need the user; Stage 1's `ce-work`
stays spawned, but planning must reach the user. `/ce-plan` writes natively to
`docs/plans/YYYY-MM-DD-NNN-<type>-<name>-plan.md` (it takes no output-path arg).
When its post-generation menu appears, select **"Done for now"** so `/go` keeps
control of issue creation and implementation — do **not** let `/ce-plan` start
`ce-work` or create the issue itself.

**Relocate into the vault (only when the vault is set).** After `/ce-plan`
returns, if the vault is the active store, move its `docs/plans/` output into the
vault: read the file, rewrite the frontmatter to the vault convention (`title`,
`type` (`feat|fix|refactor|chore`), `status: active`, `date`, and `origin` once
Stage 0.6 sets it), write it to
`$OBSIDIAN_VAULT_PATH/Projects/Calibre/plans/<YYYY-MM-DD-slug>.md`, verify the
write, then delete the `docs/plans/` copy — one source of truth, and no private
context in the public repo. When the vault is unset, **skip relocation**: the
plan stays in `docs/plans/`.

**GATE:** a plan file for this work item exists in the active store (vault if
set, else `docs/plans/`). If `/ce-plan` produced nothing, stop and report.

---

## Stage 0.6 — Ensure a backing GitHub issue

Every run merges with `closes #N`, so guarantee an issue exists:

- **Reuse** when the work item already has one — the **Issue** input's `#N`, or a
  plan carrying `origin: "GitHub issue #N — …"`. Do not create a duplicate.
- **Create** otherwise. Open a GitHub issue with a ≤70-char title and a
  **public-safe** body that summarizes the plan — no client names, partners, or
  private commercial context. Vault plans may carry private/commercial context;
  the full plan stays in the vault and only the public-safe summary becomes the
  issue body. In degraded mode (vault unset) the plan is already fully public, so
  the body may be derived from it directly without stripping. Capture the new
  number `N`:

  ```bash
  gh issue create --title "<≤70-char title>" --body-file <public-safe summary>
  ```

Record `origin: "GitHub issue #N — <short>"` in the plan's frontmatter (in the
active store). Milestone/label triage is left to the user (out of scope here).

**GATE:** a usable `#N` exists and is recorded on the plan. If issue creation
failed, stop and report.

---

## Stage 1 — Implement + open the PR (`ce-work` as a spawned agent)

Spawn **one** agent (foreground, `subagent_type: general-purpose`, no model
override — inherit) to implement the plan and open the PR. A fresh context
window keeps the heavy implementation stage clean. Give it this brief:

> Invoke the `ce-work` skill to implement the plan below for GitHub issue #N.
> Treat the pasted plan as the complete spec — do not re-plan, do not ask to
> narrow scope.
>
> Setup, non-interactively (do not stop to ask which branch): from an up-to-date
> `main`, create a feature branch named `feat/<slug>` (or `fix/<slug>`). Do not
> commit to `main`.
>
> Follow repo conventions (AGENTS.md / CLAUDE.md). Run quality gates with the
> `uv run` prefix only: `uv run pytest`, `uv run ruff check .`,
> `uv run ty check calibre/`. Do NOT touch the VN2 baseline `4992.20`.
>
> **Private-context guard.** The spec below may itself contain private or
> commercial context. Keep all of it — client names, partners, commercial
> framing — out of every commit message, code comment, file, branch name, the PR
> title/body, and any issue comment. Implement only the public-safe behavior the
> plan describes.
>
> Finish by pushing the branch and opening a PR whose body includes `closes #N`,
> a ≤70-char title, a 3-bullet summary, and a test-plan checklist.
>
> Report back: PR number, PR URL, and the branch name.
>
> --- PLAN (complete spec) ---
> <paste the full plan contents from the active store — paste the text, do not
> pass a path; an isolated worktree may not have the vault mounted>

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

**On green** (and Stage 4 gate satisfied), confirm the PR body carries `closes #N`
— Stage 0.6 guarantees the issue exists, so verify only that the agent actually
included the line — then merge:

```bash
gh pr merge <PR> --squash --delete-branch
git checkout main && git pull --ff-only
```

**GATE:** PR merged and `main` synced, OR a clear report of why it stopped short.

---

## Stage 6 — Persist outcome

**Flip the plan to shipped (always).** In the active store (vault if set, else
`docs/plans/`), update the plan file for this work item: set
`status: active → shipped` and append the outcome — PR URL, merged SHA, and any
key decisions.

- **Vault set:** this is a vault edit. Also invoke the `project-memory` skill for
  the `calibre` project where warranted:
  - **`architecture.md`** — only if this introduced a durable design decision,
    module boundary, or invariant change. Terse; cite the PR.
  - **`lessons.md`** — append any rule-for-self from a correction or pitfall hit.

  Do not touch `vision.md` unless product scope actually moved. The vault is a git
  repo — commit and push the vault edits (the plan status flip plus any memory
  updates).
- **Vault unset (degraded):** the plan lives in `docs/plans/` in this repo.
  Commit the `status: shipped` flip (a docs-only change) and push. There is no
  vault, so skip `architecture.md` / `lessons.md` / `project-memory`.

**GATE:** the work item's plan in the active store reads `status: shipped` with
the PR/SHA recorded.

---

## Done

Report, in order: the resolved **input kind** (idea / issue / plan-file) and the
**plan path** in the active store; issue #N; PR URL; merged (yes + SHA / no +
reason); CI result; any `needs-human` review threads; and memory updates (plan
status flip, plus architecture/lessons or "skipped — no vault").
