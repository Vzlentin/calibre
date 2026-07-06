# Issue tracker: GitHub (hybrid with the vault)

Issues and PRDs for this repo live as GitHub issues in `Vzlentin/calibre`,
governed by a hybrid rule — **GitHub holds live status + work orders; the
Obsidian vault's `ROADMAP.md` holds durable rationale.** Don't mirror one into
the other. Use the `gh` CLI for all issue operations.

## Existing workflow (don't re-invent)

These conventions are already in force (see CLAUDE.md → *Roadmap: GitHub for
status, vault for rationale*):

- **One issue per backlog item.** The issue body holds the full
  symptom/fix/files spec — enough for an AFK agent to act without extra context.
- **Status updates for free via `closes #N`.** A merged PR that closes an issue
  updates its status automatically; prefer this over manual state edits.
- **Milestones track the active wave.** The active milestone is named in the
  vault's `ROADMAP.md` — **don't hard-code it here** (a renamed milestone is the
  drift #89 fixed). List open milestones and their issues:

  ```bash
  gh api repos/Vzlentin/calibre/milestones --jq '.[] | select(.state=="open") | .title'
  gh issue list --milestone "<title>"
  ```

- **Parked items** carry `parked:phd` or `parked:saas` and sit **outside any
  milestone** — deferred to a later track, not awaiting action.

## Standard operations

- **Create an issue**: `gh issue create --title "..." --body "..."` (heredoc for
  multi-line bodies). Put the full symptom/fix/files spec in the body.
- **Read an issue**: `gh issue view <number> --comments` (labels come with it).
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`
  with `--label` / `--state` / `--milestone` filters as needed.
- **Comment**: `gh issue comment <number> --body "..."`
- **Label**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."` — or let a merged
  `closes #N` PR do it.

`gh` infers the repo from `git remote -v` inside a clone.

## Triage state machine (extension)

The `triage` skill moves issues through five canonical roles, applied as labels
(see `docs/agents/triage-labels.md` for the mapping): `needs-triage` →
`needs-info` / `ready-for-agent` / `needs-decision` / `wontfix`. Four are new
labels the first triage run creates; `needs-decision` already exists.

These labels are a **state** axis, orthogonal to the repo's existing labels:

- **Priority** (`P0`/`P1`/`P2`) and **type** (`bug`/`enhancement`/`tech-debt`/
  `documentation`) describe *what* and *how urgent*; triage labels describe
  *what to do next*. They combine freely.
- **`needs-decision` does double duty.** The triage skill's `ready-for-human`
  role maps to the existing `needs-decision` label — in this solo workflow
  "needs a human to implement" and "needs a human/architectural decision before
  AFK work" are the same state (the same person decides and implements). So the
  triage skill applies `needs-decision` where it would otherwise apply
  `ready-for-human`, and the label keeps its original HITL decision-gate
  meaning. See `docs/agents/triage-labels.md`.
- **`wontfix` vs `parked:*`**: `wontfix` = will not be actioned at all;
  `parked:phd`/`parked:saas` = deferred to a specific later track (still alive,
  just not now). Triage leaves `parked:*` items out of its queue by design.

## Pull requests as a triage surface

**PRs as a request surface: yes.** External PRs are treated as feature requests
(an issue with attached code) and run through the same labels and states as
issues. Collaborators' in-flight PRs are left alone.

Use the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --comments` and `gh pr diff <number>` for
  the diff.
- **List external PRs for triage**: `gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments`
  then keep only `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`,
  or `NONE` (drop `OWNER`/`MEMBER`/`COLLABORATOR`).
- **Comment / label / close**: `gh pr comment`, `gh pr edit
  --add-label`/`--remove-label`, `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be
either — resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue with the full spec in the body. Apply the appropriate
triage label; assign a milestone only if the work is active-wave (milestone name
comes from the vault `ROADMAP.md` — never hard-code it).

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments` (or `gh pr view <number> --comments` if
`#N` resolves to a PR).
