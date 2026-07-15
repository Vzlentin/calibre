# Stage 1 — implementation subagent brief

Spawn **one** foreground agent (model per the Stage 0a routing classification —
routine work on the `sidekick` agent, judgment-heavy work inheriting the
frontier model) with the brief below, filling in `#N`, `<type>/<slug>`,
`<WORKDIR>`, and the pasted plan text.

---

> Invoke the `ce-work` skill to implement the plan below for GitHub issue #N.
> Treat the provided plan as the complete spec — do not re-plan, do not ask to
> narrow scope.
>
> Setup, non-interactively (do not stop to ask which branch) — use the clause for
> this run's mode (from Stage 0d):
> - **Direct mode:** from an up-to-date `main`, create the feature branch
>   `<type>/<slug>` in this checkout. Do not commit to `main`.
> - **Worktree mode:** the branch and worktree already exist. `cd` into
>   `<WORKDIR>` (the absolute worktree path) and implement on the existing
>   `<type>/<slug>` branch — do **not** create a branch, do **not** touch the
>   main checkout.
>
> You may delegate internally per the Fusion policy, via the `task` tool:
> mechanical sub-steps (test runs, scaffolding, applying already-specified
> fixes) go to `agent: sidekick`, recon to `explore`; keep every judgment
> call — spec interpretation, design, and acceptance of delegated output — to
> yourself. `sidekick` (luna) has no native auto-delegation, so internal
> delegation is manual spawn-based. After delegating, block on `job`
> completion — never busy-poll or repeatedly inspect partial output.
> Delegates never weaken tests, skip gates, or touch frozen surfaces beyond
> their brief.
>
> Run the `uv run` quality gates — `uv run pytest` (scoped), `uv run ruff check .`,
> and `uv run ty check calibre/` — in the **FOREGROUND**, blocking on each gate's
> exit before you report. Do **not** end your turn with any gate still running in
> the background: a gate still running when the turn ends is abandoned, not
> resumed — nothing picks it back up. Commit only after the gates pass green.
>
> Finish by pushing the branch and opening a PR whose body includes `closes #N`,
> a <=70-char title, a 3-bullet summary, and a test-plan checklist.
>
> Report back: PR number, PR URL, and the branch name.
>
> --- PLAN (complete spec) ---
> <paste the full path to the plan from the resolved /project-memory store>
