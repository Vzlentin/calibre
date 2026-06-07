# Stage 1 — `ce-work` subagent brief

Spawn **one** foreground agent (no model override — inherit) with the brief
below, filling in `#N`, `<type>/<slug>`, `<WORKDIR>`, and the pasted plan text.

---

> Invoke the `ce-work` skill to implement the plan below for GitHub issue #N.
> Treat the pasted plan as the complete spec — do not re-plan, do not ask to
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
> Finish by pushing the branch and opening a PR whose body includes `closes #N`,
> a <=70-char title, a 3-bullet summary, and a test-plan checklist.
>
> Report back: PR number, PR URL, and the branch name.
>
> --- PLAN (complete spec) ---
> <paste the full plan contents from the resolved /project-memory store>
