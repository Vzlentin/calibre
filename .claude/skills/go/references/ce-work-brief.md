# Stage 1 — `ce-work` subagent brief

Spawn **one** foreground agent (no model override — inherit) with the brief
below, filling in `#N`, `<type>/<slug>`, `<WORKDIR>`, and the pasted plan text.

---

> Invoke the `ce-work` skill to implement the plan below for GitHub issue #N.
> Treat the pasted plan as the complete spec — do not re-plan, do not ask to
> narrow scope.
>
> Setup, non-interactively (do not stop to ask which branch) — use the clause for
> this run's mode (from Stage 0.7):
> - **Direct mode:** from an up-to-date `main`, create the feature branch
>   `<type>/<slug>` in this checkout. Do not commit to `main`.
> - **Worktree mode:** the branch and worktree already exist. `cd` into
>   `<WORKDIR>` (the absolute worktree path) and implement on the existing
>   `<type>/<slug>` branch — do **not** create a branch, do **not** touch the
>   main checkout.
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
> a <=70-char title, a 3-bullet summary, and a test-plan checklist.
>
> Report back: PR number, PR URL, and the branch name.
>
> --- PLAN (complete spec) ---
> <paste the full plan contents from the vault — paste the text, do not pass a
> path; an isolated worktree does not have the vault mounted>
