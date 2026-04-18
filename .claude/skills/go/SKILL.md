---
name: go
description: End-to-end ship sequence. Runs the full test suite, simplifies changed code, then opens a pull request. Invoke with /go when a feature is ready to ship.
---

You are executing the full ship sequence for this branch. Complete the steps below in order, stopping immediately if any step fails.

## Step 1 — Run the full test suite

Identify and run the project's test suite and linting/type-checking commands (check CLAUDE.md, package.json, pyproject.toml, Makefile, or similar for the right commands).

Run all quality gates. If **any** command exits non-zero:
- Show the failing output.
- Fix the root cause (do not suppress errors, do not use `--no-verify` or bypass hooks).
- Re-run the failing command to confirm it passes before moving on.
- Do NOT proceed to Step 2 until everything is green.

## Step 2 — Simplify changed code

Invoke the `simplify` skill now:

```
/simplify
```

Wait for it to complete. If it makes changes, re-run the linting and type-checking commands to confirm nothing regressed.

## Step 3 — Open a pull request

1. Check what commits are ahead of the base branch:
   ```bash
   git log main..HEAD --oneline
   git diff main...HEAD --stat
   ```

2. Push the branch if not already pushed:
   ```bash
   git push -u origin HEAD
   ```

3. Create the PR with a concise title (under 70 chars) and a body covering:
   - Summary (3 bullets max)
   - Test plan (checklist of what was verified)

4. Return the PR URL to the user.

## Done

Report: tests passed, simplification applied (or "no changes needed"), PR URL.
