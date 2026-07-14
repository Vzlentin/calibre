# Environment — local host-tooling traps

`/go` runs on a Windows host with Git Bash (MSYS) and Windows PowerShell 5.1
available alongside `git` and `gh`. These are **host facts, not project
policy**: each trap below bit a real run, and the fix shape generalizes to any
Windows + MSYS + PS5.1 checkout.

## Traps 1–3 — absorbed by the skill scripts

Three traps are now bypassed by construction, because the scripts in
`.agents/skills/go/scripts/` call `git`/`gh` via Python subprocess and parse
JSON natively — no MSYS wrappers, no shell `grep` in the verdict path:

- **Trap 1 (`git status --porcelain` reads clean on a dirty tree):** the MSYS
  `git` wrapper could emit a literal `ok` on a clean tree, breaking emptiness
  tests. `provision_worktree.py decide` reads porcelain output through
  subprocess, where the wrapper is not in the path.
- **Trap 2 (`gh pr checks --json` is wrapper-broken):** CI status is read from
  the typed `gh api .../check-runs` endpoint by `ci_verdict.py`, never a
  `--json` wrapper.
- **Trap 3 (MSYS `grep -F` core-dumps on multi-pattern scans):** a crashed
  scanner exiting non-zero looks like "no matches → clean" — a NON-verdict
  mistaken for a pass. The verdict path no longer shells out to `grep`;
  `ci_verdict.py` treats any malformed/failed read as a distinct non-verdict
  exit code, never green. For any *remaining* ad-hoc multi-pattern scan (e.g.
  private-context checks over a PR body), the rule stands: a scanner abort is
  a NON-verdict — re-run deterministically (PowerShell
  `Select-String -SimpleMatch`, one pattern at a time) rather than treating it
  as clean.

## Trap 4 — PowerShell 5.1 bulk source rewrite corrupts encoding

Bulk-rewriting source files through PowerShell 5.1 (`Get-Content` / `Set-Content`
or `-replace` over a whole file) re-encodes them as UTF-16LE with a BOM, which
corrupts the file for downstream tooling. **Never bulk-rewrite source via PS5.1.**
Use targeted, surgical edits (an editor / edit tool) instead of whole-file
content rewrites. This trap governs the agent's own edits and is not absorbed
by any script.
