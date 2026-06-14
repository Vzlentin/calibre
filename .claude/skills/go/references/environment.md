# Environment — CI-status API recipe + local host-tooling traps

`/go` runs on a Windows host with Git Bash (MSYS) and Windows PowerShell 5.1
available alongside `git` and `gh`. A few local tooling behaviors are
load-bearing — each has bitten a real run. These are **host facts, not project
policy**: the fix shape generalizes to any Windows + MSYS + PS5.1 checkout.

## CI-status recipe (`gh api .../check-runs`)

Read CI status from the typed check-runs API for the exact head SHA, never from a
`--json` wrapper (Trap 2):

```bash
HEAD_SHA=$(git rev-parse HEAD)            # worktree mode: cd "$WORKDIR" first
gh api repos/Vzlentin/calibre/commits/$HEAD_SHA/check-runs \
  --jq '.check_runs[] | {name, status, conclusion}'
```

Read the verdict from `status` **and** `conclusion` together: any run with
`status != "completed"` is **pending** (keep polling, never merge); the set is
**green** only when every run is `status == "completed"` and
`conclusion == "success"`; any completed run with another conclusion (`failure`,
`cancelled`, `timed_out`, `action_required`, …) is a **failure**. For a failed
run, pull its log with `gh run view <run-id> --log-failed` (`<run-id>` parsed
from the run's `details_url`).

## Trap 1 — `git status --porcelain` reads clean on a dirty tree

The local `git` wrapper can emit a literal `ok` on a clean tree instead of the
empty output `--porcelain` is supposed to produce, so an emptiness test
misclassifies state. **Detect clean/dirty from the full `git status` text**, not
from `--porcelain` emptiness:

```bash
git status 2>&1 | grep -qE '(nothing to commit|clean)'
CLEAN=$?    # 0 => clean tree, non-zero => dirty
```

## Trap 2 — `gh pr checks --json` is wrapper-broken

The local `gh` wrapper breaks `gh pr checks --json`, so it cannot be trusted as a
CI source of truth. **Read CI via `gh api .../check-runs`** (recipe above)
instead.

## Trap 3 — MSYS `grep -F` core-dumps on multi-pattern scans (a core-dump is a NON-verdict)

Under MSYS, `grep -F` with many `-e` patterns can core-dump (exit 134) instead of
returning a grep result. The danger is silent: a crashed scanner that exits
non-zero is easy to mistake for "no matches → clean". **A scanner abort is a
NON-verdict — never a pass.** Re-run the scan deterministically with PowerShell
`Select-String -SimpleMatch`, one pattern at a time, with explicit exit-code
handling:

```powershell
$hit = $false
foreach ($p in $patterns) {
  $m = Select-String -SimpleMatch -Pattern $p -Path $bodyFile
  if ($m) { $hit = $true; $m }     # any hit => stop, never pass
}
# A non-zero / aborted scanner exit is a NON-verdict (re-run or escalate),
# never silently treated as clean.
```

> Note: the single-pattern `git status … | grep -qE` clean/dirty detector in
> `worktree-provisioning.md` (Trap 1's recipe) is a latent instance of this same
> class. Single-pattern greps have not core-dumped in practice, so it is left
> as-is for now rather than pre-emptively rewritten to PowerShell.

## Trap 4 — PowerShell 5.1 bulk source rewrite corrupts encoding

Bulk-rewriting source files through PowerShell 5.1 (`Get-Content` / `Set-Content`
or `-replace` over a whole file) re-encodes them as UTF-16LE with a BOM, which
corrupts the file for downstream tooling. **Never bulk-rewrite source via PS5.1.**
Use targeted, surgical edits (an editor / edit tool) instead of whole-file
content rewrites.
