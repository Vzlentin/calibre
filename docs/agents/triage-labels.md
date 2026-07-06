# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those
roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `needs-decision`     | Needs a human — implementation or a decision |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the
corresponding label string from this table.

## The `ready-for-human` → `needs-decision` mapping

The `ready-for-human` role reuses the repo's **existing** `needs-decision` label
rather than creating a new one. In this solo workflow the same person decides
*and* implements, so "needs a human to implement" and "needs a human/
architectural decision before AFK work" reduce to the same state — hand it back
to the human. `needs-decision` therefore does double duty: the triage skill
applies it for the `ready-for-human` role, and it remains the HITL decision-gate
signal it already was.

Four labels are new (`needs-triage`, `needs-info`, `ready-for-agent`,
`wontfix`); the first triage run creates them. They are orthogonal to the
repo's priority (`P0`/`P1`/`P2`) and type (`bug`/`enhancement`/`tech-debt`/
`documentation`) labels, and distinct from `parked:phd`/`parked:saas` (deferred
to a later track, not "needs a human now"). See `docs/agents/issue-tracker.md`
for how triage state interacts with the rest of the workflow.
