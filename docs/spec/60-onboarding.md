---
title: "Onboarding — the first-contributor path"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 60 — Onboarding

This chapter is the first-contributor path: the **first brick** — one small,
real, self-contained module a new contributor implements end-to-end from spec
chapters alone — and the first runnable test they make green. It is the
concrete answer to "where do I start?" and it doubles as the standalone-
readability check for the whole spec (chapter 00): if the brick is buildable
from its named chapters without the old repo, the private annex, or a
maintainer, the spec reads on its own.

## Name the first brick: the seasonal-naive forecasting adapter

The first brick is a **seasonal-naive forecasting adapter** — the simplest
member of the classical-statistical family the baseline adapter set declares
(chapter 04, `[REG-4]`). It forecasts each horizon step from the same seasonal
phase of the last full season before the origin. For season length `m`, step
`h` targets the period `origin + (h − 1)` (chapter 02 `[FRA-1]`), and the
forecast is the history value at `origin − m + ((h − 1) mod m)` — the most
recent observation sharing that step's phase; for a horizon within one season
(`H ≤ m`, the fixture case) this is simply the value one season earlier. No
fitting math, no external library call, no randomness — the whole model is a
lookup into the task's own history, always into the last full pre-origin
season (`origin − m … origin − 1`, all strictly pre-origin per `[TSK-2]`). A
task whose history is shorter than one season, or missing the looked-up
period, fails loudly at `predict` (the `[ADA-5]`/`[ADA-6]` loud-failure rule)
rather than emitting a partial or fabricated frame.

It qualifies against all three selection criteria:

1. **Spec-sufficient.** Everything it needs lives in exactly two chapters:
   chapter 02 supplies the vocabulary and the forecast-frame column contract
   (`[FRA-1]`–`[FRA-3]`: the typed columns, the full row key
   `(series key, origin, horizon step, model name)`, and schema validation),
   and chapter 04 supplies the adapter protocol it implements — construct from
   a model configuration, expose `fit`/`predict`, emit one row per
   `(series key, horizon step)` for each series in the task, register under a
   backend identifier (`[REG-1]`), stay task-closed (`[ADA-1]`) and scope-blind
   (`[SCO-2]`). Nothing from chapters 03, 05–12, or any annex is required.
2. **Seam-free.** It emits point forecasts only; it declares none of the
   optional capabilities (`[ADA-5]`), touches no calibration, reconciliation,
   ordering, or scoring surface, and depends on no `40-gated-seams/` ruling
   and no open invalidation tag. Its behavior is fully determined by chapters
   02 and 04 as they stand.
3. **Day-scale.** The model is a shift-by-`m` lookup; the work is the adapter
   protocol wiring, not the forecasting. A contributor new to the codebase
   reaches a green test in under a day.

## Read exactly three files first

The first-contributor path reads three files before writing code, in order:

`00-overview.md` → `02-domain-model.md` → `04-forecasting-plugins.md`.

`00` gives the map and the two-layer contract; `02` gives the terms
(series, panel, forecast task, origin, horizon, forecast frame) and the frame
column contract the adapter must emit; `04` gives the protocol the brick
implements. If the brick turns out to need anything outside these three files,
that is a defect in the brick or in one of the three chapters — never in the
reader (chapter 00's standalone-readability rule).

## Build it — the acceptance walkthrough

From the three chapters alone, the contributor:

1. **Reads the frame contract**: chapter 02 for the exact typed columns
   `predict` must return and schema validation (`[FRA-1]`–`[FRA-3]`), and
   chapter 04 for the predict-output form — one row per
   `(series key, horizon step)` for each series in the task.
2. **Implements the adapter** against chapter 04's protocol: a class
   constructed from a model configuration carrying the season length `m`,
   with `fit(task)` recording only the per-series history needed for the
   lookup and `predict(task)` emitting the seasonal-naive frame. `predict`
   before a successful `fit` is an error, not an empty result (`[ADA-6]`).
   The adapter is scope-blind: the same class serves a one-series task and a
   many-series task, branching on neither (`[SCO-2]`).
3. **Registers the backend** under an explicit identifier so a model
   configuration can name it (`[REG-1]`); no default backend.
4. **Normalizes output** to the frame column contract regardless of internal
   shape (`[ADA-3]`), and makes the result deterministic given the same task
   and configuration (`[ADA-2]`).

The walkthrough consults no old-repo path and resolves no annex pointer.

## Make the first test green

The first runnable test combines chapter 04's predict-output contract with
chapter 02's frame validation (`[FRA-3]`), specialized to seasonal-naive:

> Given a fixture panel with a known season length `m` and a chosen origin,
> `predict` returns a forecast frame that (a) passes the chapter 02 frame
> schema (`[FRA-3]`) — one row per `(series key, horizon step)`, steps
> `1..H`, point column typed float — and (b) carries, at each step `h`, the
> history value at `origin − m + ((h − 1) mod m)` (the season-lagged value of
> the period step `h` targets, `origin + (h − 1)`).

It is hand-checkable: with `m = 7` and a short daily fixture, step 1 targets
the origin day and reads the value exactly one week (seven days) earlier —
same weekday — and every expected value is read straight off the history by
eye. The natural second test is
the lifecycle guard (`[ADA-6]`): `predict` before `fit` raises a capability/
lifecycle error rather than returning empty rows. Passing the first test is
the onboarding bar; the second confirms the protocol edges.

## After the first brick

With the seasonal-naive adapter green, the reading order opens up along the
contributor's interest: a second forecasting family reads only more of
chapter 04; the calibration surface adds chapter 05; the ordering surface adds
chapters 08 and 02's decision vocabulary. The gated seam surfaces
(`40-gated-seams/`) are deliberately last — nothing on the first-contributor
path depends on them.

## Provenance

None. This chapter restates the gate structure recorded in `00-overview.md`
and the adapter contract of chapter 04; it consumes no old-engine behavior.
The seasonal-naive baseline mirrors the classical-statistical family the old
repo's `statsforecast` adapter wrapped, cited there as provenance only.
