# Sequential-adaptive parity reference research

Context: [Select the sequential-adaptive parity reference](https://github.com/Vzlentin/calibre/issues/400).

## Recommendation

Use the ACI implementation in Angelopoulos et al., *Conformal PID Control for Time-Series Prediction*:

- repository: [`aangelopoulos/conformal-time-series`](https://github.com/aangelopoulos/conformal-time-series)
- immutable commit: [`b729c3f5ff633bfc43f0f7ca08199b549c2573ac`](https://github.com/aangelopoulos/conformal-time-series/commit/b729c3f5ff633bfc43f0f7ca08199b549c2573ac)
- source: [`core/methods.py`, lines 76–111](https://github.com/aangelopoulos/conformal-time-series/blob/b729c3f5ff633bfc43f0f7ca08199b549c2573ac/core/methods.py#L76-L111)
- license: MIT

It is paper-linked, exposes ACI over a score stream, fixes the quantile method to `higher` on the adaptive branch, and is small enough to mint a deterministic trace without retaining a runtime dependency.

## Reference semantics

The pinned source:

1. initializes the controller state at the target alpha;
2. switches to the adaptive branch only when `t_pred > T_burnin`;
3. emits an infinite threshold when the raw adaptive alpha is at or below `1 / (t_pred + 1)`;
4. clips alpha only when forming the quantile level;
5. selects the adaptive threshold with NumPy's `higher` quantile method;
6. leaves the raw controller state unclipped after the update;
7. updates by `alpha_t = alpha_t - learning_rate * gradient`, equivalent to feedback toward the target error rate.

These are reference facts. A separate design decision must state which become successor method policy rather than parity-fixture behavior.

## Trace to mint

Use a small non-negative score stream with deliberate hits, misses, rank changes, and alpha-boundary behavior. A suitable starting fixture is:

```json
{
  "scores": [1, 4, 2, 8, 3, 9, 0, 7, 5, 6, 10, 2],
  "alpha": 0.25,
  "learning_rate": 0.125,
  "window_length": 5,
  "burn_in": 3,
  "ahead": 1,
  "quantile_method": "higher"
}
```

Capture at every step:

- prediction index and score-window boundaries;
- alpha before and after feedback;
- clipped quantile level;
- finite/non-finite branch;
- selected rank and threshold;
- covered/error indicator.

Persist finite floats as binary64 hexadecimal strings. Bind the trace to the source URL and commit, generator version, Python and NumPy versions, and output digest. The third-party package is needed only to mint/review the trace, never to run the permanent test.

## Parity contract

Require exact agreement for ordering, window boundaries, branch selection, selected rank, threshold, and covered/error indicators. The `higher` threshold is a selected fixture score and should agree exactly.

For controller alpha, the source declares no numeric tolerance. Derive a binary64 recurrence-rounding bound rather than inventing a default epsilon:

\[
\operatorname{atol}_t = \gamma_{3t}\left(|\alpha_0| + t|\eta|(|\alpha|+1)\right),
\qquad
\gamma_k = \frac{k\epsilon}{1-k\epsilon},\quad \epsilon=2^{-52}.
\]

Fail on any discrete mismatch regardless of alpha tolerance. A biting witness changes one miss indicator or one score around a rank boundary and requires the first divergence to be identified at the affected step.

## Alternatives rejected

- [`mzaffran/AdaptiveConformalPredictionsTimeSeries`](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries) at commit [`131656fe4c25251bad745f52db3c2d7cb1c24bbb`](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries/commit/131656fe4c25251bad745f52db3c2d7cb1c24bbb): credible and MIT-licensed, but its ACI path is coupled to model refitting and uses a NumPy-default interpolation convention, making a compact, version-stable score-stream trace less suitable.
- MAPIE: maintained and permissively licensed, but its broader time-series interval machinery is not the pinned published ACI score-stream reference required here.

## Residual decision

The reference leaves a policy question: whether burn-in, unclipped raw alpha, quantile-level clipping, and one-sided score-threshold semantics are normative successor behavior or only reference-trace semantics. Resolve that explicitly before decomposing S3-U13.
