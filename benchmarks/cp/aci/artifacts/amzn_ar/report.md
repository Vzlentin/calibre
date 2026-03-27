# One-Step ACI Parity Report

## Run Metadata

- Reference repo commit: `b729c3f5ff633bfc43f0f7ca08199b549c2573ac`
- Dataset: `AMZN`
- Forecast model artifact: `ar`
- Score contract: `signed-residual`
- Burn-in: `100`
- Nominal alpha: `0.1`
- Tail alpha used for each signed-residual side: `0.05`
- Window length: `10000000`

## Contract Translation

- Warm start: match the reference warm-up exactly until `t > T_burnin`, then start the local controller with the full pre-adaptive score history.
- Conformity scores: split the reference `signed-residual` score into lower and upper scalar tails and run one local controller per tail.
- Finite-sample quantile rule: run the local controller with `quantile_rule='higher'` to mirror the reference repo's `method='higher'` selection.
- Alpha bounds: run the local controller with `alpha_bounds=None` so alpha can evolve without clipping, matching the reference update path.
- Quantile representation: combine the two local tail thresholds into the same asymmetric interval form as the reference repo.
- Alpha update timing: both controllers update after observing the current score.

## Verdicts

### lr = 0.1

- Verdict: `exact`
- Diagnosis: No divergence detected.
- First lower-tail `q` difference: `None`
- First upper-tail `q` difference: `None`
- First miss-sequence difference: `None`
- Coverage delta on evaluation region: `0.0`
- Average width delta on evaluation region: `n/a`

### lr = 0.05

- Verdict: `exact`
- Diagnosis: No divergence detected.
- First lower-tail `q` difference: `None`
- First upper-tail `q` difference: `None`
- First miss-sequence difference: `None`
- Coverage delta on evaluation region: `0.0`
- Average width delta on evaluation region: `n/a`

### lr = 0.01

- Verdict: `exact`
- Diagnosis: No divergence detected.
- First lower-tail `q` difference: `None`
- First upper-tail `q` difference: `None`
- First miss-sequence difference: `None`
- Coverage delta on evaluation region: `0.0`
- Average width delta on evaluation region: `n/a`

### lr = 0.005

- Verdict: `exact`
- Diagnosis: No divergence detected.
- First lower-tail `q` difference: `None`
- First upper-tail `q` difference: `None`
- First miss-sequence difference: `None`
- Coverage delta on evaluation region: `0.0`
- Average width delta on evaluation region: `n/a`

### lr = 0.0001

- Verdict: `exact`
- Diagnosis: No divergence detected.
- First lower-tail `q` difference: `None`
- First upper-tail `q` difference: `None`
- First miss-sequence difference: `None`
- Coverage delta on evaluation region: `0.0`
- Average width delta on evaluation region: `0.0`

