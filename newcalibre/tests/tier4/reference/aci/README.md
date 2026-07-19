# Pinned ACI reference trace

This directory contains an offline trace minted from the method-defining ACI implementation. The test suite reads only `trace.json`; the reference repository and its dependencies are mint-time inputs, not test dependencies.

## Authority and license

- Repository: [`aangelopoulos/conformal-time-series`](https://github.com/aangelopoulos/conformal-time-series)
- Commit: `b729c3f5ff633bfc43f0f7ca08199b549c2573ac`
- Source: `core/methods.py::aci`, lines 76–111 at that commit
- Git source-blob ID: `c338340aa93d646770142db9ae8cb55b5340a856`
- Source-blob SHA-256: `8aad182d6ec3bc16c4e632012b6f259b8e4a89722efbe4a1df576054278f6fa6`
- License: MIT, repository `LICENSE` blob `3e152a0069505a6bccbb510673c0db17389dafd3`

The disposable mint used CPython 3.12.13 with NumPy 1.26.4. Its complete package set was `numpy==1.26.4`, `packaging==26.2`, `pandas==2.2.3`, `patsy==1.0.2`, `python-dateutil==2.9.0.post0`, `pytz==2026.2`, `scipy==1.17.1`, `six==1.17.0`, `statsmodels==0.14.4`, `tqdm==4.67.1`, and `tzdata==2026.3`.

## Fixture and policy cases

All cases use scores `[1, 4, 2, 8, 3, 9, 0, 7, 8, 6, 10, 2]`, target alpha `0.25`, active-window length `5`, ahead `1`, and NumPy's `method="higher"` on the adaptive branch.

1. `shared-adaptive-eta-0.125` uses burn-in `3` and learning rate `0.125`. Rows 4–11 are shared comparisons. Row 8 is the closed-threshold equality (`score == threshold == 8`).
2. `reference-burn-in-prefix-linear` extends burn-in to `6`. Rows 5–6 enter the reference's prefix-wide, default-linear interpolation branch and are labelled `reference-only-burn-in`; this is not a successor configuration surface.
3. `prefix-count-eta-0.18` uses burn-in `3` and learning rate `0.18`. The trace's prefix-count predicate remains finite at step 6, while the successor's active-window predicate is intentionally unresolvable there. The row is labelled `intentional-prefix-count-departure`.

Every finite float is a canonical binary64 hexadecimal string. Positive infinity is represented only as `{"non_finite":"+infinity"}`. Score windows are half-open. `selected_higher_rank` is zero-based and is `null` when no `higher` selection occurs. `covered` uses the reference's closed threshold; `error` is its exact complement. The `feedback_applied` flag distinguishes a measured burn-in threshold from an adaptive recurrence.

## Re-mint procedure

Use a disposable directory. The only presentation change is replacing the imported `tqdm` callable after module import; the pinned `aci` function body is executed unchanged.

```bash
rm -rf /tmp/aci-reference-mint
mkdir -p /tmp/aci-reference-mint
cd /tmp/aci-reference-mint
git clone https://github.com/aangelopoulos/conformal-time-series.git source
git -C source checkout --detach b729c3f5ff633bfc43f0f7ca08199b549c2573ac
test "$(git -C source rev-parse HEAD:core/methods.py)" = \
  c338340aa93d646770142db9ae8cb55b5340a856
sha256sum source/core/methods.py
```

Save the following as `/tmp/aci-reference-mint/mint_trace.py`:

```python
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parent / "source"
OUTPUT = Path(__file__).parent / "trace.json"
SCORES = np.asarray([1, 4, 2, 8, 3, 9, 0, 7, 8, 6, 10, 2], dtype=np.float64)
TARGET_ALPHA = 0.25
WINDOW_LENGTH = 5
AHEAD = 1

sys.path.insert(0, str(SOURCE))
methods = importlib.import_module("core.methods")
methods.tqdm = lambda values: values


def float_hex(value: float) -> str:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("finite float required")
    return normalized.hex()


def encoded_threshold(value: float) -> str | dict[str, str]:
    if math.isinf(value) and value > 0.0:
        return {"non_finite": "+infinity"}
    return float_hex(value)


def comparison_for(case_id: str, step: int) -> str:
    if case_id == "shared-adaptive-eta-0.125":
        return "reference-initialization" if step <= 3 else "shared"
    if case_id == "reference-burn-in-prefix-linear":
        if step <= 4:
            return "reference-initialization"
        if step <= 6:
            return "reference-only-burn-in"
        return "post-reference-burn-in"
    if step <= 3:
        return "reference-initialization"
    if step <= 5:
        return "shared"
    if step == 6:
        return "intentional-prefix-count-departure"
    return "post-departure-reference"


def mint_case(
    case_id: str,
    classification: str,
    *,
    learning_rate: float,
    burn_in: int,
    expected_divergence: dict[str, object] | None,
) -> dict[str, object]:
    result = methods.aci(
        SCORES.copy(),
        TARGET_ALPHA,
        learning_rate,
        WINDOW_LENGTH,
        burn_in,
        AHEAD,
    )
    thresholds = np.asarray(result["q"], dtype=np.float64)
    alpha_before = np.asarray(result["alpha"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    for step, threshold in enumerate(thresholds):
        t_pred = step - AHEAD + 1
        adaptive = t_pred > burn_in
        if adaptive:
            window_start = max(t_pred - WINDOW_LENGTH, 0)
            branch = (
                "adaptive-unresolvable-prefix-count"
                if math.isinf(float(threshold))
                else "adaptive-higher"
            )
        else:
            window_start = 0
            branch = (
                "burn-in-prefix-linear"
                if t_pred > math.ceil(1.0 / TARGET_ALPHA)
                else "burn-in-insufficient-prefix"
            )
        window_stop = t_pred
        before = float(alpha_before[step])
        covered = bool(float(threshold) >= float(SCORES[step]))
        error = int(not covered)
        after = before
        if adaptive:
            gradient = -TARGET_ALPHA if covered else 1.0 - TARGET_ALPHA
            after = before - learning_rate * gradient
        level = 1.0 - float(np.clip(before, 0.0, 1.0))
        rank: int | None = None
        if branch == "adaptive-higher":
            selected = sorted(float(value) for value in SCORES[window_start:window_stop])
            rank = selected.index(float(threshold))
        rows.append(
            {
                "alpha_after_feedback": float_hex(after),
                "alpha_before": float_hex(before),
                "clipped_quantile_level": float_hex(level),
                "comparison": comparison_for(case_id, step),
                "covered": int(covered),
                "error": error,
                "feedback_applied": adaptive,
                "reference_branch": branch,
                "score_window": {"start": window_start, "stop": window_stop},
                "selected_higher_rank": rank,
                "selected_threshold": encoded_threshold(float(threshold)),
                "source_index": step,
                "t_pred": t_pred,
            }
        )
    return {
        "classification": classification,
        "expected_first_successor_divergence": expected_divergence,
        "id": case_id,
        "inputs": {
            "ahead": AHEAD,
            "burn_in": burn_in,
            "learning_rate": float_hex(learning_rate),
            "quantile_method": "higher",
            "scores": [float_hex(float(value)) for value in SCORES],
            "target_alpha": float_hex(TARGET_ALPHA),
            "window_length": WINDOW_LENGTH,
        },
        "rows": rows,
    }


payload = {
    "cases": [
        mint_case(
            "shared-adaptive-eta-0.125",
            "shared-adaptive",
            learning_rate=0.125,
            burn_in=3,
            expected_divergence=None,
        ),
        mint_case(
            "reference-burn-in-prefix-linear",
            "trace-only-reference-burn-in",
            learning_rate=0.125,
            burn_in=6,
            expected_divergence={
                "quantity": "burn-in-branch",
                "reference": "burn-in-prefix-linear",
                "step": 5,
                "successor": "adaptive-higher",
            },
        ),
        mint_case(
            "prefix-count-eta-0.18",
            "deliberate-prefix-count-departure",
            learning_rate=0.18,
            burn_in=3,
            expected_divergence={
                "quantity": "branch-predicate",
                "reference": "adaptive-higher",
                "step": 6,
                "successor": "adaptive-unresolvable-active-window",
            },
        ),
    ]
}
payload_bytes = json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
document = {
    "payload": payload,
    "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    "schema": "aci-reference-trace",
    "schema_version": 1,
}
OUTPUT.write_text(
    json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(document["payload_sha256"])
```

Execute it in the fully pinned disposable environment and compare the result:

```bash
cd /tmp/aci-reference-mint
uv run --isolated --python 3.12.13 \
  --with numpy==1.26.4 --with packaging==26.2 --with pandas==2.2.3 \
  --with patsy==1.0.2 --with python-dateutil==2.9.0.post0 --with pytz==2026.2 \
  --with scipy==1.17.1 --with six==1.17.0 --with statsmodels==0.14.4 \
  --with tqdm==4.67.1 --with tzdata==2026.3 mint_trace.py
# Printed payload digest:
# 86c7da88a4b21ae867d6f658e47b02544300e536e6589b2816513a44576032ac
cmp trace.json "$REPO/newcalibre/tests/tier4/reference/aci/trace.json"
```

## Canonical payload digest and byte check

The payload digest is SHA-256 over UTF-8 JSON with sorted keys, no insignificant whitespace, no NaN, and separators `(',', ':')`. The document itself uses sorted keys, two-space indentation, and one trailing newline. From `newcalibre/`, run this twice; both generated files must match each other and the committed bytes:

```bash
for run in 1 2; do
  uv run --locked python - "$run" <<'PY' > "/tmp/aci-trace-canonical-$run.json"
import hashlib
import json
import sys
from pathlib import Path

path = Path("tests/tier4/reference/aci/trace.json")
value = json.loads(path.read_bytes(), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
payload = json.dumps(
    value["payload"],
    allow_nan=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
).encode()
assert hashlib.sha256(payload).hexdigest() == value["payload_sha256"]
canonical = (
    json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
).encode()
assert canonical == path.read_bytes()
sys.stdout.buffer.write(canonical)
PY
done
cmp /tmp/aci-trace-canonical-1.json /tmp/aci-trace-canonical-2.json
cmp /tmp/aci-trace-canonical-1.json tests/tier4/reference/aci/trace.json
sha256sum /tmp/aci-trace-canonical-{1,2}.json tests/tier4/reference/aci/trace.json
```
