"""Replay the pinned ACI reference trace through the successor runtime."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from newcalibre.conformal import (
    CalibrationResult,
    CalibrationSeedBatch,
    DeliveryBatch,
    ForecastKey,
    ResolvedObservation,
    derive_partition_label,
    resolve_method,
)
from newcalibre.conformal.state import JsonStateCodec
from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    CensoringAssertion,
    EmissionScope,
)
from newcalibre.domain._canonical_json import canonical_json_bytes

pytestmark = pytest.mark.tier4

_TRACE_PATH = Path(__file__).with_name("trace.json")
_METHOD = "sequential-adaptive-per-step"
_MODEL = "aci-reference"
_SERIES = "reference-series"
_ORIGIN = pd.Timestamp("2026-01-01")
_ROOT_FIELDS = {"payload", "payload_sha256", "schema", "schema_version"}
_CASE_FIELDS = {
    "classification",
    "expected_first_successor_divergence",
    "id",
    "inputs",
    "rows",
}
_INPUT_FIELDS = {
    "ahead",
    "burn_in",
    "learning_rate",
    "quantile_method",
    "scores",
    "target_alpha",
    "window_length",
}
_ROW_FIELDS = {
    "alpha_after_feedback",
    "alpha_before",
    "clipped_quantile_level",
    "comparison",
    "covered",
    "error",
    "feedback_applied",
    "reference_branch",
    "score_window",
    "selected_higher_rank",
    "selected_threshold",
    "source_index",
    "t_pred",
}
_EXPECTED_CASES = {
    "shared-adaptive-eta-0.125": "shared-adaptive",
    "reference-burn-in-prefix-linear": "trace-only-reference-burn-in",
    "prefix-count-eta-0.18": "deliberate-prefix-count-departure",
}
_COMPARISONS = {
    "intentional-prefix-count-departure",
    "post-departure-reference",
    "post-reference-burn-in",
    "reference-initialization",
    "reference-only-burn-in",
    "shared",
}
_BRANCHES = {
    "adaptive-higher",
    "adaptive-unresolvable-prefix-count",
    "burn-in-insufficient-prefix",
    "burn-in-prefix-linear",
}
_INFINITY = {"non_finite": "+infinity"}


class _DuplicateJsonField(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonField(f"duplicate field {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _canonical_document(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _load_reference_trace(path: Path = _TRACE_PATH) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise AssertionError(f"ACI reference trace is not strict JSON: {error}") from error
    _validate_trace(value)
    assert raw == _canonical_document(value), "ACI reference trace bytes are not canonical"
    return cast(dict[str, object], value)


def _validate_trace(value: object) -> None:
    assert isinstance(value, dict) and set(value) == _ROOT_FIELDS, (
        "ACI reference trace has an invalid root schema"
    )
    assert value["schema"] == "aci-reference-trace" and value["schema_version"] == 1, (
        "ACI reference trace has an invalid identity"
    )
    payload = value["payload"]
    assert isinstance(payload, dict) and set(payload) == {"cases"}, (
        "ACI reference trace payload must contain only cases"
    )
    cases = payload["cases"]
    assert isinstance(cases, list), "ACI reference trace cases must be a list"
    identities: dict[str, str] = {}
    for case in cases:
        _validate_case(case)
        case_value = cast(dict[str, object], case)
        case_id = cast(str, case_value["id"])
        assert case_id not in identities, f"ACI reference trace has duplicate case ID {case_id!r}"
        identities[case_id] = cast(str, case_value["classification"])
    assert identities == _EXPECTED_CASES, "ACI reference trace case inventory is not exact"
    digest = value["payload_sha256"]
    assert isinstance(digest, str) and len(digest) == 64, (
        "ACI reference trace payload digest is malformed"
    )
    actual = hashlib.sha256(
        canonical_json_bytes(payload, path="ACI reference trace payload")
    ).hexdigest()
    assert digest == actual, "ACI reference trace payload digest mismatch"


def _validate_case(value: object) -> None:
    assert isinstance(value, dict) and set(value) == _CASE_FIELDS, (
        "ACI reference trace case has an invalid schema"
    )
    assert isinstance(value["id"], str) and value["id"], "ACI reference case ID is invalid"
    assert isinstance(value["classification"], str) and value["classification"], (
        "ACI reference case classification is invalid"
    )
    inputs = value["inputs"]
    assert isinstance(inputs, dict) and set(inputs) == _INPUT_FIELDS, (
        "ACI reference case inputs have an invalid schema"
    )
    for name in ("ahead", "burn_in", "window_length"):
        assert type(inputs[name]) is int and inputs[name] >= 0, (
            f"ACI reference input {name} is invalid"
        )
    assert inputs["ahead"] == 1 and inputs["window_length"] == 5, (
        "ACI reference fixture identity is invalid"
    )
    assert inputs["quantile_method"] == "higher", "ACI reference quantile method is invalid"
    learning_rate = _finite_hex(inputs["learning_rate"], name="learning rate")
    target_alpha = _finite_hex(inputs["target_alpha"], name="target alpha")
    scores_raw = inputs["scores"]
    assert isinstance(scores_raw, list) and scores_raw, "ACI reference scores are invalid"
    scores = tuple(_finite_hex(item, name="score") for item in scores_raw)
    rows = value["rows"]
    assert isinstance(rows, list) and len(rows) == len(scores), (
        "ACI reference rows must align exactly with scores"
    )
    prior_after = target_alpha
    for step, row in enumerate(rows):
        _validate_row(
            row,
            step=step,
            scores=scores,
            target_alpha=target_alpha,
            learning_rate=learning_rate,
            window_length=cast(int, inputs["window_length"]),
            ahead=cast(int, inputs["ahead"]),
            prior_after=prior_after,
        )
        prior_after = _finite_hex(
            cast(dict[str, object], row)["alpha_after_feedback"],
            name="alpha after feedback",
        )
    divergence = value["expected_first_successor_divergence"]
    assert divergence is None or (
        isinstance(divergence, dict)
        and set(divergence) == {"quantity", "reference", "step", "successor"}
        and type(divergence["step"]) is int
        and all(
            isinstance(divergence[name], str) for name in ("quantity", "reference", "successor")
        )
    ), "ACI reference expected-divergence declaration is invalid"


def _validate_row(
    value: object,
    *,
    step: int,
    scores: tuple[float, ...],
    target_alpha: float,
    learning_rate: float,
    window_length: int,
    ahead: int,
    prior_after: float,
) -> None:
    assert isinstance(value, dict) and set(value) == _ROW_FIELDS, (
        "ACI reference row has an invalid schema"
    )
    assert value["source_index"] == step, "ACI reference row order is invalid"
    t_pred = step - ahead + 1
    assert value["t_pred"] == t_pred, "ACI reference prediction index is invalid"
    assert value["comparison"] in _COMPARISONS, "ACI reference comparison label is invalid"
    branch = value["reference_branch"]
    assert branch in _BRANCHES, "ACI reference branch label is invalid"
    window = value["score_window"]
    assert isinstance(window, dict) and set(window) == {"start", "stop"}, (
        "ACI reference score window has an invalid schema"
    )
    expected_start = max(t_pred - window_length, 0) if str(branch).startswith("adaptive") else 0
    assert window == {"start": expected_start, "stop": t_pred}, (
        "ACI reference score window bounds are invalid"
    )
    alpha_before = _finite_hex(value["alpha_before"], name="alpha before feedback")
    alpha_after = _finite_hex(value["alpha_after_feedback"], name="alpha after feedback")
    level = _finite_hex(value["clipped_quantile_level"], name="quantile level")
    assert alpha_before == prior_after, "ACI reference alpha recurrence is discontinuous"
    assert level == 1.0 - min(1.0, max(0.0, alpha_before)), (
        "ACI reference clipped quantile level is invalid"
    )
    assert type(value["covered"]) is int and value["covered"] in (0, 1), (
        "ACI reference covered indicator is invalid"
    )
    assert type(value["error"]) is int and value["error"] in (0, 1), (
        "ACI reference error indicator is invalid"
    )
    assert value["covered"] + value["error"] == 1, (
        "ACI reference covered and error indicators disagree"
    )
    feedback = value["feedback_applied"]
    assert type(feedback) is bool and feedback == str(branch).startswith("adaptive"), (
        "ACI reference feedback flag is invalid"
    )
    threshold = _threshold(value["selected_threshold"])
    assert value["covered"] == int(threshold >= scores[step]), (
        "ACI reference covered indicator does not match its closed threshold"
    )
    rank = value["selected_higher_rank"]
    if branch == "adaptive-higher":
        assert type(rank) is int and rank >= 0 and math.isfinite(threshold), (
            "ACI reference higher selection is invalid"
        )
        ordered = sorted(scores[cast(int, window["start"]) : cast(int, window["stop"])])
        assert rank < len(ordered) and ordered[rank] == threshold, (
            "ACI reference higher rank does not select its threshold"
        )
    elif branch == "burn-in-prefix-linear":
        assert rank is None and math.isfinite(threshold), (
            "ACI reference linear burn-in selection is invalid"
        )
    else:
        assert rank is None and math.isinf(threshold), (
            "ACI reference unresolved selection must use explicit infinity"
        )
    expected_after = alpha_before
    if feedback:
        gradient = -target_alpha if value["covered"] else 1.0 - target_alpha
        expected_after = alpha_before - learning_rate * gradient
    assert alpha_after == expected_after, "ACI reference alpha update is invalid"


def _finite_hex(value: object, *, name: str) -> float:
    assert isinstance(value, str), f"ACI reference {name} must be a binary64 hexadecimal string"
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise AssertionError(f"ACI reference {name} is not hexadecimal") from error
    assert math.isfinite(parsed) and parsed.hex() == value, (
        f"ACI reference {name} is not canonical binary64"
    )
    return parsed


def _threshold(value: object) -> float:
    if isinstance(value, dict):
        assert value == _INFINITY, "ACI reference non-finite token is invalid"
        return math.inf
    return _finite_hex(value, name="selected threshold")


def _reference_case(document: Mapping[str, object], case_id: str) -> dict[str, object]:
    payload = cast(dict[str, object], document["payload"])
    cases = cast(list[dict[str, object]], payload["cases"])
    return next(case for case in cases if case["id"] == case_id)


def _case_scores(case: Mapping[str, object]) -> tuple[float, ...]:
    inputs = cast(dict[str, object], case["inputs"])
    return tuple(_finite_hex(value, name="score") for value in cast(list[object], inputs["scores"]))


def _frame(step: int) -> pd.DataFrame:
    origin = _ORIGIN + pd.Timedelta(days=step)
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series([_SERIES], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime([origin]),
            ACTUAL_VALUE: pd.Series([math.nan], dtype="float64"),
            POINT_FORECAST: pd.Series([0.0], dtype="float64"),
            HORIZON_STEP: pd.Series([1], dtype="int64"),
            ORIGIN: pd.to_datetime([origin]),
            MODEL_NAME: pd.Series([_MODEL], dtype="string"),
        }
    )


def _observation(result: CalibrationResult, *, score: float) -> ResolvedObservation:
    row = result.forecasts.iloc[0]
    key = ForecastKey(
        series_key=cast(str, row[SERIES_KEY]),
        origin=pd.Timestamp(row[ORIGIN]),
        horizon_step=cast(int, row[HORIZON_STEP]),
        model_name=cast(str, row[MODEL_NAME]),
    )
    return ResolvedObservation(
        forecast_key=key,
        target_timestamp=pd.Timestamp(row[TARGET_TIMESTAMP]),
        actual=score,
        point_forecast=cast(float, row[POINT_FORECAST]),
        censoring_assertion=CensoringAssertion.UNCENSORED,
        availability_bound=None,
        issued=result.issuances[key],
    )


def _replay_successor_branches_through(
    case: Mapping[str, object],
    *,
    stop: int,
) -> tuple[str, ...]:
    inputs = cast(dict[str, object], case["inputs"])
    scores = _case_scores(case)
    runtime = resolve_method(
        {
            "method": _METHOD,
            "coverage": 1.0 - _finite_hex(inputs["target_alpha"], name="target alpha"),
            "calibration_window": inputs["window_length"],
            "learning_rate": _finite_hex(inputs["learning_rate"], name="learning rate"),
        }
    )
    label = derive_partition_label(_MODEL, "global", EmissionScope.PER_STEP)
    states = runtime.calibrate(CalibrationSeedBatch({label: ()}))
    branches: list[str] = []

    for step in range(stop + 1):
        result = runtime.apply(_frame(step), states)
        states = result.state
        facts = next(iter(result.issuances.values()))
        if facts.bounds_null_reason is None:
            branches.append("adaptive-higher")
        elif facts.bounds_null_reason == "unresolvable-working-level":
            branches.append("adaptive-unresolvable-active-window")
        else:
            branches.append("warm-up")
        if step < stop:
            effect = runtime.observe(
                DeliveryBatch({label: (_observation(result, score=scores[step]),)}),
                states,
            )
            states = effect.state

    return tuple(branches)


def _assert_reference_value(
    *,
    step: int,
    quantity: str,
    expected: object,
    actual: object,
    atol: float | None = None,
) -> None:
    matches = (
        expected == actual
        if atol is None
        else abs(cast(float, expected) - cast(float, actual)) <= atol
    )
    if not matches:
        raise AssertionError(
            f"ACI reference mismatch: step={step} quantity={quantity} "
            f"expected={expected!r} actual={actual!r}"
        )


def _alpha_atol(
    *,
    t: int,
    alpha_0: float,
    eta: float,
    target_alpha: float,
) -> float:
    # Each feedback recurrence has at most three rounded binary64 operations. Standard
    # gamma accumulation therefore bounds those 3t operations without a fitted tolerance.
    epsilon = 2**-52
    k = 3 * t
    gamma_k = k * epsilon / (1.0 - k * epsilon)
    atol_t = gamma_k * (abs(alpha_0) + t * abs(eta) * (abs(target_alpha) + 1.0))
    return atol_t


def _assert_case_matches(
    case: Mapping[str, object],
    *,
    scores_override: tuple[float, ...] | None = None,
) -> None:
    inputs = cast(dict[str, object], case["inputs"])
    reference_scores = _case_scores(case)
    scores = reference_scores if scores_override is None else scores_override
    assert len(scores) == len(reference_scores), "ACI replay score override has the wrong length"
    target_alpha = _finite_hex(inputs["target_alpha"], name="target alpha")
    learning_rate = _finite_hex(inputs["learning_rate"], name="learning rate")
    window_length = cast(int, inputs["window_length"])
    rows = cast(list[dict[str, object]], case["rows"])
    adaptive_rows = tuple(
        row for row in rows if cast(int, row["source_index"]) > cast(int, inputs["burn_in"])
    )
    first_step = cast(int, adaptive_rows[0]["source_index"])
    _assert_reference_value(
        step=first_step,
        quantity="row-order",
        expected=tuple(range(first_step, len(rows))),
        actual=tuple(cast(int, row["source_index"]) for row in adaptive_rows),
    )

    runtime = resolve_method(
        {
            "method": _METHOD,
            "coverage": 1.0 - target_alpha,
            "calibration_window": window_length,
            "learning_rate": learning_rate,
        }
    )
    label = derive_partition_label(_MODEL, "global", EmissionScope.PER_STEP)
    states = runtime.calibrate(CalibrationSeedBatch({label: scores[:first_step]}))
    codec = JsonStateCodec(_METHOD, 1)

    for row in adaptive_rows:
        step = cast(int, row["source_index"])
        before = cast(dict[str, object], codec.decode(states[label], expected_label=label))
        retained = tuple(cast(float, value) for value in cast(list[object], before["scores"]))
        actual_window = {"start": step - len(retained), "stop": step}
        _assert_reference_value(
            step=step,
            quantity="t-pred",
            expected=row["t_pred"],
            actual=step,
        )
        _assert_reference_value(
            step=step,
            quantity="score-window-bounds",
            expected=row["score_window"],
            actual=actual_window,
        )
        expected_window = reference_scores[actual_window["start"] : actual_window["stop"]]
        _assert_reference_value(
            step=step,
            quantity="score-window-values",
            expected=expected_window,
            actual=retained,
        )
        alpha_before = cast(float, before["raw_alpha"])
        feedback_before = cast(int, before["feedback_count"])
        expected_alpha_before = _finite_hex(row["alpha_before"], name="alpha before feedback")
        _assert_reference_value(
            step=step,
            quantity="alpha-before-feedback",
            expected=expected_alpha_before,
            actual=alpha_before,
            atol=_alpha_atol(
                t=feedback_before,
                alpha_0=target_alpha,
                eta=learning_rate,
                target_alpha=target_alpha,
            ),
        )
        level = 1.0 - min(1.0, max(0.0, alpha_before))
        _assert_reference_value(
            step=step,
            quantity="clipped-quantile-level",
            expected=_finite_hex(row["clipped_quantile_level"], name="quantile level"),
            actual=level,
        )

        result = runtime.apply(_frame(step), states)
        states = result.state
        facts = next(iter(result.issuances.values()))
        if facts.bounds_null_reason is None:
            branch = "adaptive-higher"
            threshold = facts.upper_bound - cast(float, result.forecasts.iloc[0][POINT_FORECAST])
            ordered = sorted(retained)
            rank = ordered.index(threshold) if threshold in ordered else None
        else:
            branch = "adaptive-unresolvable-active-window"
            threshold = math.inf
            rank = None
        observation = _observation(result, score=scores[step])
        effect = runtime.observe(DeliveryBatch({label: (observation,)}), states)
        states = effect.state
        after = cast(dict[str, object], codec.decode(states[label], expected_label=label))
        covered = int(threshold >= scores[step])
        error = 1 - covered

        _assert_reference_value(
            step=step,
            quantity="branch-predicate",
            expected=row["reference_branch"],
            actual=branch,
        )
        _assert_reference_value(
            step=step,
            quantity="selected-higher-rank",
            expected=row["selected_higher_rank"],
            actual=rank,
        )
        _assert_reference_value(
            step=step,
            quantity="selected-threshold",
            expected=_threshold(row["selected_threshold"]),
            actual=threshold,
        )
        _assert_reference_value(
            step=step,
            quantity="covered",
            expected=row["covered"],
            actual=covered,
        )
        _assert_reference_value(
            step=step,
            quantity="error",
            expected=row["error"],
            actual=error,
        )
        feedback_after = cast(int, after["feedback_count"])
        _assert_reference_value(
            step=step,
            quantity="alpha-after-feedback",
            expected=_finite_hex(row["alpha_after_feedback"], name="alpha after feedback"),
            actual=cast(float, after["raw_alpha"]),
            atol=_alpha_atol(
                t=feedback_after,
                alpha_0=target_alpha,
                eta=learning_rate,
                target_alpha=target_alpha,
            ),
        )


@pytest.mark.oracle_gate("aci-reference-parity")
def test_shared_adaptive_rows_match_the_pinned_reference() -> None:
    document = _load_reference_trace()
    _assert_case_matches(_reference_case(document, "shared-adaptive-eta-0.125"))


def test_trace_schema_canonical_bytes_and_payload_digest_are_strict() -> None:
    document = _load_reference_trace()
    extra = copy.deepcopy(document)
    extra["unexpected"] = True
    with pytest.raises(AssertionError, match="invalid root schema"):
        _validate_trace(extra)

    duplicate = copy.deepcopy(document)
    duplicate_cases = cast(
        list[dict[str, object]],
        cast(dict[str, object], duplicate["payload"])["cases"],
    )
    duplicate_cases.append(copy.deepcopy(duplicate_cases[0]))
    with pytest.raises(AssertionError, match="duplicate case ID"):
        _validate_trace(duplicate)

    bad_digest = copy.deepcopy(document)
    bad_digest["payload_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="payload digest mismatch"):
        _validate_trace(bad_digest)


def test_reference_burn_in_is_trace_only_and_enters_prefix_interpolation() -> None:
    document = _load_reference_trace()
    case = _reference_case(document, "reference-burn-in-prefix-linear")
    rows = cast(list[dict[str, object]], case["rows"])
    trace_only = [row for row in rows if row["comparison"] == "reference-only-burn-in"]
    divergence = cast(dict[str, object], case["expected_first_successor_divergence"])

    assert case["classification"] == "trace-only-reference-burn-in"
    assert [row["source_index"] for row in trace_only] == [5, 6]
    assert {row["reference_branch"] for row in trace_only} == {"burn-in-prefix-linear"}
    assert all(row["selected_higher_rank"] is None for row in trace_only)
    assert divergence == {
        "quantity": "burn-in-branch",
        "reference": "burn-in-prefix-linear",
        "step": 5,
        "successor": "adaptive-higher",
    }
    step = cast(int, divergence["step"])
    successor_branches = _replay_successor_branches_through(case, stop=step)
    assert rows[step]["reference_branch"] == divergence["reference"]
    assert successor_branches[step] == divergence["successor"]


def test_prefix_count_departure_first_occurs_at_declared_step_six_predicate() -> None:
    document = _load_reference_trace()
    case = _reference_case(document, "prefix-count-eta-0.18")
    divergence = cast(dict[str, object], case["expected_first_successor_divergence"])

    with pytest.raises(AssertionError) as raised:
        _assert_case_matches(case)

    assert case["classification"] == "deliberate-prefix-count-departure"
    assert divergence == {
        "quantity": "branch-predicate",
        "reference": "adaptive-higher",
        "step": 6,
        "successor": "adaptive-unresolvable-active-window",
    }
    assert str(raised.value) == (
        "ACI reference mismatch: step=6 quantity=branch-predicate "
        "expected='adaptive-higher' actual='adaptive-unresolvable-active-window'"
    )
