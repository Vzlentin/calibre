"""Keep uninstantiated tolerance-class-4 contracts visibly pending."""

import pytest

pytestmark = pytest.mark.tier2


@pytest.mark.xfail(
    strict=True,
    reason="Pending U5c: kill/resume needs the time-loop and restart path.",
)
def test_resumed_run_matches_uninterrupted_run() -> None:
    pytest.fail("U5c must replace this placeholder with a biting resume contract.")


@pytest.mark.xfail(
    strict=True,
    reason="Pending U16: distribution invariance needs the dispatch substrate.",
)
def test_distributed_run_matches_sequential_run() -> None:
    pytest.fail("U16 must replace this placeholder with a biting distribution contract.")


@pytest.mark.xfail(
    strict=True,
    reason="Pending U10: state equality needs serializable calibration state.",
)
def test_serialized_state_matches_never_serialized_state() -> None:
    pytest.fail("U10 must replace this placeholder with a biting state round-trip contract.")


@pytest.mark.xfail(
    strict=True,
    reason="Pending U5c: byte identity needs a seed-sensitive test adapter.",
)
def test_same_seed_produces_same_bytes() -> None:
    pytest.fail("U5c must replace this placeholder with a biting seed contract.")
