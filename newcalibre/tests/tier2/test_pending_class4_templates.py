"""Tier-2 self-consistency templates: visibly pending until their owning units land.

Gate A instantiates only the meaningful class-4 legs (same seed == same
bytes with a seed-sensitive adapter; resumed == uninterrupted), which need
the engine loop and the seeded test adapter from U5. They are recorded here
as *skips with named owners* rather than green-washed passes; later class-4
legs (serialized calibration state U10, distributed == sequential U16,
two-driver equivalence U14) stay pending even at the gate.
"""

import pytest

pytestmark = pytest.mark.tier2


@pytest.mark.skip(reason="pending U5c: same-seed byte identity needs the seeded test adapter")
def test_same_seed_same_bytes():
    """Non-vacuous only once a seed-sensitive adapter exists (seasonal-naive cannot bite)."""


@pytest.mark.skip(reason="pending U5c: kill/resume needs the time-loop driver and resume path")
def test_resumed_equals_uninterrupted():
    """Resume template lands with the U5c time-loop composition."""
