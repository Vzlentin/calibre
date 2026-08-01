"""Keep absent promoted captures visible instead of green-washing tier 3."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.tier3


def test_promoted_capture_precondition(promoted_captures_root: Path) -> None:
    assert promoted_captures_root.is_dir()
