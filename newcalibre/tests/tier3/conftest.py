"""Gate tier 3 on promoted capture bytes."""

from __future__ import annotations

from pathlib import Path

import pytest

CAPTURES_ROOT = Path(__file__).parents[3] / "stage3" / "evidence" / "captures"


@pytest.fixture(scope="session")
def promoted_captures_root() -> Path:
    """Expose promoted bytes or visibly skip until U7b lands them."""
    if not CAPTURES_ROOT.is_dir():
        pytest.skip("tier 3 skipped: promoted oracle captures are absent pending U7b")
    return CAPTURES_ROOT
