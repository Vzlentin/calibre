"""Make the throwaway ``gate_b_proto`` package importable for the test suite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
