"""Pytest conftest to ensure the project's `src/` directory is importable during tests."""
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    # prepend so local package is preferred
    sys.path.insert(0, str(SRC))
