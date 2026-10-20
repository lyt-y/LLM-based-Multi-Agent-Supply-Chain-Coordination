"""Locate the project source tree for direct and module script execution."""

from pathlib import Path
import sys


def configure_paths():
    source = str(Path(__file__).resolve().parents[1] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
