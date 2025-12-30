"""
Test configuration to ensure the package is importable without an installed distribution.

By prepending the local `src` directory to sys.path, `python -m pytest` works
in a fresh clone without `pip install -e .`.
"""

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
