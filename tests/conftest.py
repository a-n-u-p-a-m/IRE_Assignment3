"""Same path fix as the root conftest, for `pytest` run from inside tests/."""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
