"""Make `pytest` work from a bare checkout.

The test modules import `annbench`, which lives under `src/`. `run_all.sh`
exports PYTHONPATH=src before running anything, but someone running `pytest`
directly -- a marker checking the test suite, most likely -- has no reason to
know that. pytest imports the rootdir conftest before collecting, so putting
`src` on the path here makes `python3 -m pytest tests` work from a fresh clone
with no environment setup.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
