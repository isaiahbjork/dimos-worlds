"""Opt-in end-to-end run of go2-lot-night (DimOS workers + real-time MuJoCo; takes minutes).

    DIMOS_WORLDS_E2E=1 pytest tests/test_lot_e2e.py -s
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("DIMOS_WORLDS_E2E") != "1", reason="set DIMOS_WORLDS_E2E=1 to run")
def test_go2_walks_to_named_places():
    script = Path(__file__).with_name("e2e_go2_lot_night.py")
    proc = subprocess.run(
        [sys.executable, str(script), "row-d", "south-fence", "row-b"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    results = [line for line in proc.stdout.splitlines() if line.startswith("RESULT")]
    print("\n".join(results))
    assert len(results) == 3, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert proc.returncode == 0, "\n".join(results)
