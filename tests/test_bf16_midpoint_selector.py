import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("c++"), "requires a C++ compiler")
class Bf16MidpointSelectorTests(unittest.TestCase):
    def test_all_finite_cell_boundaries_and_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "probe"
            subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off",
                            str(ROOT / "tests/native/bf16_midpoint_selector_probe.cpp"),
                            "-o", str(binary)], check=True, capture_output=True, timeout=30)
            result = subprocess.run([str(binary)], check=True, text=True,
                                    capture_output=True, timeout=10)
            self.assertEqual(json.loads(result.stdout), {"checked": 130552, "mismatches": 0})
