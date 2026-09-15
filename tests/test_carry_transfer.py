import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CarryTransferTests(unittest.TestCase):
    def test_original_k16_intermediates_and_rejected_boundaries(self):
        with tempfile.TemporaryDirectory(prefix="qrt-carry-transfer-") as directory:
            executable = Path(directory) / "carry-transfer"
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                 "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                 "-fno-sanitize-recover=all", "-I",
                 str(ROOT / "native/providers/moe_accumulator"),
                 str(ROOT / "tests/native/carry_transfer_host_selftest.cpp"),
                 "-o", str(executable)], check=True, timeout=30,
            )
            result = subprocess.run(
                [str(executable)], check=True, capture_output=True, text=True, timeout=30,
            )
        reports = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(reports), 4)
        self.assertEqual({r["batch_groups"] for r in reports[:3]}, {2, 4, 8})
        for report in reports[:3]:
            self.assertEqual(report["blocks"], 32768)
            self.assertEqual(report["canonical_groups"], 32768 * report["batch_groups"])
            self.assertEqual(report["raw_mismatches"], 0)
            self.assertGreater(report["certified_transfer"], 1000)
            self.assertGreaterEqual(report["final_normalize_transfer"], report["certified_transfer"])
        self.assertEqual(reports[-1], {
            "kind": "carry_transfer_boundaries", "rejected": 36, "outputs_unchanged": True,
        })
        print(result.stdout.strip(), flush=True)
