"""Keep expanded reference captures anchored to the immutable token fixtures."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from capture_gb10_token_matrix import fingerprints, fixtures  # noqa: E402


ORACLES = {
    7169: ROOT / "contracts/arbitrary_q7169_gb10_oracle.json",
    8192: ROOT / "contracts/hprefill_q8192_gb10_oracle.json",
}


class Gb10TokenMatrixTests(unittest.TestCase):
    def test_controls_and_neighbors_preserve_real_token_prefixes(self):
        cases, oracles = fixtures(ORACLES)
        by_name = {case["name"]: case for case in cases}
        for family in (7169, 8192):
            original = by_name[f"q{family}-out32"]
            extended = by_name[f"q{family}-out512"]
            self.assertEqual(original["prompt_token_ids"], extended["prompt_token_ids"])
            self.assertTrue(original["control"] and extended["control"])
            self.assertEqual(original["prompt"]["u32le_sha256"], oracles[family]["prompt"]["u32le_sha256"])
            self.assertEqual(by_name[f"q{family - 1}-out32"]["prompt_token_ids"],
                             original["prompt_token_ids"][:-1])
            self.assertEqual(by_name[f"q{family + 1}-out32"]["prompt_token_ids"][:-1],
                             original["prompt_token_ids"])
            for length in (family - 1, family + 1):
                self.assertFalse(by_name[f"q{length}-out32"]["control"])
            expected = oracles[family]["expected"]
            digest = fingerprints(expected["output_token_ids"])
            self.assertEqual(digest["u32le_sha256"], expected["output_token_ids_u32le_sha256"])
            self.assertEqual(digest["u32le_fnv1a64"], expected["output_token_ids_u32le_fnv1a64"])

    def test_modified_oracle_is_rejected_before_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "oracle.json"
            changed.write_bytes(ORACLES[7169].read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "immutable control"):
                fixtures({7169: changed, 8192: ORACLES[8192]})

    def test_dry_run_has_no_numerical_acceptance_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "capture"
            command = [sys.executable, str(ROOT / "scripts/capture_gb10_token_matrix.py"),
                       "--oracle-q7169", str(ORACLES[7169]),
                       "--oracle-q8192", str(ORACLES[8192]),
                       "--source-commit", "0" * 40, "--output-dir", str(output),
                       "--timeout-seconds", "590"]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
            record = json.loads((output / "capture.json").read_text())
            self.assertFalse(record["completed"])
            self.assertFalse(record["controls_qualified"])
            self.assertFalse(record["windows_acceptance"])
            self.assertFalse(record["prefix_caching"])
            self.assertNotIn("cases", record)
            before = (output / "capture.json").read_bytes()
            repeated = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual((output / "capture.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
