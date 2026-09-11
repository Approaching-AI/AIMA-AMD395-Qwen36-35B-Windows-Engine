"""Keep expanded reference captures anchored to the immutable token fixtures."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from capture_gb10_token_matrix import fingerprints, fixtures, sampled_prefill_boundary  # noqa: E402


ORACLES = {
    7169: ROOT / "contracts/arbitrary_q7169_gb10_oracle.json",
    8192: ROOT / "contracts/hprefill_q8192_gb10_oracle.json",
}


class Gb10TokenMatrixTests(unittest.TestCase):
    def test_frozen_matrix_binds_complete_outputs_and_original_controls(self):
        path = ROOT / "contracts/gb10_cold_token_matrix_20260911_oracle.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "7a6feb488f4dd136e49c4d7227c1fc325e3216af3dae3e9b60aa0b3025f3d1b1")
        matrix = json.loads(path.read_text())
        materialized, oracles = fixtures(ORACLES)
        cases = {case["name"]: case for case in matrix["cases"]}
        self.assertEqual(set(cases), {case["name"] for case in materialized})
        for fixture in materialized:
            case = cases[fixture["name"]]
            expected = case["expected"]
            for key, value in fixture["prompt"].items():
                self.assertEqual(case["prompt"][key], value)
            self.assertEqual(len(expected["output_token_ids"]), fixture["output_count"])
            self.assertEqual(expected["first_token_id"], expected["output_token_ids"][0])
            self.assertEqual(expected["first_token_raw_logit_tolerance"], 0.125)
            digest = fingerprints(expected["output_token_ids"])
            self.assertEqual(digest["u32le_sha256"], expected["output_token_ids_u32le_sha256"])
            self.assertEqual(digest["u32le_fnv1a64"], expected["output_token_ids_u32le_fnv1a64"])
            worker = case["raw_logits"]
            self.assertFalse(worker["observer_modifies_output"])
            self.assertFalse(worker["sampling_boundary"]["discarded"])
            self.assertEqual(worker["sampling_boundary"]["processed_tokens"],
                             fixture["prompt"]["token_count"])
            self.assertEqual(worker["raw_argmax_token"], expected["first_token_id"])
            self.assertEqual(worker["raw_logit"], expected["first_token_raw_logit"])
            if fixture["control"]:
                frozen = oracles[fixture["family"]]["expected"]
                self.assertEqual(expected["output_token_ids"][:32], frozen["output_token_ids"])
                self.assertEqual(expected["first_token_raw_logit"], frozen["first_token_raw_logit"])

    def test_partial_prefill_logits_follow_the_sampler_discard_boundary(self):
        self.assertFalse(sampled_prefill_boundary(1, 8193, 8192, True, 8193))
        self.assertTrue(sampled_prefill_boundary(1, 8193, 8193, False, 8193))
        self.assertTrue(sampled_prefill_boundary(1, 7169, 7169, False, 7169))
        for arguments in ((1, 8193, 8192, False, 8193), (1, 8193, 8193, True, 8193),
                          (2, 7169, 7169, False, 7169), (1, 8193, 8193, False, 8192)):
            with self.assertRaises(ValueError):
                sampled_prefill_boundary(*arguments)

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
