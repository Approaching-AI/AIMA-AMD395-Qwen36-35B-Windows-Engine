"""Bind actual branch references to both unchanged GB10 controls and history."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from capture_gb10_token_matrix import fingerprints, fixtures  # noqa: E402


class PartialPrefixReferenceTests(unittest.TestCase):
    def test_controls_and_divergent_inputs_are_actual_frozen_histories(self):
        path = ROOT / "contracts/gb10_partial_prefix_actual_tokens_20260912_oracle.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "240855912800eb232b0149fb7696c8bd6af395e337e7f8dbb9ea6345d99a2bc7")
        record = json.loads(path.read_text())
        materialized, oracles = fixtures({
            7169: ROOT / "contracts/arbitrary_q7169_gb10_oracle.json",
            8192: ROOT / "contracts/hprefill_q8192_gb10_oracle.json",
        })
        self.assertTrue(record["controls_qualified"])
        self.assertFalse(record["old_control_oracles_changed"])
        self.assertFalse(record["windows_acceptance"])
        self.assertFalse(record["reference_outputs_injected"])
        self.assertFalse(record["native_tensor_inputs"])
        self.assertFalse(record["prefix_caching"])
        self.assertEqual(len(record["controls"]), 2)
        for control, length in zip(record["controls"], (7169, 8192)):
            self.assertEqual(control["name"], f"q{length}-out32")
            self.assertTrue(control["full_matrix_case_pass"])
            expected = oracles[length]["expected"]
            self.assertEqual(control["output_token_ids"], expected["output_token_ids"])
            self.assertEqual(control["first_token_raw_logit"], expected["first_token_raw_logit"])
        owner = materialized[0]["prompt_token_ids"]
        self.assertEqual(record["owner"]["u32le_sha256"], fingerprints(owner)["u32le_sha256"])
        self.assertEqual([(c["prefix_tokens"], len(c["suffix_token_ids"]))
                          for c in record["cases"]], [(6208, 1), (6656, 1), (7168, 1), (7168, 5)])
        for case in record["cases"]:
            prefix, prompt = case["prefix_tokens"], case["prompt"]["token_ids"]
            self.assertEqual(prompt[:prefix], owner[:prefix])
            self.assertNotEqual(prompt[prefix], owner[prefix])
            self.assertEqual(prompt[prefix:], case["suffix_token_ids"])
            self.assertEqual(len(prompt), case["prompt"]["token_count"])
            for key, value in fingerprints(prompt).items():
                self.assertEqual(case["prompt"][key], value)
            expected, raw = case["expected"], case["raw_logits"]
            self.assertEqual(len(expected["output_token_ids"]), 32)
            self.assertEqual(expected["first_token_raw_logit_tolerance"], 0.125)
            self.assertEqual(raw["raw_argmax_token"], expected["output_token_ids"][0])
            self.assertEqual(raw["raw_logit"], expected["first_token_raw_logit"])
            self.assertFalse(raw["observer_modifies_output"])
            self.assertFalse(raw["sampling_boundary"]["discarded"])
            self.assertEqual(raw["sampling_boundary"]["processed_tokens"], len(prompt))
            for key, value in fingerprints(expected["output_token_ids"]).items():
                self.assertEqual(expected["output_token_ids_" + key], value)
            rows = case["qualified_rows"]
            self.assertEqual([r["position"] for r in rows], [len(prompt) - 1, len(prompt)])
            self.assertEqual([r["input_token_id"] for r in rows],
                             [prompt[-1], expected["output_token_ids"][0]])
            self.assertTrue(all(r["matches_generated_history"] for r in rows))

    def test_postprocessing_recovery_does_not_invent_a_host_measurement(self):
        record = json.loads((ROOT / "contracts/gb10_partial_prefix_actual_tokens_20260912_oracle.json").read_text())
        recovery = record["record_recovery"]
        self.assertEqual(recovery["model_container_exit_code"], 0)
        self.assertTrue(recovery["controller_postprocessing_timed_out"])
        self.assertTrue(recovery["original_dispatch_record_missing"])
        self.assertTrue(recovery["recovery_read_only"])
        self.assertTrue(recovery["all_missing_binaries_recovered_and_verified"])
        self.assertIsNone(recovery["minimum_host_available_bytes"])
        self.assertTrue(record["binary_capture_verification"]["all_capture_hashes_verified"])
        self.assertEqual(record["binary_capture_verification"]["binary_files"], 3000)


if __name__ == "__main__":
    unittest.main()
