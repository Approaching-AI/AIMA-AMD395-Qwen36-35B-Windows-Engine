from pathlib import Path
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "explicit_capture", ROOT / "scripts/capture_gb10_token_matrix.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class ExplicitGb10CasesTests(unittest.TestCase):
    def test_runtime_branch_preflight_preserves_controls_without_gpu_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = root / "cases.json"
            item = {"name": "q65-actual-branch", "prompt_token_ids": [42] * 64 + [7],
                    "output_count": 32}
            cases.write_text(json.dumps([item]))
            command = [sys.executable, str(ROOT / "scripts/capture_gb10_token_matrix.py"),
                       "--source-commit", "a" * 40,
                       "--oracle-q7169", str(ROOT / "contracts/arbitrary_q7169_gb10_oracle.json"),
                       "--oracle-q8192", str(ROOT / "contracts/hprefill_q8192_gb10_oracle.json"),
                       "--additional-cases", str(cases), "--runtime-boundaries", "--output-dir"]
            subprocess.run(command + [str(root / "preflight")], check=True,
                           capture_output=True, text=True, timeout=10)
            record = json.loads((root / "preflight/preflight.json").read_text())
            self.assertEqual([x["name"] for x in record["fixtures"]],
                             ["q7169-out32", "q8192-out32", "q65-actual-branch"])
            self.assertFalse(record["completed"] or record["controls_qualified"])
            item["output_count"] = 1
            cases.write_text(json.dumps([item]))
            rejected = subprocess.run(command + [str(root / "invalid")], capture_output=True,
                                      text=True, timeout=10)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("first generated input", rejected.stderr)
            self.assertFalse((root / "invalid").exists())

    def test_real_tokens_keep_controls_and_reject_ambiguous_inputs(self):
        controls = [{"name": "q7169-out32"}, {"name": "q8192-out32"},
                    {"name": "unused-old-fixture"}]
        good = [{"name": "prefix-64-branch", "prompt_token_ids": [42] * 64 + [7],
                 "output_count": 32}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"

            def read(value):
                path.write_text(json.dumps(value))
                return capture.explicit_cases(path, controls)

            result = read(good)
            self.assertEqual(result[:2], controls[:2])
            self.assertEqual(result[2]["prompt_token_ids"], good[0]["prompt_token_ids"])
            self.assertFalse(result[2]["control"])
            self.assertEqual(result[2]["prompt"]["token_count"], 65)
            self.assertEqual(result[2]["prompt"]["u32le_fnv1a64"],
                             capture.fingerprints(good[0]["prompt_token_ids"])["u32le_fnv1a64"])
            invalid = [[], {}, good * 2, good * 13]
            for field, values in {
                "name": ["../escape", "q7169-out32", "unused-old-fixture", "", "x" * 65, 7],
                "prompt_token_ids": [[], [True], [-1], [248320], [1.0], [1] * 16385],
                "output_count": [0, 513, True, 1.5],
                "expected_token": [42],
            }.items():
                for value in values:
                    item = copy.deepcopy(good)
                    item[0][field] = value
                    invalid.append(item)
            for item in invalid:
                with self.subTest(item=str(item)[:90]), self.assertRaises(ValueError):
                    read(item)

    def test_additional_runtime_boundaries_require_actual_input_history(self):
        prompt = [42] * 64 + [7]
        worker = {"runtime_boundaries": {
            "selected_positions": [64, 65], "full_attention_cache": None,
            "transactions": [
                {"ordinal": 0, "first_position": 0, "input_token_ids": prompt,
                 "rows": [{"row": 64, "position": 64, "input_token_id": 7}]},
                {"ordinal": 1, "first_position": 65, "input_token_ids": [99],
                 "rows": [{"row": 0, "position": 65, "input_token_id": 99}]},
            ],
        }}
        capture.qualify_runtime_capture(worker, prompt, [99, 100])
        self.assertTrue(all(row["matches_generated_history"]
                            for tx in worker["runtime_boundaries"]["transactions"]
                            for row in tx["qualified_rows"]))
        for changed_prompt, changed_output in [(prompt, [98, 100]), ([43] + prompt[1:], [99, 100])]:
            with self.assertRaisesRegex(ValueError, "matching generated histories"):
                capture.qualify_runtime_capture(copy.deepcopy(worker), changed_prompt, changed_output)
