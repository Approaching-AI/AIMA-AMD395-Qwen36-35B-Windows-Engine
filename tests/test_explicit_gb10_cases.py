from pathlib import Path
import copy
import importlib.util
import json
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
                "name": ["../escape", "q7169-out32", "", "x" * 65, 7],
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
