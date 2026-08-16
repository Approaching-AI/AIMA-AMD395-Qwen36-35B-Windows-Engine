import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_prefill_length_smoothness.py"
SPEC = importlib.util.spec_from_file_location("verify_prefill_smoothness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrefillLengthSmoothnessTests(unittest.TestCase):
    def test_cases_are_cold_and_control_content_within_triplets(self) -> None:
        cases = MODULE.build_cases((4096, 8192), 2, 123)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["first_token_id"] for case in cases}), 12)
        for repetition in range(2):
            for center in (4096, 8192):
                group = [
                    case
                    for case in cases
                    if case["repetition"] == repetition
                    and case["center_prompt_tokens"] == center
                ]
                self.assertEqual({case["delta"] for case in group}, {-1, 0, 1})
                self.assertEqual(len({case["content_seed"] for case in group}), 1)

    def test_summary_rejects_an_eleven_percent_boundary_cliff(self) -> None:
        records = []
        for prompt_tokens, ttft_ms in ((8191, 1000.0), (8192, 1000.0), (8193, 1110.0)):
            records.append(
                {
                    "center_prompt_tokens": 8192,
                    "prompt_tokens": prompt_tokens,
                    "ttft_ms": ttft_ms,
                }
            )
        summary = MODULE.summarize(records)
        self.assertFalse(summary["local_gates"][0]["passed"])
        self.assertFalse(summary["all_gates_passed"])

    def test_summary_accepts_smooth_triplets_and_global_throughput(self) -> None:
        records = []
        for center, base_ms in ((4096, 2048.0), (8192, 4096.0)):
            for delta, adjustment in ((-1, 5.0), (0, 0.0), (1, 8.0)):
                records.append(
                    {
                        "center_prompt_tokens": center,
                        "prompt_tokens": center + delta,
                        "ttft_ms": base_ms + adjustment,
                    }
                )
        summary = MODULE.summarize(records)
        self.assertTrue(summary["all_local_gates_passed"])
        self.assertTrue(summary["global_throughput_gate"]["passed"])
        self.assertTrue(summary["all_gates_passed"])


if __name__ == "__main__":
    unittest.main()
