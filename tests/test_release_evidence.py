import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseEvidenceTests(unittest.TestCase):
    def test_openai_acceptance_covers_product_surface_and_prompt_matrix(self) -> None:
        path = ROOT / "benchmarks" / "openai" / "openai-http-acceptance-v1.0.0.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(value["passed"])
        checks = {check["name"]: check for check in value["checks"]}
        self.assertTrue(all(check["passed"] for check in checks.values()))
        self.assertEqual(
            {
                "completion_sse",
                "chat_sse",
                "structured_tool_call",
                "tool_result_continuation",
                "structured_tool_call_sse",
                "bounded_fifo_request_queue",
                "continuous_prompt_length_matrix",
                "context_limit_rejection",
                "http_prefix_reuse",
            }
            - checks.keys(),
            set(),
        )

        matrix = checks["continuous_prompt_length_matrix"]["details"]
        expected_lengths = [
            262143,
            131073,
            65537,
            32769,
            16385,
            8193,
            8192,
            8191,
            1025,
            17,
        ]
        self.assertEqual(matrix["tested_lengths"], expected_lengths)
        self.assertEqual(
            [case["prompt_tokens"] for case in matrix["cases"]], expected_lengths
        )
        self.assertTrue(all(case["gb10_text_match"] for case in matrix["cases"]))

        queue = checks["bounded_fifo_request_queue"]["details"]
        self.assertEqual(queue["request_count"], 3)
        self.assertGreaterEqual(queue["queue_waiting_observed"], 2)
        self.assertEqual(queue["queue_after"]["rejected_total"], 0)
        self.assertEqual(queue["queue_after"]["timed_out_total"], 0)

        prefix = checks["http_prefix_reuse"]["details"]
        self.assertTrue(prefix["repeat_output_match"])
        self.assertTrue(prefix["unrelated_prefix_guard"])
        self.assertEqual(
            prefix["probe_order"],
            ["shared_a", "shared_b", "unrelated", "shared_a_repeat"],
        )
        self.assertTrue(
            any(marker["kind"] == "seed" for marker in prefix["markers"])
        )
        self.assertTrue(
            any(marker["kind"] == "hit" for marker in prefix["markers"])
        )

    def test_product_matrix_contains_twelve_correctness_attached_rows(self) -> None:
        path = ROOT / "benchmarks" / "performance" / "product-matrix-v1.0.0.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "pass")
        self.assertEqual(len(value["rows"]), 12)
        q8192 = next(row for row in value["rows"] if row["id"] == "cold-q8192-peak")
        self.assertLessEqual(q8192["ttft_ms"], value["acceptance"]["q8192_ttft_max_ms"])
        self.assertGreaterEqual(
            q8192["prefill_tokens_per_second"],
            value["acceptance"]["q8192_prefill_min_tokens_per_second"],
        )
        self.assertLessEqual(q8192["load_ms"], value["acceptance"]["load_max_ms"])
        self.assertTrue(value["q8192_correctness"]["pass"])

    def test_mmlu_public_rows_match_summary_and_hash(self) -> None:
        directory = ROOT / "benchmarks" / "eval"
        summary = json.loads(
            (directory / "mmlu-pro-summary-v1.0.0.json").read_text(encoding="utf-8")
        )
        rows_path = directory / "mmlu-pro-full-parity-v1.0.0.jsonl"
        payload = rows_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(), summary["published_rows"]["sha256"]
        )
        counters = {
            "rows": 0,
            "candidate_correct": 0,
            "reference_correct": 0,
            "candidate_parsed": 0,
            "reference_parsed": 0,
            "prediction_agreement": 0,
            "correctness_agreement": 0,
        }
        for line in payload.splitlines():
            row = json.loads(line)
            self.assertNotIn("endpoint", row)
            self.assertNotIn("content", row)
            counters["rows"] += 1
            counters["candidate_correct"] += int(row["candidate"]["correct"] is True)
            counters["reference_correct"] += int(row["reference"]["correct"] is True)
            counters["candidate_parsed"] += int(row["candidate"]["parsed"] is True)
            counters["reference_parsed"] += int(row["reference"]["parsed"] is True)
            counters["prediction_agreement"] += int(row["agreement"]["prediction"])
            counters["correctness_agreement"] += int(row["agreement"]["correctness"])
        for field, value in counters.items():
            self.assertEqual(value, summary["score"][field])

    def test_external_continuation_cases_are_complete(self) -> None:
        path = ROOT / "benchmarks" / "correctness" / "gb10-continuations-v1.0.0.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "pass")
        self.assertEqual(len(value["cases"]), 3)
        self.assertTrue(all(case["pass"] for case in value["cases"]))
        self.assertTrue(
            all(case["matching_tokens_per_request"] == 512 for case in value["cases"])
        )


if __name__ == "__main__":
    unittest.main()
