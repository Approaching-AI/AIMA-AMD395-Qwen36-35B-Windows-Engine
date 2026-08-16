import hashlib
import json
from pathlib import Path
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseEvidenceTests(unittest.TestCase):
    def test_q8192_neighbor_product_gate_is_complete_and_bound(self) -> None:
        performance_path = (
            ROOT
            / "benchmarks"
            / "performance"
            / "q8192-neighbor-product-gate-amd395-v1.0.1.json"
        )
        value = json.loads(performance_path.read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "pass")
        self.assertRegex(value["source"]["commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(value["platform"]["host"], "baiying")
        self.assertEqual(value["model"]["id"], "qwen3.6-35b-a3b")

        reference_path = (
            ROOT
            / "benchmarks"
            / "correctness"
            / "gb10-q8192-neighbor-continuation-reference-v1.0.1.json"
        )
        verifier_path = ROOT / "scripts" / "verify_q8192_neighbor_continuity.py"
        self.assertEqual(
            hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            value["gb10_reference"]["sha256"],
        )
        self.assertEqual(
            hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
            value["command"]["script_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "native" / "providers" / "whole_provider.cpp").read_bytes()
            ).hexdigest(),
            value["source"]["whole_provider_source_sha256"],
        )
        self.assertEqual(
            hashlib.sha256((ROOT / "engine" / "runtime.env").read_bytes()).hexdigest(),
            value["source"]["runtime_profile_source_sha256"],
        )
        self.assertEqual(
            value["acceptance"]["neighbor_to_q8192_median_ratio_max"],
            1.10,
        )
        self.assertEqual(
            value["acceptance"]["neighbor_positive_residual_max_ms"],
            500.0,
        )
        self.assertEqual(value["acceptance"]["q8192_ttft_max_ms"], 4187.416)

        cases = value["cases"]
        self.assertEqual(len(cases), 18)
        self.assertEqual([case["ordinal"] for case in cases], list(range(18)))
        self.assertTrue(all(case["gb10_match"] for case in cases))
        expected_cohorts = {
            (prompt_tokens, max_tokens): 3
            for prompt_tokens in (8191, 8192, 8193)
            for max_tokens in (1, 2)
        }
        actual_cohorts = {
            key: sum(
                case["prompt_tokens"] == key[0]
                and case["max_tokens"] == key[1]
                for case in cases
            )
            for key in expected_cohorts
        }
        self.assertEqual(actual_cohorts, expected_cohorts)
        self.assertEqual(
            value["summary"]["gb10_matching_cases"],
            len(cases),
        )

        q8192_ttft = [
            case["ttft_ms"] for case in cases if case["prompt_tokens"] == 8192
        ]
        self.assertAlmostEqual(
            statistics.median(q8192_ttft),
            value["summary"]["q8192_all_six_median_ttft_ms"],
            places=6,
        )
        self.assertLessEqual(
            max(q8192_ttft), value["acceptance"]["q8192_ttft_max_ms"]
        )
        for gate in value["continuity_gates"]:
            self.assertTrue(gate["passed"])
            self.assertLessEqual(
                gate["neighbor_to_center_ratio"],
                value["acceptance"]["neighbor_to_q8192_median_ratio_max"],
            )
            self.assertLessEqual(
                gate["positive_residual_ms"],
                value["acceptance"]["neighbor_positive_residual_max_ms"],
            )

    def test_prefill_length_smoothness_gate_is_complete_and_bound(self) -> None:
        performance_path = (
            ROOT
            / "benchmarks"
            / "performance"
            / "prefill-length-smoothness-amd395-v1.0.1.json"
        )
        value = json.loads(performance_path.read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "pass")
        self.assertEqual(value["platform"]["host"], "baiying")
        self.assertEqual(value["model"]["id"], "qwen3.6-35b-a3b")
        self.assertRegex(value["source"]["commit"], r"^[0-9a-f]{40}$")

        verifier_path = ROOT / "scripts" / "verify_prefill_length_smoothness.py"
        whole_provider_path = ROOT / "native" / "providers" / "whole_provider.cpp"
        smooth_tail_source_path = (
            ROOT
            / "native"
            / "providers"
            / "triton_moe"
            / "qrt_triton_moe_q8192_provider.cpp"
        )
        smooth_tail_generator_path = (
            ROOT
            / "native"
            / "generators"
            / "compile_q8192_triton_selected_moe.py"
        )
        self.assertEqual(
            hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
            value["command"]["script_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(whole_provider_path.read_bytes()).hexdigest(),
            value["source"]["whole_provider_source_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(smooth_tail_source_path.read_bytes()).hexdigest(),
            value["source"]["smooth_tail_provider_source_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(smooth_tail_generator_path.read_bytes()).hexdigest(),
            value["source"]["smooth_tail_generator_sha256"],
        )

        acceptance = value["acceptance"]
        self.assertEqual(acceptance["local_ttft_max_to_min_ratio_max"], 1.10)
        self.assertEqual(acceptance["local_positive_residual_max_ms"], 500.0)
        self.assertEqual(
            acceptance["global_median_throughput_max_to_min_ratio_max"],
            1.30,
        )
        self.assertEqual(value["correctness"]["case_count"], 72)
        self.assertEqual(value["correctness"]["gb10_matching_cases"], 72)
        self.assertTrue(value["correctness"]["passed"])
        self.assertEqual(value["summary"]["case_count"], 72)
        self.assertEqual(value["summary"]["gb10_matching_cases"], 72)
        self.assertTrue(value["summary"]["correctness_passed"])
        self.assertTrue(value["summary"]["continuity_passed"])

        centers = [4096, 6144, 8192, 9216, 10240, 12288, 14336, 16384]
        gates = value["local_continuity_gates"]
        self.assertEqual([gate["center_prompt_tokens"] for gate in gates], centers)
        for gate in gates:
            center = gate["center_prompt_tokens"]
            self.assertEqual(gate["neighbor_triplet"], [center - 1, center, center + 1])
            self.assertTrue(gate["passed"])
            self.assertLessEqual(
                gate["ttft_max_to_min_ratio"],
                acceptance["local_ttft_max_to_min_ratio_max"],
            )
            self.assertLessEqual(
                gate["positive_residual_ms"],
                acceptance["local_positive_residual_max_ms"],
            )

        expected_lengths = [
            length
            for center in centers
            for length in (center - 1, center, center + 1)
        ]
        cohorts = value["cohorts"]
        self.assertEqual(
            [cohort["prompt_tokens"] for cohort in cohorts],
            expected_lengths,
        )
        self.assertTrue(all(len(cohort["samples_ms"]) == 3 for cohort in cohorts))
        global_gate = value["global_throughput_gate"]
        self.assertTrue(global_gate["passed"])
        self.assertLessEqual(
            global_gate["median_throughput_max_to_min_ratio"],
            acceptance["global_median_throughput_max_to_min_ratio_max"],
        )

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
