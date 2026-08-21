from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_prefill_random_length_plateaus.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_prefill_random_length_plateaus", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(MODULE)


def record(
    pair_id: int,
    kind: str,
    prompt_tokens: int,
    speed: float,
    *,
    gb10_match: bool = True,
    ttft_ms: float | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "pair_id": pair_id,
        "interval_index": 0,
        "kind": kind,
        "prompt_tokens": prompt_tokens,
        "prefill_tokens_per_second": speed,
        "gb10_match": gb10_match,
    }
    if ttft_ms is not None:
        result["ttft_ms"] = ttft_ms
    return result


def qualified_service_record() -> dict[str, object]:
    digest = "a" * 64
    oracle, oracle_sha256 = MODULE.load_gb10_q8192_oracle()
    model_path = r"D:\models\Qwen3.6-35B-A3B"
    return {
        "record_type": "qrt_random_length_terminal_service_start",
        "host": "BAIYING",
        "repo_commit": "6cdce444177f54e8985a2d6148da863369bece32",
        "model_path": model_path,
        "model_id": "qwen3.6-35b-a3b",
        "gdn_route": "aiter",
        "engine_self_hashes_diagnostic_only": True,
        "q8192_correctness_pass": True,
        "q8192_performance_pass": True,
        "q8192_model_engine_load_pass": True,
        "q8192_route_pass": True,
        "q8192_moe_fixed_route_pass": True,
        "q8192_continuation_token_for_token_pass": True,
        "q8192_decode_route_pass": True,
        "q8192_prefix_continuation_pass": True,
        "q8192_prefix_continuation_correctness_pass": True,
        "q8192_prefix_continuation_performance_pass": True,
        "q8192_prefix_route_pass": True,
        "model_engine_load_pass": True,
        "dynamic_logical_moe_provider": True,
        "q8192_moe_dynamic_logical_component_pass": True,
        "q8192_moe_dynamic_logical_q8192_exact_pass": True,
        "q8192_moe_dynamic_logical_case_count": 32,
        "q8192_moe_expected_full_provider_hash_diagnostic_only": True,
        "q8192_moe_dynamic_logical_timing_samples": 3,
        "q8192_moe_dynamic_logical_timing_stat": "median",
        "q8192_moe_dynamic_logical_nonfinite": 0,
        "q8192_moe_dynamic_logical_component_only": True,
        "q8192_moe_dynamic_logical_reset_included": False,
        "attention_dynamic_logical_case_count": 32,
        "attention_dynamic_logical_reset_included": False,
        "attention_dynamic_logical_component_only": True,
        "gdn_dynamic_logical_case_count": 32,
        "gdn_dynamic_logical_mode_count": 64,
        "gdn_dynamic_logical_reset_included": False,
        "gdn_dynamic_logical_component_only": True,
        "model_engine_load_ms": 29_000.0,
        "model_engine_load_max_ms": 30_000.0,
        "q8192_model_engine_load_ms": 20_000.0,
        "q8192_prefill_tokens_per_second": 2048.0,
        "q8192_prefill_metric_relative_error": 0.0,
        "q8192_prefill_metric_relative_error_max": 0.001,
        "q8192_prefill_min_tokens_per_second": 1506.407,
        "q8192_ttft_ms": 4000.0,
        "q8192_ttft_max_ms": 4187.416,
        "q8192_decode_tokens_per_second": 30.0,
        "q8192_decode_min_tokens_per_second": 28.168,
        "q8192_tpot_ms": 33.0,
        "q8192_tpot_max_ms": 35.502,
        "q8192_prefix_decode_tokens_per_second": 30.0,
        "q8192_prefix_tpot_ms": 33.0,
        "q8192_prefix_model_engine_load_ms": 20_000.0,
        "q8192_first_token": 144,
        "q8192_first_token_raw_logit": 10.375,
        "q8192_first_token_raw_logit_absolute_difference": 0.1,
        "q8192_first_token_raw_logit_tolerance": 0.125,
        "q8192_gb10_oracle_sha256": oracle_sha256,
        "q8192_gb10_oracle_capture_request_sha256": oracle["capture"][
            "request_sha256"
        ],
        "q8192_gb10_oracle_capture_response_sha256": oracle["capture"][
            "response_sha256"
        ],
        "q8192_gb10_oracle_capture_command_file_sha256": oracle["capture"][
            "command_file_sha256"
        ],
        "q8192_gb10_continuation_capture_request_sha256": oracle[
            "continuation_capture"
        ]["request_sha256"],
        "q8192_gb10_continuation_capture_response_sha256": oracle[
            "continuation_capture"
        ]["response_sha256"],
        "q8192_gb10_continuation_capture_command_file_sha256": oracle[
            "continuation_capture"
        ]["command_file_sha256"],
        "q8192_expected_output_token_ids_u32le_sha256": (
            MODULE.GB10_Q8192_CONTINUATION_SHA256
        ),
        "q8192_expected_output_token_ids_u32le_fnv1a64": (
            MODULE.GB10_Q8192_CONTINUATION_FNV1A64
        ),
        "q8192_continuation_token_count": 32,
        "q8192_decode_step_count": 31,
        "q8192_prefix_tokens": 8191,
        "q8192_prefix_suffix_tokens": 1,
        "q8192_prefix_hit_count": 2,
        "server_prefix_cache_enabled": False,
        "exact_first_token_prefill": True,
        "exact_letter_classifier": False,
        "cold_prefix_contract": (
            "server_prefix_cache_disabled_and_globally_unique_first_token_id"
        ),
        "q8192_product_record_sha256": digest,
        "qualified_environment_sha256": digest,
        "service_environment_sha256": digest,
        "q8192_moe_build_provenance_sha256": digest,
        "q8192_moe_provider_sha256": digest,
        "whole_provider_sha256": digest,
        "attention_provider_sha256": digest,
        "attention_build_provenance": (
            r"D:\runtime\ck-fmha\build-provenance.json"
        ),
        "attention_build_provenance_sha256": digest,
        "gdn_build_provenance": (
            r"D:\runtime\aiter-gdn\qualification-provenance.json"
        ),
        "gdn_build_provenance_sha256": digest,
        "gdn_provider": r"D:\runtime\aiter-gdn\provider.dll",
        "gdn_provider_sha256": digest,
        "gdn_kernel_dir": r"D:\runtime\aiter-gdn",
        "gdn_artifacts": [
            {
                "path": rf"D:\runtime\aiter-gdn\{name}",
                "file": name,
                "bytes": index + 1,
                "sha256": digest,
            }
            for index, name in enumerate(MODULE.GDN_EXPECTED_ARTIFACTS)
        ],
        "q8192_moe_artifacts": [
            {
                "file": name,
                "bytes": index + 1,
                "sha256": digest,
            }
            for index, name in enumerate(MODULE.Q8192_MOE_EXPECTED_ARTIFACTS)
        ],
        "command": [
            r"D:\runtime\engine\qrt.exe",
            "start",
            "--model",
            model_path,
        ],
    }


class RandomLengthPlateauTests(unittest.TestCase):
    def test_verifier_uses_the_qwen36_runtime_vocabulary_contract(self) -> None:
        self.assertEqual(MODULE.QWEN36_VOCAB_SIZE, 248_320)

    def test_service_evidence_validator_binds_final_runtime_and_artifacts(self) -> None:
        service = qualified_service_record()
        contract = MODULE.validate_amd_service_record(
            service,
            amd_host="baiying",
            repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
            model_path=r"D:\models\Qwen3.6-35B-A3B",
            model_id="qwen3.6-35b-a3b",
        )
        self.assertTrue(contract["model_engine_load_pass"])
        self.assertEqual(
            contract["q8192_gb10_oracle_sha256"],
            MODULE.sha256_file(MODULE.GB10_Q8192_ORACLE_PATH),
        )
        self.assertFalse(contract["server_prefix_cache_enabled"])

        cached_service = copy.deepcopy(service)
        cached_service["server_prefix_cache_enabled"] = True
        with self.assertRaisesRegex(
            MODULE.VerificationError, "qualified run"
        ):
            MODULE.validate_amd_service_record(
                cached_service,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        self_hash_gated_service = copy.deepcopy(service)
        self_hash_gated_service["engine_self_hashes_diagnostic_only"] = False
        with self.assertRaisesRegex(
            MODULE.VerificationError, "qualified run"
        ):
            MODULE.validate_amd_service_record(
                self_hash_gated_service,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        contaminated_component_timing = copy.deepcopy(service)
        contaminated_component_timing[
            "q8192_moe_dynamic_logical_reset_included"
        ] = True
        with self.assertRaisesRegex(
            MODULE.VerificationError, "qualified run"
        ):
            MODULE.validate_amd_service_record(
                contaminated_component_timing,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        nonfinite_moe_component = copy.deepcopy(service)
        nonfinite_moe_component["q8192_moe_dynamic_logical_nonfinite"] = 1
        with self.assertRaisesRegex(
            MODULE.VerificationError, "qualified run"
        ):
            MODULE.validate_amd_service_record(
                nonfinite_moe_component,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        single_sample_moe_component = copy.deepcopy(service)
        single_sample_moe_component[
            "q8192_moe_dynamic_logical_timing_samples"
        ] = 1
        with self.assertRaisesRegex(
            MODULE.VerificationError, "qualified run"
        ):
            MODULE.validate_amd_service_record(
                single_sample_moe_component,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        hash_gated_moe_component = copy.deepcopy(service)
        hash_gated_moe_component[
            "q8192_moe_expected_full_provider_hash_diagnostic_only"
        ] = False
        with self.assertRaisesRegex(
            MODULE.VerificationError, "qualified run"
        ):
            MODULE.validate_amd_service_record(
                hash_gated_moe_component,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        mislabeled_moe_component = copy.deepcopy(service)
        mislabeled_moe_component[
            "q8192_moe_dynamic_logical_component_only"
        ] = False
        with self.assertRaisesRegex(
            MODULE.VerificationError, "qualified run"
        ):
            MODULE.validate_amd_service_record(
                mislabeled_moe_component,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        contaminated_attention_timing = copy.deepcopy(service)
        contaminated_attention_timing[
            "attention_dynamic_logical_reset_included"
        ] = True
        with self.assertRaisesRegex(
            MODULE.VerificationError, "qualified run"
        ):
            MODULE.validate_amd_service_record(
                contaminated_attention_timing,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        contaminated_gdn_timing = copy.deepcopy(service)
        contaminated_gdn_timing[
            "gdn_dynamic_logical_reset_included"
        ] = True
        with self.assertRaisesRegex(
            MODULE.VerificationError, "qualified run"
        ):
            MODULE.validate_amd_service_record(
                contaminated_gdn_timing,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        substituted_gdn_artifact = copy.deepcopy(service)
        substituted_gdn_artifact["gdn_artifacts"][0]["file"] = "other.dll"
        with self.assertRaisesRegex(
            MODULE.VerificationError, "unexpected AITER GDN artifact names"
        ):
            MODULE.validate_amd_service_record(
                substituted_gdn_artifact,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        unbound_gdn_provider = copy.deepcopy(service)
        unbound_gdn_provider["gdn_artifacts"][0]["sha256"] = "b" * 64
        with self.assertRaisesRegex(
            MODULE.VerificationError, "GDN provider hash is not artifact-bound"
        ):
            MODULE.validate_amd_service_record(
                unbound_gdn_provider,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        invalid_load = copy.deepcopy(service)
        invalid_load["model_engine_load_ms"] = 30_000.001
        with self.assertRaisesRegex(
            MODULE.VerificationError, "final service load exceeds"
        ):
            MODULE.validate_amd_service_record(
                invalid_load,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        invalid_oracle = copy.deepcopy(service)
        invalid_oracle["q8192_gb10_oracle_sha256"] = "b" * 64
        with self.assertRaisesRegex(
            MODULE.VerificationError, "qualified run"
        ):
            MODULE.validate_amd_service_record(
                invalid_oracle,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        invalid_artifact = copy.deepcopy(service)
        invalid_artifact["q8192_moe_artifacts"][0]["sha256"] = "not-a-hash"
        with self.assertRaisesRegex(
            MODULE.VerificationError, "artifact binding"
        ):
            MODULE.validate_amd_service_record(
                invalid_artifact,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

    def test_service_validator_binds_native_shared_runtime_and_aot(self) -> None:
        service = qualified_service_record()
        digest = "a" * 64
        service["dynamic_moe_native_shared_selected"] = True
        service["dynamic_moe_base_aot_mode"] = (
            "generated_from_current_source"
        )
        service["dynamic_moe_base_aot_block_m"] = 64
        service["dynamic_moe_base_aot_group_m"] = 1
        service["dynamic_moe_base_aot_dynamic_logical_abi"] = True
        service["q8192_moe_required_runtime_environment"] = [
            {"name": name, "value": value}
            for name, value in (
                MODULE.Q8192_MOE_NATIVE_SHARED_REQUIRED_RUNTIME_ENVIRONMENT
            )
        ]
        service["q8192_moe_environment"] = [
            name
            for name, _ in (
                MODULE.Q8192_MOE_NATIVE_SHARED_REQUIRED_RUNTIME_ENVIRONMENT
            )
        ]
        service["q8192_moe_artifacts"].extend(
            {
                "file": name,
                "bytes": index + 100,
                "sha256": digest,
            }
            for index, name in enumerate(
                MODULE.Q8192_MOE_NATIVE_SHARED_AUXILIARY_ARTIFACTS
            )
        )
        validated = MODULE.validate_amd_service_record(
            service,
            amd_host="baiying",
            repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
            model_path=r"D:\models\Qwen3.6-35B-A3B",
            model_id="qwen3.6-35b-a3b",
        )
        self.assertTrue(validated["dynamic_moe_native_shared_selected"])

        wrong_environment = copy.deepcopy(service)
        wrong_environment["q8192_moe_required_runtime_environment"][0][
            "value"
        ] = "0"
        with self.assertRaisesRegex(
            MODULE.VerificationError, "runtime environment differs"
        ):
            MODULE.validate_amd_service_record(
                wrong_environment,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        stale_base_aot = copy.deepcopy(service)
        stale_base_aot["dynamic_moe_base_aot_group_m"] = 8
        with self.assertRaisesRegex(
            MODULE.VerificationError, "base AOT differs"
        ):
            MODULE.validate_amd_service_record(
                stale_base_aot,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

        missing_aot = copy.deepcopy(service)
        missing_aot["q8192_moe_artifacts"].pop()
        with self.assertRaisesRegex(
            MODULE.VerificationError, "incomplete q8192 MoE artifact"
        ):
            MODULE.validate_amd_service_record(
                missing_aot,
                amd_host="baiying",
                repo_commit="6cdce444177f54e8985a2d6148da863369bece32",
                model_path=r"D:\models\Qwen3.6-35B-A3B",
                model_id="qwen3.6-35b-a3b",
            )

    def test_build_cases_keeps_each_random_bracketed_after_pair_shuffle(self) -> None:
        cases = MODULE.build_cases((2048, 2560, 3072), 6, 39536119)
        self.assertEqual(len(cases), 36)
        for offset in range(0, len(cases), 3):
            triplet = cases[offset : offset + 3]
            self.assertEqual(
                [case["kind"] for case in triplet],
                ["upper_anchor_before", "random", "upper_anchor_after"],
            )
            self.assertEqual(len({case["pair_id"] for case in triplet}), 1)
            self.assertEqual(triplet[0]["prompt_tokens"], triplet[2]["prompt_tokens"])

    def test_choose_random_lengths_covers_boundaries_and_integer_spikes(self) -> None:
        generator = MODULE.random.Random(39536119)
        lengths = MODULE.choose_random_lengths(2048, 2560, 6, generator)
        self.assertEqual(len(lengths), 6)
        self.assertIn(2049, lengths)
        self.assertIn(2559, lengths)
        aligned = [
            value
            for value in lengths
            if value % MODULE.INTEGER_SPIKE_ALIGNMENT == 0
        ]
        self.assertEqual(len(aligned), 1)
        self.assertIn(aligned[0] - 1, lengths)
        self.assertIn(aligned[0] + 1, lengths)
        self.assertTrue(
            any(
                value % MODULE.INTEGER_SPIKE_ALIGNMENT not in (0, 1, 63)
                for value in lengths
            )
        )

    def test_build_cases_supports_fresh_globally_unique_first_tokens(self) -> None:
        cases = MODULE.build_cases(
            (2048, 2560, 3072), 6, 20260817, first_token_base=50_000
        )
        first_token_ids = [case["first_token_id"] for case in cases]
        self.assertEqual(len(first_token_ids), len(set(first_token_ids)))
        self.assertEqual(min(first_token_ids), 50_000)
        self.assertEqual(max(first_token_ids), 50_000 + len(cases) - 1)
        warmup_first_token = 50_000 + len(cases)
        self.assertNotIn(warmup_first_token, first_token_ids)
        self.assertLess(warmup_first_token, MODULE.QWEN36_VOCAB_SIZE)

    def test_default_matrix_contains_all_234_measurements(self) -> None:
        cases = MODULE.build_cases(
            MODULE.DEFAULT_BOUNDARIES,
            MODULE.MINIMUM_RANDOM_CASES_PER_INTERVAL,
            39_536_119,
        )
        self.assertEqual(len(cases), 234)
        self.assertEqual(
            sum(case["kind"] == "random" for case in cases),
            78,
        )
        self.assertEqual(
            len({case["first_token_id"] for case in cases}),
            234,
        )
        self.assertEqual(
            {case["interval_index"] for case in cases},
            set(range(13)),
        )

    def test_ttft_metric_must_be_finite_and_positive(self) -> None:
        self.assertEqual(MODULE.require_positive_ttft_ms(1.25, "sample"), 1.25)
        for invalid in (None, 0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                MODULE.VerificationError, "finite positive"
            ):
                MODULE.require_positive_ttft_ms(invalid, "sample")

    def test_summarize_uses_geometric_bracket_reference(self) -> None:
        records = [
            record(0, "upper_anchor_before", 2560, 1000.0),
            record(0, "random", 2203, 999.8),
            record(0, "upper_anchor_after", 2560, 999.6),
            record(1, "upper_anchor_before", 2560, 1010.0),
            record(1, "random", 2419, 1009.5),
            record(1, "upper_anchor_after", 2560, 1009.0),
        ]
        summary = MODULE.summarize(records, (2048, 2560), 0.97, 1.03, 1.03)
        self.assertTrue(summary["correctness_passed"])
        self.assertTrue(summary["measurement_stability_passed"])
        self.assertTrue(summary["performance_passed"])
        self.assertEqual(summary["performance_eligible_pair_count"], 2)
        self.assertAlmostEqual(
            records[1]["paired_upper_anchor_prefill_tokens_per_second"],
            (1000.0 * 999.6) ** 0.5,
        )

    def test_anchor_drift_invalidates_performance_pair(self) -> None:
        records = [
            record(0, "upper_anchor_before", 2560, 1000.0),
            record(0, "random", 2203, 950.0),
            record(0, "upper_anchor_after", 2560, 900.0),
        ]
        summary = MODULE.summarize(records, (2048, 2560), 0.97, 1.03, 1.03)
        self.assertTrue(summary["correctness_passed"])
        self.assertFalse(summary["measurement_stability_passed"])
        self.assertFalse(summary["performance_passed"])
        self.assertEqual(summary["performance_eligible_pair_count"], 0)
        self.assertFalse(records[1]["performance_eligible"])

    def test_uniformly_slow_interval_fails_absolute_anchor_gate(self) -> None:
        records = [
            record(0, "upper_anchor_before", 2560, 1400.0),
            record(0, "random", 2203, 1400.0),
            record(0, "upper_anchor_after", 2560, 1400.0),
        ]
        summary = MODULE.summarize(
            records,
            (2048, 2560),
            0.97,
            1.03,
            1.03,
            MODULE.RETAINED_PREFILL_TOKENS_PER_SECOND,
            MODULE.RETAINED_Q8192_TTFT_MAX_MS,
        )
        interval = summary["intervals"][0]
        self.assertTrue(interval["relative_performance_passed"])
        self.assertFalse(interval["upper_anchor_absolute_performance_passed"])
        self.assertFalse(interval["performance_passed"])
        self.assertFalse(summary["performance_passed"])

    def test_random_point_below_retained_speed_fails_absolute_gate(self) -> None:
        records = [
            record(0, "upper_anchor_before", 2560, 1550.0),
            record(0, "random", 2203, 1506.0),
            record(0, "upper_anchor_after", 2560, 1550.0),
        ]
        summary = MODULE.summarize(
            records,
            (2048, 2560),
            0.97,
            1.03,
            1.03,
            MODULE.RETAINED_PREFILL_TOKENS_PER_SECOND,
            MODULE.RETAINED_Q8192_TTFT_MAX_MS,
        )
        interval = summary["intervals"][0]
        self.assertTrue(interval["relative_performance_passed"])
        self.assertTrue(interval["upper_anchor_absolute_performance_passed"])
        self.assertFalse(interval["random_absolute_performance_passed"])
        self.assertFalse(records[1]["absolute_performance_passed"])
        self.assertFalse(interval["performance_passed"])

    def test_q8192_anchor_ttft_gate_is_independent_of_ratio_and_speed(self) -> None:
        records = [
            record(
                0,
                "upper_anchor_before",
                8192,
                1950.0,
                ttft_ms=4200.0,
            ),
            record(0, "random", 7943, 1948.0, ttft_ms=4077.0),
            record(
                0,
                "upper_anchor_after",
                8192,
                1952.0,
                ttft_ms=4196.0,
            ),
        ]
        summary = MODULE.summarize(
            records,
            (7680, 8192),
            0.97,
            1.03,
            1.03,
            MODULE.RETAINED_PREFILL_TOKENS_PER_SECOND,
            MODULE.RETAINED_Q8192_TTFT_MAX_MS,
        )
        interval = summary["intervals"][0]
        self.assertTrue(interval["relative_performance_passed"])
        self.assertTrue(interval["upper_anchor_absolute_performance_passed"])
        self.assertFalse(interval["q8192_upper_anchor_ttft_passed"])
        self.assertFalse(interval["performance_passed"])
        self.assertFalse(summary["performance_passed"])

    def test_cli_defaults_preserve_retained_performance_targets(self) -> None:
        args = MODULE.parse_args(
            [
                "--amd-base-url",
                "http://127.0.0.1:18000",
                "--gb10-base-url",
                "http://127.0.0.1:18001",
                "--output",
                "unused.json",
            ]
        )
        self.assertEqual(
            args.upper_anchor_min_prefill_tokens_per_second,
            MODULE.RETAINED_PREFILL_TOKENS_PER_SECOND,
        )
        self.assertEqual(
            args.q8192_upper_anchor_ttft_max_ms,
            MODULE.RETAINED_Q8192_TTFT_MAX_MS,
        )
        self.assertEqual(args.amd_host, "baiying")
        self.assertEqual(args.gb10_host, "gb10-4t")
        self.assertEqual(args.amd_model_path, r"D:\models\Qwen3.6-35B-A3B")
        self.assertEqual(
            args.random_per_interval,
            MODULE.MINIMUM_RANDOM_CASES_PER_INTERVAL,
        )

    def test_cli_rejects_relaxed_retained_performance_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = str(Path(temporary_directory) / "report.json")
            common = [
                "--amd-base-url",
                "http://127.0.0.1:18000",
                "--gb10-base-url",
                "http://127.0.0.1:18001",
                "--output",
                output,
            ]
            with self.assertRaisesRegex(
                MODULE.VerificationError, "cannot relax the retained"
            ):
                MODULE.main(
                    common
                    + [
                        "--upper-anchor-min-prefill-tokens-per-second",
                        "1506.406",
                    ]
                )

            relaxed_shape_arguments = (
                ("--ratio-min", "0.96", "cannot relax"),
                ("--ratio-max", "1.04", "cannot relax"),
                (
                    "--anchor-max-to-min-ratio",
                    "1.04",
                    "cannot relax",
                ),
                (
                    "--random-per-interval",
                    "5",
                    "retained minimum",
                ),
            )
            for name, value, error_pattern in relaxed_shape_arguments:
                with self.subTest(name=name), self.assertRaisesRegex(
                    MODULE.VerificationError, error_pattern
                ):
                    MODULE.main(common + [name, value])

    def test_main_requires_qualified_amd_service_provenance(self) -> None:
        commit = "6cdce444177f54e8985a2d6148da863369bece32"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            common = [
                "--amd-base-url",
                "http://127.0.0.1:18000",
                "--gb10-base-url",
                "http://127.0.0.1:18001",
                "--output",
                str(root / "report.json"),
                "--repo-commit",
                commit,
            ]
            with self.assertRaisesRegex(
                MODULE.VerificationError, "--amd-service-record is required"
            ):
                MODULE.main(common)

            service_record = root / "start-record.json"
            service_record.write_text(
                '{"record_type":"qrt_random_length_terminal_service_start",'
                '"host":"wrong-host"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                MODULE.VerificationError,
                "AMD service record does not match the qualified run",
            ):
                MODULE.main(
                    common + ["--amd-service-record", str(service_record)]
                )
            with self.assertRaisesRegex(
                MODULE.VerificationError, "cannot relax the retained"
            ):
                MODULE.main(
                    common
                    + [
                        "--q8192-upper-anchor-ttft-max-ms",
                        "4187.417",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
