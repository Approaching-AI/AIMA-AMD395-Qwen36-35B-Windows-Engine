from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_prefill_random_length_route_log.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_prefill_random_length_route_log", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(MODULE)


def interval_upper(prompt_tokens: int) -> int:
    return 8192 if prompt_tokens == 8192 else 2560


def report(*lengths: int, passed: bool = True) -> dict[str, object]:
    return {
        "record_type": "qrt_prefill_random_length_plateau_verification",
        "passed": passed,
        "records": [
            {
                "prompt_tokens": length,
                "interval_upper_inclusive": interval_upper(length),
            }
            for length in lengths
        ],
    }


def execution_report(*lengths: int) -> dict[str, object]:
    execution_records = [
        {
            "prompt_tokens": length,
            "interval_upper_inclusive": interval_upper(length),
            "execution_position": position,
            "amd_token_ids": [144 + position],
        }
        for position, length in enumerate(lengths)
    ]
    return {
        "schema_version": 2,
        "record_type": "qrt_prefill_random_length_plateau_verification",
        "passed": True,
        "records": list(reversed(execution_records)),
    }


def report_with_warmup(*lengths: int) -> dict[str, object]:
    value = report(*lengths)
    value["request_policy"] = {"warmup_contract": MODULE.WARMUP_CONTRACT}
    value["warmup"] = {
        "record_type": "qrt_prefill_random_length_warmup",
        "measured": False,
        "prompt_tokens": 8192,
        "gb10_match": True,
        "server_prefix_cache_enabled": False,
        "passed": True,
        "amd_token_ids": [144],
        "gb10_token_ids": [144],
    }
    return value


def successful_lines(prompt_tokens: int, output_token: int = 144) -> list[str]:
    attention = (
        "full_attention_ck_q1_kv8192"
        if prompt_tokens == 8192
        else "full_attention_ck_q1_dynamic"
    )
    lines = [
        "BATCH_MARK qwen36_resident_request_thread_plan_ready "
        "thread_id=7 plans=10 elapsed_ms=0.1 pass=1",
    ]
    lines.extend(
        "BATCH_MARK full_attention_ck_compact_bf16 "
        f"layer={layer} target_tokens={prompt_tokens} "
        f"history_tokens={prompt_tokens} direct_export=1 "
        "exact_q16384=0 exact_q32768=0 exact_q65536=0 "
        "exact_q131_context=0 exact_arbitrary_dynamic=1 "
        "legacy_prepare=0 q_dense_bf16=1 k_inplace_bf16=1 "
        "v_direct_bf16=1 gated_context_bf16=1 tail_fused=1"
        for layer in MODULE.EXPECTED_DYNAMIC_FULL_ATTENTION_LAYERS
    )
    lines.extend(
        "BATCH_MARK q8192_aiter_fused_gdn_provider "
        f"layer={layer} tokens={prompt_tokens} exact_arbitrary_dynamic=1 "
        "fla_chunk_gdn_arithmetic=0 "
        "secondary_fla_chunk_gdn_provider=0 "
        "gdn_input=native_normalized_postconv async=1 "
        "full_sequence_recurrence=1 independent_q8192_state_merge=0"
        for layer in MODULE.EXPECTED_DYNAMIC_GDN_LAYERS
    )
    lines.extend(
        "BATCH_MARK q8192_triton_selected_moe_full_provider_v2 "
        f"layer={layer} selected_tokens={prompt_tokens} "
        f"selected_routes={prompt_tokens * 8} provider_tile_tokens=8192 "
        "provider_tile_count=1 "
        f"provider_tail_tokens={'0' if prompt_tokens == 8192 else prompt_tokens} "
        "provider_tail_padded=0 smooth_tail_moe=0 "
        "smooth_tail_rounded_tokens=0 smooth_tail_padding_tokens=0 "
        "smooth_tail_transaction_count=0 "
        "smooth_tail_dense_ceil_provider=0 "
        "dynamic_logical_moe_provider=1 "
        "short_lossless_palette_requested=0 short_weight_int8_requested=0"
        for layer in MODULE.EXPECTED_DYNAMIC_MOE_LAYERS
    )
    lines.extend(
        [
        f"BATCH_MARK {attention} layer=39 target_tokens=1 "
        f"history_tokens={prompt_tokens} runtime_kv_length={prompt_tokens} "
        "full_prefix_ck_launch=0 exact_terminal_correction=1",
        "BATCH_MARK layer39_q1_triton_0626_routed_backend "
        f"layer=39 prefill_tokens={prompt_tokens}",
        "BATCH_MARK q1_terminal_device_corridor_activate "
        f"layer=39 prefill_tokens={prompt_tokens} "
        "backend=triton_0626_raw_bf16",
        "BATCH_MARK q1_terminal_device_corridor_triton_metadata_upload "
        f"layer=39 prefill_tokens={prompt_tokens} "
        "global_topk_ids=1 topk_weights=1",
        "BATCH_MARK q1_terminal_device_corridor_output_activate "
        f"layer=39 prefill_tokens={prompt_tokens}",
        "BATCH_MARK q1_terminal_device_corridor_final_norm_activate "
        f"prefill_tokens={prompt_tokens}",
        ]
    )
    if prompt_tokens < 8192:
        lines.append(
            "QRT_SERVER_MARK exact_first_token_prefill "
            f"input_tokens={prompt_tokens} verifier_input_tokens={prompt_tokens} "
            f"output_token={output_token} requested_output_tokens=1 classifier=0 "
            "ttft_ms=1000.0 resident_prefix_mutated=0"
        )
    return lines


class RandomLengthRouteLogTests(unittest.TestCase):
    def test_audit_requires_every_occurrence_of_each_terminal_route(self) -> None:
        lines = successful_lines(2203) + successful_lines(8192) * 2
        audit = MODULE.audit_routes(
            report(2203, 8192, 8192),
            MODULE.parse_marker_lines("\n".join(lines)),
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["expected_request_count"], 3)
        self.assertFalse(audit["missing_requirements"])

    def test_audit_rejects_missing_dynamic_attention_marker(self) -> None:
        lines = [
            line
            for line in successful_lines(2203)
            if "full_attention_ck_q1_dynamic" not in line
        ]
        audit = MODULE.audit_routes(
            report(2203), MODULE.parse_marker_lines("\n".join(lines))
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["missing_requirements"][0]["marker"],
            "full_attention_ck_q1_dynamic",
        )

    def test_audit_rejects_cross_request_layer_count_smearing(self) -> None:
        first_request = successful_lines(2203)
        second_request = [
            line
            for line in successful_lines(2203)
            if MODULE.GDN_MARKER not in line
        ]
        audit = MODULE.audit_routes(
            report(2203, 2203),
            MODULE.parse_marker_lines("\n".join(first_request + second_request)),
        )
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["missing_requirements"])
        self.assertTrue(audit["request_sequence_failures"])
        self.assertEqual(
            audit["request_sequence_failures"][0]["request_index"], 1
        )
        self.assertEqual(
            audit["request_sequence_failures"][0]["marker"], MODULE.GDN_MARKER
        )

    def test_audit_rejects_one_missing_dynamic_layer(self) -> None:
        missing_layer = MODULE.EXPECTED_DYNAMIC_MOE_LAYERS[-1]
        lines = [
            line
            for line in successful_lines(2203)
            if not (
                MODULE.SMOOTH_TAIL_MARKER in line
                and f"layer={missing_layer} " in line
            )
        ]
        audit = MODULE.audit_routes(
            report(2203), MODULE.parse_marker_lines("\n".join(lines))
        )
        self.assertFalse(audit["passed"])
        layer_failure = next(
            failure
            for failure in audit["request_sequence_failures"]
            if failure.get("marker") == MODULE.SMOOTH_TAIL_MARKER
        )
        self.assertNotIn(missing_layer, layer_failure["actual_layers"])

    def test_audit_rejects_missing_request_boundary(self) -> None:
        lines = [
            line
            for line in successful_lines(2203)
            if MODULE.REQUEST_START_MARKER not in line
        ]
        audit = MODULE.audit_routes(
            report(2203), MODULE.parse_marker_lines("\n".join(lines))
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["request_segment_count"], 0)
        self.assertEqual(
            audit["request_sequence_failures"][0]["actual_count"], 0
        )

    def test_audit_rejects_non_dynamic_full_attention(self) -> None:
        lines = [
            line.replace(
                "exact_arbitrary_dynamic=1",
                "exact_arbitrary_dynamic=0",
            )
            if MODULE.FULL_ATTENTION_MARKER in line
            else line
            for line in successful_lines(2203)
        ]
        audit = MODULE.audit_routes(
            report(2203), MODULE.parse_marker_lines("\n".join(lines))
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            len(audit["semantic_failures"]),
            len(MODULE.EXPECTED_DYNAMIC_FULL_ATTENTION_LAYERS),
        )
        self.assertTrue(
            all(
                failure["marker"] == MODULE.FULL_ATTENTION_MARKER
                for failure in audit["semantic_failures"]
            )
        )

    def test_audit_rejects_dynamic_attention_length_spoof(self) -> None:
        lines = [
            line.replace("runtime_kv_length=2203", "runtime_kv_length=1")
            for line in successful_lines(2203)
        ]
        audit = MODULE.audit_routes(
            report(2203), MODULE.parse_marker_lines("\n".join(lines))
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(len(audit["semantic_failures"]), 1)
        self.assertEqual(
            audit["semantic_failures"][0]["marker"],
            MODULE.DYNAMIC_ATTENTION_MARKER,
        )

    def test_audit_rejects_fallback_and_local_slot_metadata(self) -> None:
        lines = successful_lines(2203)
        lines = [line.replace("global_topk_ids=1", "global_topk_ids=0") for line in lines]
        lines.append(
            "BATCH_MARK q1_terminal_device_corridor_output_fallback "
            "layer=39 prefill_tokens=2203 reason=handoff_miss"
        )
        audit = MODULE.audit_routes(
            report(2203), MODULE.parse_marker_lines("\n".join(lines))
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(len(audit["semantic_failures"]), 1)
        self.assertEqual(len(audit["fallbacks"]), 1)

    def test_audit_cannot_qualify_a_failed_plateau_report(self) -> None:
        audit = MODULE.audit_routes(
            report(2203, passed=False),
            MODULE.parse_marker_lines("\n".join(successful_lines(2203))),
        )
        self.assertFalse(audit["plateau_report_passed"])
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["missing_requirements"])

    def test_audit_rejects_padded_or_non_dynamic_moe(self) -> None:
        lines = [
            line.replace(
                "dynamic_logical_moe_provider=1",
                "dynamic_logical_moe_provider=0",
            ).replace(
                "provider_tail_padded=0",
                "provider_tail_padded=1",
            )
            for line in successful_lines(2203)
        ]
        audit = MODULE.audit_routes(
            report(2203), MODULE.parse_marker_lines("\n".join(lines))
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            len(audit["semantic_failures"]),
            len(MODULE.EXPECTED_DYNAMIC_MOE_LAYERS),
        )
        self.assertTrue(
            all(
                failure["marker"] == MODULE.SMOOTH_TAIL_MARKER
                for failure in audit["semantic_failures"]
            )
        )

    def test_audit_rejects_fixed_or_chunked_gdn(self) -> None:
        lines = []
        for line in successful_lines(2203):
            if MODULE.GDN_MARKER in line:
                line = line.replace(
                    "exact_arbitrary_dynamic=1",
                    "exact_arbitrary_dynamic=0",
                ).replace(
                    "independent_q8192_state_merge=0",
                    "independent_q8192_state_merge=1",
                )
            lines.append(line)
        audit = MODULE.audit_routes(
            report(2203), MODULE.parse_marker_lines("\n".join(lines))
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            len(audit["semantic_failures"]),
            len(MODULE.EXPECTED_DYNAMIC_GDN_LAYERS),
        )
        self.assertTrue(
            all(
                failure["marker"] == MODULE.GDN_MARKER
                for failure in audit["semantic_failures"]
            )
        )

    def test_audit_rejects_missing_exact_uncached_prefill_marker(self) -> None:
        lines = [
            line
            for line in successful_lines(2203)
            if "exact_first_token_prefill" not in line
        ]
        audit = MODULE.audit_routes(
            report(2203), MODULE.parse_marker_lines("\n".join(lines))
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["missing_requirements"][0]["marker"],
            MODULE.EXACT_PREFILL_MARKER,
        )

    def test_audit_rejects_prefix_cache_seed_or_hit(self) -> None:
        for marker in MODULE.PREFIX_CACHE_MARKERS:
            with self.subTest(marker=marker):
                lines = successful_lines(2203) + [
                    f"QRT_SERVER_MARK {marker} prefix_tokens=2202 "
                    "suffix_tokens=1 output_tokens=1"
                ]
                audit = MODULE.audit_routes(
                    report(2203), MODULE.parse_marker_lines("\n".join(lines))
                )
                self.assertFalse(audit["passed"])
                self.assertEqual(
                    audit["cold_cache_violations"][0]["marker"], marker
                )

    def test_audit_binds_exact_prefill_markers_to_execution_order(self) -> None:
        lines = successful_lines(2203) + successful_lines(8192)
        lines = [
            line.replace("output_token=144", "output_token=144")
            for line in lines
        ]
        audit = MODULE.audit_routes(
            execution_report(2203, 8192),
            MODULE.parse_marker_lines("\n".join(lines)),
        )
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["exact_prefill_sequence_failures"])

        wrong_output = [
            line.replace("output_token=144", "output_token=999")
            for line in lines
        ]
        rejected = MODULE.audit_routes(
            execution_report(2203, 8192),
            MODULE.parse_marker_lines("\n".join(wrong_output)),
        )
        self.assertFalse(rejected["passed"])
        self.assertEqual(
            rejected["exact_prefill_sequence_failures"][0]["mismatches"][
                "output_token"
            ]["expected"],
            "144",
        )

    def test_audit_counts_the_unmeasured_q8192_warmup_route(self) -> None:
        lines = successful_lines(8192) + successful_lines(2203)
        audit = MODULE.audit_routes(
            report_with_warmup(2203),
            MODULE.parse_marker_lines("\n".join(lines)),
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["expected_request_count"], 2)
        self.assertEqual(audit["expected_measurement_request_count"], 1)
        self.assertEqual(audit["expected_warmup_request_count"], 1)

        missing_warmup_route = MODULE.audit_routes(
            report_with_warmup(2203),
            MODULE.parse_marker_lines("\n".join(successful_lines(2203))),
        )
        self.assertFalse(missing_warmup_route["passed"])
        self.assertTrue(missing_warmup_route["missing_requirements"])


if __name__ == "__main__":
    unittest.main()
