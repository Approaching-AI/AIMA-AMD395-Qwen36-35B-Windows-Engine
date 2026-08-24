import hashlib
import json
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from public_hygiene import scan_bytes  # noqa: E402


class CurrentPublicCandidateContracts(unittest.TestCase):
    def test_selected_q8192_aot_inventory_is_clean_and_bound(self) -> None:
        aot = ROOT / "native" / "aot" / "gfx1151"
        expected = {
            "q8192_selected_moe_route_count.hsaco":
                "48cc7d3660fa6051f085996e827fe9d0485f52be306273d4d1a9bea8d7b39a39",
            "q8192_selected_moe_route_prefix_by_program.hsaco":
                "3aaec2826f234daad9a308ebc8957e0cc309d524ddc92984c39e5027eb80db4d",
            "q8192_selected_moe_route_padded_prefix.hsaco":
                "8c6fd362ea0b5e9b7d43ec99ae404ddef021ffed9844ad1203a82c307483b914",
            "q8192_selected_moe_route_scatter.hsaco":
                "dd87e1236cda5eb292b0c918ee98d04218e1ee174a2e9b86e98bf21644d15c7f",
            "q8192_selected_moe_gate_up_silu.hsaco":
                "945acb545a0cdbdb333ad7ed863a081b4e4a6a0b0872142482e307713529c0fe",
            "q8192_selected_moe_down.hsaco":
                "bd8b6970d1bcc86fe8eb8f9ee4a8d70fb24222badeddf9e477e9596c29d6c30a",
            "q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco":
                "cba71fdc8d510bc8f41d1e2501d3a64a8e7f2d3eb1078d4a940cc1c23a25be54",
            "q8192_triton_0626_zero_correction_gate_finalize.hsaco":
                "8105adcbd809bb01982c065bcd8f03b168313ff8f31fd78a4ffd04728846a4c6",
            "q8192_triton_0626_conditional_exact_down_rows4.hsaco":
                "2b430b0226d09af12b36018eed1a9f761141bee37793d5879c142cdd9cb8c1c4",
        }
        self.assertEqual(len(expected), 9)
        for name, expected_hash in expected.items():
            path = aot / name
            payload = path.read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_hash)
            self.assertEqual(scan_bytes(path.as_posix(), payload), [])

        metadata_path = aot / "q8192_triton_selected_moe_f32out.json"
        metadata_payload = metadata_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(metadata_payload).hexdigest(),
            "41e77d3afdecba63ee15c63e5f585f44ae29a6220e296e2752dcc18cfd24dc14",
        )
        metadata = json.loads(metadata_payload)
        self.assertEqual(metadata["shape"]["tokens"], 8192)
        self.assertEqual(metadata["shape"]["block_m"], 64)
        self.assertEqual(metadata["shape"]["group_m"], 1)
        self.assertTrue(metadata["postprocess"]["debug_sections_stripped"])
        for kernel in metadata["kernels"]:
            payload = (aot / kernel["file"]).read_bytes()
            self.assertEqual(len(payload), kernel["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), kernel["sha256"])

    def test_current_silu_aot_metadata_binds_sources_and_artifacts(self) -> None:
        aot = ROOT / "native" / "aot" / "gfx1151"
        records = (
            (
                "q8192_triton_0626_row_major_sorted_conditional_exact_gate_metadata.json",
                "kernel",
            ),
            (
                "q8192_triton_0626_zero_correction_gate_finalize_metadata.json",
                None,
            ),
        )
        for metadata_name, kernel_key in records:
            metadata = json.loads((aot / metadata_name).read_text(encoding="utf-8"))
            source = ROOT / metadata["source"]
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                metadata["source_sha256"],
            )
            if "imported_source" in metadata:
                imported_source = ROOT / metadata["imported_source"]
                self.assertEqual(
                    hashlib.sha256(imported_source.read_bytes()).hexdigest(),
                    metadata["imported_source_sha256"],
                )
            kernel = metadata[kernel_key] if kernel_key is not None else metadata
            payload = (aot / kernel["file"]).read_bytes()
            self.assertEqual(len(payload), kernel["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), kernel["sha256"])
            self.assertTrue(metadata["postprocess"]["debug_sections_stripped"])
            self.assertEqual(metadata["postprocess"]["private_home_path_count"], 0)
            self.assertTrue(metadata["qualification"]["component_only"])
            self.assertFalse(metadata["qualification"]["inference_success_claimed"])

    def test_v2_sorted_bf16_policy_is_implemented(self) -> None:
        provider = (
            ROOT / "native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp"
        ).read_text(encoding="utf-8")
        smoke = (
            ROOT / "native/providers/triton_moe/q8192_triton_selected_moe_smoke.cpp"
        ).read_text(encoding="utf-8")
        for fragment in (
            "combine_route_order_kernel(",
            "const int32_t *topk_ids,",
            "bool vllm_sorted_bf16_route_sum,",
            "topk_ids[route_base + route]",
            "route_order[step] = selected_route;",
            "const float contribution_bf16 = bf16_to_float(float_to_bf16(",
            "q8192_vllm_sorted_bf16_route_sum_enabled(),",
        ):
            self.assertIn(fragment, provider)
        for fragment in (
            "fill_router_debug_oracle_kernel(",
            "(expert - 10u) % 11u == 0u",
            '" router_debug_oracle=ranked_equal_top8_bf16"',
        ):
            self.assertIn(fragment, smoke)

    def test_windows_builder_strips_and_scans_selected_aot(self) -> None:
        build = (ROOT / "scripts/baiying_build_triton_moe_q8192.ps1").read_text(
            encoding="utf-8"
        )
        for fragment in (
            '[string]$WslLlvmStrip = "/opt/rocm/llvm/bin/llvm-strip"',
            "$WslLlvmStrip --strip-debug $wslAotPath",
            'debug_sections_stripped = $true',
            '$kernel.sha256 = (',
            '"AOT contains a private home path after debug stripping: $name"',
            "aot_debug_sections_stripped = $true",
            "aot_private_home_path_count = 0",
        ):
            self.assertIn(fragment, build)

    def test_complete_aot_evidence_is_component_scoped(self) -> None:
        evidence = json.loads(
            (
                ROOT
                / "benchmarks/performance/"
                "prefill-diagnostic-public-complete-aot-r1187-r1190.json"
            ).read_text(encoding="utf-8")
        )
        build = evidence["public_complete_aot_build"]
        smoke = evidence["full_smoke"]
        real = evidence["real_layer3"]
        transport = evidence["windows_transport_verification"]
        for record in (build, smoke, real, transport):
            path = ROOT / record["command_file"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["command_file_sha256"],
            )
        self.assertEqual(build["selected_kernel_count"], 9)
        self.assertEqual(build["private_home_path_count"], 0)
        self.assertTrue(
            build[
                "independent_regeneration_artifact_and_metadata_hashes_match"
            ]
        )
        self.assertEqual(smoke["dynamic_logical_case_count"], 32)
        self.assertEqual(smoke["dynamic_logical_q8192_mismatches"], 0)
        self.assertEqual(smoke["dynamic_logical_nonfinite"], 0)
        self.assertTrue(smoke["dynamic_logical_tail_guard_pass"])
        self.assertEqual(smoke["provider_backend_mask"], 15)
        self.assertEqual(real["output_bf16_fnv1a64"], "e1ade524d263ae24")
        self.assertEqual(real["nonfinite_count"], 0)
        self.assertLessEqual(real["max_abs_difference_to_gb10_bf16"], 0.125)
        self.assertEqual(real["abs_difference_over_0_125_count"], 0)
        self.assertEqual(transport["artifact_count"], 9)
        self.assertEqual(transport["private_home_path_count"], 0)
        self.assertFalse(transport["gpu_execution"])
        self.assertTrue(evidence["scope"]["component_only"])
        self.assertFalse(evidence["scope"]["windows_full_model_product_gate_attached"])
        self.assertFalse(evidence["scope"]["inference_success_claimed"])

    def test_random_length_evidence_rejects_integer_spikes(self) -> None:
        evidence = json.loads(
            (
                ROOT
                / "benchmarks/performance/"
                "prefill-diagnostic-current-provider-random-length-r1181-r1183.json"
            ).read_text(encoding="utf-8")
        )
        result = evidence["adjudicated_result"]
        self.assertTrue(result["all_78_pairs_exercised"])
        self.assertTrue(result["all_11_flagged_candidates_repeated_15_times"])
        self.assertFalse(
            result["reproducible_integer_length_performance_spike_detected"]
        )
        self.assertTrue(result["component_random_length_stability_pass"])

    def test_baiying_qualification_chain_keeps_product_gates(self) -> None:
        files_and_fragments = {
            "scripts/.candidate_build_dynamic_logical_moe_r1093.ps1": (
                "generated_from_current_source",
                "base_aot_dynamic_logical_abi",
                "dynamic_logical_q8192_exact_pass",
            ),
            "scripts/.candidate_build_dynamic_logical_moe_native_shared_r1168.ps1": (
                "-NativeSharedSelected",
                "qualification-provenance.json",
            ),
            "scripts/.candidate_run_product_q8192_fla_boundary_r1076.ps1": (
                "continuation_token_for_token_pass",
                "prefix_continuation_pass",
                "model_engine_load_pass",
                "acceptance_pass",
            ),
            "scripts/.candidate_start_random_length_terminal_r1091.ps1": (
                "q8192_continuation_token_for_token_pass",
                "q8192_prefix_continuation_pass",
                "model_engine_load_pass",
            ),
        }
        for relative, fragments in files_and_fragments.items():
            text = (ROOT / relative).read_text(encoding="utf-8")
            for fragment in fragments:
                self.assertIn(fragment, text)

    def test_candidate_script_references_are_present(self) -> None:
        references: set[str] = set()
        for pattern in (".candidate*.ps1", ".candidate*.sh"):
            for script in (ROOT / "scripts").glob(pattern):
                references.update(
                    match.replace("\\", "/")
                    for match in re.findall(
                        r"scripts[\\/][A-Za-z0-9_.-]+",
                        script.read_text(encoding="utf-8"),
                    )
                )
        missing = sorted(
            reference
            for reference in references
            if not (ROOT / reference).is_file()
        )
        self.assertEqual(missing, [])

    def test_q1024_owner_provider_uses_the_published_abi_header(self) -> None:
        provider = (
            ROOT
            / "native/providers/ck_fmha/"
            "qrt_qwen36_q16384_q1024_owner_provider.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '#include "../../src/qrt_qwen36_q1024_owner.h"', provider
        )
        self.assertNotIn(
            '#include "../../src/c/qrt_qwen36_q1024_owner.h"', provider
        )


if __name__ == "__main__":
    unittest.main()
