from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeRouteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = (ROOT / "native/providers/whole_provider.cpp").read_text(
            encoding="utf-8"
        )
        cls.bridge = (ROOT / "native/src/qrt_server_bridge.c").read_text(
            encoding="utf-8"
        )
        cls.lifecycle = (ROOT / "engine/qrt-server/src/lifecycle.rs").read_text(
            encoding="utf-8"
        )
        cls.api = (ROOT / "engine/qrt-server/src/api.rs").read_text(encoding="utf-8")
        cls.ck_fmha = (
            ROOT / "native/providers/ck_fmha/qrt_ck_fmha_q8192_provider.cpp"
        ).read_text(encoding="utf-8")
        cls.ck_fmha_api = (
            ROOT / "native/providers/ck_fmha/fmha_fwd_api.cpp"
        ).read_text(encoding="utf-8")
        cls.ck_fmha_instance = (
            ROOT
            / "native/providers/ck_fmha/fmha_fwd_gfx1151_d256_bf16_f32out.cpp"
        ).read_text(encoding="utf-8")
        cls.ck_fmha_smoke = (
            ROOT / "native/providers/ck_fmha/q8192_ck_fmha_direct_smoke.cpp"
        ).read_text(encoding="utf-8")
        cls.ck_fmha_build = (
            ROOT / "scripts/baiying_build_ck_fmha_q8192.ps1"
        ).read_text(encoding="utf-8")
        cls.aiter_gdn = (
            ROOT / "native/providers/gdn/qrt_aiter_fused_gdn_q8192_provider.cpp"
        ).read_text(encoding="utf-8")
        cls.aiter_gdn_smoke = (
            ROOT / "native/providers/gdn/q8192_aiter_fused_gdn_smoke.cpp"
        ).read_text(encoding="utf-8")
        cls.aiter_gdn_build = (
            ROOT / "scripts/baiying_build_aiter_fused_gdn_q8192.ps1"
        ).read_text(encoding="utf-8")
        cls.q8192_build = (
            ROOT / "scripts/baiying_build_triton_moe_q8192.ps1"
        ).read_text(encoding="utf-8")
        cls.q8192_smoke = (
            ROOT / "native/providers/triton_moe/q8192_triton_selected_moe_smoke.cpp"
        ).read_text(encoding="utf-8")

    def test_resident_route_accepts_every_non_retained_length(self) -> None:
        start = self.provider.index("bool qwen36_exact_arbitrary_product_path_enabled(")
        end = self.provider.index(
            "bool descriptor_product_retained_or_exact_arbitrary_shape(", start
        )
        route = self.provider[start:end]
        for fragment in (
            "qwen36_resident_session_capture_is_active()",
            '"QRT_QWEN36_WHOLE_PROVIDER_ARBITRARY_CONTEXT"',
            "prefill_tokens < kRetainedPrefillTokens",
            "prefill_tokens <= QRT_QWEN36_MAX_POSITION_EMBEDDINGS",
            "prefill_tokens != kRetainedPrefillTokens",
        ):
            self.assertIn(fragment, route)
        self.assertNotIn("expected_output", route)

    def test_http_bridge_forces_complete_resident_route(self) -> None:
        start = self.bridge.index("qrt_status_t qrt_server_engine_create_v1(")
        end = self.bridge.index("void qrt_server_engine_free_v1(", start)
        create = self.bridge[start:end]
        for name in (
            "QRT_QWEN36_WHOLE_PROVIDER_ARBITRARY_CONTEXT",
            "QRT_QWEN36_WHOLE_PROVIDER_DIRECT_REQUEST_ENTRY",
            "QRT_QWEN36_WHOLE_PROVIDER_RESIDENT_SESSION",
        ):
            self.assertIn(name, create)

    def test_maximum_context_uses_parameterized_gpu_providers(self) -> None:
        for fragment in (
            "qrt_ck_fmha_q131073_chunk8192_bf16_launch",
            "qrt_ck_fmha_q262143_chunk8192_bf16_launch",
            "qrt_ck_fmha_q131073_tile8192_bf16_launch",
            "qrt_ck_fmha_q262143_tile8192_bf16_launch",
        ):
            self.assertIn(fragment, self.ck_fmha)
            self.assertIn(fragment, self.provider)

    def test_q8192_neighbors_keep_runtime_lengths_on_fast_providers(self) -> None:
        for fragment in (
            "qrt_ck_fmha_dynamic_bf16_launch",
            "tokens > kQ262144Tokens",
            "mask_enum::mask_top_left",
            "HasHeadPartitionFields",
            "set_head_partition_fields(args, kQueryHeads)",
        ):
            self.assertIn(fragment, self.ck_fmha)
        for fragment in (
            "qrt_ck_fmha_dynamic_bf16_launch",
            "ck_fmha_neighbor_smoke",
            "kNeighborLowTokens",
            "kNeighborHighTokens",
            "kNeighborMaxAbsTolerance",
            "neighbor_low_metrics.above_tolerance == 0u",
            "neighbor_high_metrics.above_tolerance == 0u",
            "dlopen(path, RTLD_NOW | RTLD_LOCAL)",
        ):
            self.assertIn(fragment, self.ck_fmha_smoke)
        for source in (self.ck_fmha_api, self.ck_fmha_instance):
            self.assertIn("QRT_CK_ARCH_TYPE", source)
        self.assertIn("defined(__gfx1151__)", self.ck_fmha_instance)
        for fragment in (
            '"struct\\s+gfx115_t"',
            '"struct\\s+gfx11_t"',
            '"-DQRT_CK_ARCH_TYPE=$ckArchType"',
            "ck_arch_type = $ckArchType",
        ):
            self.assertIn(fragment, self.ck_fmha_build)
        for fragment in (
            "qrt_aiter_fused_gdn_launch_async_dynamic",
            "AITER dynamic F32 fused-GDN requires 1..262144 tokens",
            'tokens,\n        "dynamic"',
            "kPathSeparator",
        ):
            self.assertIn(fragment, self.aiter_gdn)
        for fragment in (
            "kQ8191Tokens",
            "kQ8193Tokens",
            "qrt_aiter_fused_gdn_launch_async_dynamic",
            'use_dynamic_launch ? "dynamic" : "fixed"',
            "dlopen(path, RTLD_NOW | RTLD_LOCAL)",
            "target_device=AMD395",
        ):
            self.assertIn(fragment, self.aiter_gdn_smoke)
        for fragment in (
            "$smokeOutputQ8191",
            "$smokeOutputQ8193",
            "$asyncParityModeCount -ne 8",
        ):
            self.assertIn(fragment, self.aiter_gdn_build)
        for fragment in (
            "ck_fmha_exact_arbitrary_dynamic_shape",
            "QRT_QWEN36_EXACT_ARBITRARY_DYNAMIC_CK_LAYER_MASK",
            "QRT_QWEN36_EXACT_ARBITRARY_DYNAMIC_CK_LAYER_MASK_SEQUENCE",
            "exact_arbitrary_dynamic_ck_mask_sequence",
            "qwen36_resident_session_full_attention_layer_mask()",
            "use_ck_fmha_dynamic_full_attention_provider",
            "load_dynamic_aiter_fused_gdn_provider",
            "use_exact_arbitrary_dynamic_aiter_fused_gdn",
            "exact_arbitrary_q8192_moe",
            "prefill_tokens >= 4096u",
            "!maximum_context_streamed_prefill_tokens(prefill_tokens)",
            "provider_tile_tokens",
            "token_position_count >= 4096u",
            "token_position_count <= QRT_QWEN36_MAX_POSITION_EMBEDDINGS",
            "QRT_QWEN36_EXACT_ARBITRARY_LM_HEAD_TOPK_DIAGNOSTIC",
            "qwen36_exact_arbitrary_lm_head_topk",
        ):
            self.assertIn(fragment, self.provider)
        moe_start = self.provider.index(
            "bool run_qwen36_whole_provider_selected_moe_full_v2("
        )
        moe_end = self.provider.index(
            "bool run_qwen36_whole_provider_selected_moe_surface(", moe_start
        )
        moe_route = self.provider[moe_start:moe_end]
        self.assertIn("!exact_arbitrary_q8192_moe", moe_route)
        self.assertIn("static_cast<size_t>(kRetainedPrefillTokens)", moe_route)
        self.assertIn("provider_tail_padded=", moe_route)
        linear_start = self.provider.index(
            "bool run_repeated_prefill_resident_linear_stack_for_targets("
        )
        linear_end = self.provider.index(
            "bool run_early_resident_linear_attention_layers(",
            linear_start,
        )
        linear_route = self.provider[linear_start:linear_end]
        dynamic_gdn_start = linear_route.index(
            "const bool use_exact_arbitrary_dynamic_aiter_fused_gdn"
        )
        dynamic_gdn_end = linear_route.index(
            "const unsigned int use_exact_q131_context_aiter_fused_gdn",
            dynamic_gdn_start,
        )
        dynamic_gdn_route = linear_route[dynamic_gdn_start:dynamic_gdn_end]
        for fragment in (
            "!maximum_context_streamed_prefill_tokens(prefill_tokens)",
            "!use_exact_q16384_aiter_fused_gdn",
            "!use_exact_q32768_aiter_fused_gdn",
            "!use_exact_q65536_aiter_fused_gdn",
        ):
            self.assertIn(fragment, dynamic_gdn_route)

        attention_start = self.provider.index(
            "bool run_full_attention_prefill_resident_core_for_targets("
        )
        attention_end = self.provider.index(
            "bool run_full_attention_prefill_layer_helper_for_targets(",
            attention_start,
        )
        attention_route = self.provider[attention_start:attention_end]
        dynamic_ck_start = attention_route.index(
            "const bool ck_fmha_exact_arbitrary_dynamic_shape"
        )
        dynamic_ck_end = attention_route.index(
            "const bool ck_fmha_long_exact_shape", dynamic_ck_start
        )
        dynamic_ck_route = attention_route[dynamic_ck_start:dynamic_ck_end]
        for fragment in (
            "!maximum_context_streamed_prefill_tokens(prefill_tokens)",
            "!ck_fmha_q16384_exact_shape",
            "!ck_fmha_q32768_exact_shape",
            "!ck_fmha_q65536_exact_shape",
            "!ck_fmha_q131_context_exact_shape",
        ):
            self.assertIn(fragment, dynamic_ck_route)
        for fragment in (
            "QRT_AITER_GDN_DEFINE_LONG_EXPORTS(kQ131073Tokens, q131073)",
            "QRT_AITER_GDN_DEFINE_LONG_EXPORTS(kQ262143Tokens, q262143)",
            "qrt_aiter_fused_gdn_seeded_bf16_launch_async_dynamic",
        ):
            self.assertIn(fragment, self.aiter_gdn)
        for fragment in (
            "qrt_aiter_fused_gdn_q131073_launch_async",
            "qrt_aiter_fused_gdn_q262143_launch_async",
            "qrt_aiter_fused_gdn_seeded_bf16_launch_async_dynamic",
        ):
            self.assertIn(fragment, self.provider)

    def test_high_id_arbitration_is_bounded_and_arbitrary_only(self) -> None:
        start = self.provider.index(
            "__global__ void lm_head_bf16_window_high_id_rows_kernel("
        )
        end = self.provider.index(
            "constexpr unsigned int kLmHeadInverseF32SweepWindowCount", start
        )
        kernel = self.provider[start:end]
        for fragment in (
            "ulp < maximum_ulp_distance",
            "maximum_eligible_count_mask & (1u << eligible_count)",
            "topk_ids[row_base + candidate] > maximum_id",
            "topk_ids[row_base] = winning_id;",
        ):
            self.assertIn(fragment, kernel)
        active_start = self.provider.index(
            "const bool exact_arbitrary_lm_head_bf16_window_high_id_active ="
        )
        active_end = self.provider.index(
            "exact_arbitrary_lm_head_bf16_window_high_id_max_ulps", active_start
        )
        self.assertIn(
            "qwen36_exact_arbitrary_product_path_enabled(prefill_tokens)",
            self.provider[active_start:active_end],
        )
        self.assertIn(
            "resident_continuous_long_context_provider_requested(prefill_tokens)",
            self.provider[active_start:active_end],
        )
        self.assertIn(
            "exact_arbitrary_lm_head_bf16_window_high_id_global_requested",
            self.provider[active_start:active_end],
        )
        self.assertNotIn("input_tokens", self.provider[active_start:active_end])
        self.assertNotIn("expected_output", self.provider[active_start:active_end])

    def test_retained_q8192_uses_independent_numeric_window(self) -> None:
        self.assertIn(
            "kRetainedQ8192DefaultInverseF32MaxUlps = 35u", self.provider
        )
        start = self.provider.index(
            "const bool retained_q8192_lm_head_inverse_f32_default ="
        )
        end = self.provider.index(
            "const bool q16_distribution_full_logits_requested", start
        )
        route = self.provider[start:end]
        self.assertIn("!exact_arbitrary_lm_head_bf16_window_high_id_active", route)
        self.assertNotIn("expected_output", route)

    def test_q8192_event_ring_applies_backpressure_and_reuses_aot(self) -> None:
        self.assertIn(
            "QRT_TRITON_MOE_FULL_V3_EVENT_SLOTS + 1u", self.q8192_smoke
        )
        for fragment in (
            '[string]$ReuseAotDir = ""',
            "$aotReused = -not [string]::IsNullOrWhiteSpace",
            '"q8192_triton_selected_moe_f32out.json"',
            "full_provider_v3_async_chain_calls=$expectedAsyncChainCalls",
            'Join-Path $OutDir "build-provenance.json"',
        ):
            self.assertIn(fragment, self.q8192_build)

    def test_cli_and_health_expose_complete_contract(self) -> None:
        for fragment in (
            "pub arbitrary_moe_provider: Option<PathBuf>",
            'command.arg("--arbitrary-moe-provider").arg(provider)',
            'std::env::set_var("QRT_QWEN36_EXACT_ARBITRARY_Q1024_MOE_PROVIDER", "1")',
        ):
            self.assertIn(fragment, self.lifecycle)
        for fragment in (
            '"batch_size": 1',
            '"continuous_prompt_lengths": true',
            '"streaming": true',
            '"tool_calls": true',
            '"prefix_cache": true',
            '"bounded_fifo_queue": true',
        ):
            self.assertIn(fragment, self.api)


if __name__ == "__main__":
    unittest.main()
