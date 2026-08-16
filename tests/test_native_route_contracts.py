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
        cls.q8192_provider = (
            ROOT
            / "native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp"
        ).read_text(encoding="utf-8")
        cls.runtime_env = (ROOT / "engine/runtime.env").read_text(encoding="utf-8")

    def test_resident_route_can_include_retained_q8192(self) -> None:
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
            '"QRT_QWEN36_EXACT_ARBITRARY_RETAINED_Q8192"',
        ):
            self.assertIn(fragment, route)
        self.assertIn(
            "QRT_QWEN36_EXACT_ARBITRARY_RETAINED_Q8192=1",
            self.runtime_env,
        )
        self.assertNotIn("expected_output", route)

    def test_neighbor_product_profile_uses_accepted_arithmetic(self) -> None:
        self.assertIn(
            "QRT_PREFILL_DESCRIPTOR_BATCH_Q8192_AITER_FUSED_GDN_MIN_LAYER=0",
            self.runtime_env,
        )
        self.assertIn(
            "QRT_QWEN36_EXACT_ARBITRARY_CONV_ARITHMETIC_MODE=3",
            self.runtime_env,
        )

    def test_exact_arbitrary_q8192_disables_specialized_q1_route(self) -> None:
        helper_start = self.provider.index(
            "bool qwen36_specialized_retained_q8192_path_enabled("
        )
        helper_end = self.provider.index(
            "bool descriptor_product_retained_or_exact_arbitrary_shape(",
            helper_start,
        )
        helper = self.provider[helper_start:helper_end]
        self.assertIn("prefill_tokens == kRetainedPrefillTokens", helper)
        self.assertIn(
            "!qwen36_exact_arbitrary_product_path_enabled(prefill_tokens)",
            helper,
        )
        self.assertGreaterEqual(
            self.provider.count(
                "qwen36_specialized_retained_q8192_path_enabled("
            ),
            3,
        )
        self.assertNotIn("expected_output", helper)

    def test_arbitrary_prefill_uses_terminal_q1_target_plan(self) -> None:
        helper_start = self.provider.index(
            "bool qwen36_layer39_q1_kv8192_target_plan_shape_enabled("
        )
        helper_end = self.provider.index(
            "bool descriptor_product_retained_or_exact_arbitrary_shape(",
            helper_start,
        )
        helper = self.provider[helper_start:helper_end]
        self.assertIn("prefill_tokens > 0u", helper)
        self.assertIn(
            "prefill_tokens <= QRT_QWEN36_MAX_POSITION_EMBEDDINGS", helper
        )
        self.assertIn(
            "qwen36_specialized_retained_q8192_path_enabled(prefill_tokens)",
            helper,
        )
        self.assertIn(
            "qwen36_exact_arbitrary_product_path_enabled(prefill_tokens)",
            helper,
        )
        self.assertIn("qwen36_resident_session_capture_is_active()", helper)
        for feature in (
            "QRT_QWEN36_LAYER39_DYNAMIC_TERMINAL_COMPACT_Q",
            "QRT_QWEN36_LAYER39_DYNAMIC_TERMINAL_PACKED_MOE",
            "QRT_QWEN36_LAYER39_DYNAMIC_TERMINAL_DEVICE_CORRIDOR",
        ):
            self.assertIn(feature, helper)
        self.assertGreaterEqual(
            self.provider.count(
                "qwen36_layer39_q1_kv8192_target_plan_shape_enabled("
            ),
            4,
        )
        self.assertNotIn("expected_output", helper)

        alias_start = self.provider.index(
            "const bool layer39_q1_retained_compact_alias ="
        )
        alias_end = self.provider.index(
            "const bool layer39_q1_packed_routed_kernel_env_requested =",
            alias_start,
        )
        retained_only_alias = self.provider[alias_start:alias_end]
        self.assertIn(
            "qwen36_layer39_dynamic_terminal_packed_moe_enabled(prefill_tokens)",
            retained_only_alias,
        )
        self.assertIn(
            "run->selected_token_ids.front() == prefill_tokens - 1u",
            retained_only_alias,
        )
        self.assertNotIn("expected_output", retained_only_alias)

    def test_decode_corridor_clears_only_inherited_hip_launch_status(self) -> None:
        start = self.provider.index(
            "bool run_qwen36_resident_decode_linear_activation_corridor("
        )
        end = self.provider.index(
            "bool run_qwen36_resident_decode_q1_moe_activation_corridor(",
            start,
        )
        corridor = self.provider[start:end]
        clear = corridor.index(
            "const hipError_t inherited_launch_status = hipGetLastError();"
        )
        check = corridor.index("auto check_launch =", clear)
        first_kernel = corridor.index("hipLaunchKernelGGL(", check)
        self.assertLess(clear, check)
        self.assertLess(check, first_kernel)
        self.assertIn(
            "qwen36_resident_decode_inherited_hip_status_clear",
            corridor[clear:check],
        )
        self.assertIn("diagnostic_only=1", corridor[clear:check])

    def test_q8192_causal_verifier_is_reentrant_and_preserves_metadata(self) -> None:
        start = self.provider.index(
            "const bool causal_padded_neighbor_verify_requested ="
        )
        end = self.provider.index(
            "if (out_result->output_token_count > 0u)", start
        )
        route = self.provider[start:end]
        for fragment in (
            '"QRT_QWEN36_Q8192_CAUSAL_PADDED_Q8193_PREFILL_VERIFY"',
            '"QRT_QWEN36_Q8192_CAUSAL_PADDED_VERIFY_INPUT_TOKENS"',
            "causal_verifier_input_token_count",
            "verified_continuation.output_token_id != verified_token",
            "out_result->continuation = verified_continuation",
            "out_result->descriptor_result.continuation = verified_continuation",
            "resident_prefix_mutated=0",
            "causal_future_masked=1",
        ):
            self.assertIn(fragment, route)
        self.assertIn(
            "QRT_QWEN36_Q8192_CAUSAL_PADDED_Q8193_PREFILL_VERIFY=0",
            self.runtime_env,
        )
        self.assertIn(
            "QRT_QWEN36_Q8192_CAUSAL_PADDED_VERIFY_INPUT_TOKENS=8193",
            self.runtime_env,
        )
        self.assertIn(
            "std::recursive_mutex g_required_batch_marker_filter_mutex;",
            self.provider,
        )
        self.assertIn(
            "std::unique_lock<std::recursive_mutex>(",
            self.provider,
        )
        self.assertIn(
            "ScopedQwen36CausalPaddedPrefillVerifier causal_padding_scope(",
            self.provider,
        )
        self.assertIn(
            "!g_qwen36_causal_padded_prefill_verifier_active",
            self.provider,
        )
        verifier_start = self.provider.index(
            "bool qwen36_run_exact_low_margin_prefill_verifier(\n",
            self.provider.index(
                "bool qwen36_run_exact_low_margin_prefill_verifier(\n"
            ) + 1,
        )
        verifier_end = self.provider.index(
            "qrt_qwen36_whole_provider_exact_first_token_v1(",
            verifier_start,
        )
        verifier = self.provider[verifier_start:verifier_end]
        clear = verifier.index(
            "const hipError_t inherited_verifier_status = hipGetLastError();"
        )
        recursive_run = verifier.index(
            "verifier_ok = qrt_qwen36_whole_provider_prefill_v1(", clear
        )
        self.assertLess(clear, recursive_run)
        self.assertIn(
            "qwen36_exact_prefill_verifier_inherited_hip_status_clear",
            verifier[clear:recursive_run],
        )

    def test_q8193_uses_isolated_one_ulp_low_id_policy(self) -> None:
        kernel_start = self.provider.index(
            "__global__ void lm_head_bf16_one_ulp_low_id_rows_kernel("
        )
        kernel_end = self.provider.index(
            "__global__ void lm_head_bf16_window_high_id_rows_kernel(",
            kernel_start,
        )
        kernel = self.provider[kernel_start:kernel_end]
        for fragment in (
            "device_float_to_bf16(topk_logits[row_base])",
            "topk_logits[row_base + 2u] < five_ulp_floor",
            "topk_logits[row_base + slot] >= candidate_floor",
            "topk_ids[row_base + slot] < selected_id",
            "topk_ids[row_base] = winning_id;",
        ):
            self.assertIn(fragment, kernel)

        active_start = self.provider.index(
            "const bool q8193_bf16_one_ulp_low_id_active ="
        )
        active_end = self.provider.index(
            "exact_arbitrary_lm_head_bf16_window_high_id_max_ulps",
            active_start,
        )
        active = self.provider[active_start:active_end]
        for fragment in (
            "!g_qwen36_causal_padded_prefill_verifier_active",
            "prefill_tokens == kRetainedPrefillTokens + 1u",
            "run->selected_token_ids.back() == prefill_tokens - 1u",
            "qwen36_exact_arbitrary_product_path_enabled(prefill_tokens)",
        ):
            self.assertIn(fragment, active)
        self.assertNotIn("input_tokens", active)
        high_id_start = self.provider.index(
            "const bool exact_arbitrary_lm_head_bf16_window_high_id_active ="
        )
        high_id_end = self.provider.index(
            "const bool q8193_bf16_one_ulp_low_id_active =", high_id_start
        )
        high_id = self.provider[high_id_start:high_id_end]
        self.assertIn("q8193_bf16_one_ulp_low_id_shape", high_id)
        self.assertNotIn("expected_output", active)
        self.assertIn(
            "BATCH_MARK qwen36_q8193_bf16_one_ulp_low_id",
            self.provider,
        )
        launch_start = self.provider.index(
            "if (q8193_bf16_one_ulp_low_id_active) {",
            self.provider.index(
                "hipLaunchKernelGGL(\n                    lm_head_bf16_window_high_id_rows_kernel"
            ),
        )
        launch_end = self.provider.index(
            "if (exact_arbitrary_lm_head_grouped_arbitration_mode", launch_start
        )
        launch = self.provider[launch_start:launch_end]
        self.assertIn("(run->selected_token_count - 1u) * topk", launch)
        self.assertIn("                    1u\n                );", launch)
        self.assertIn(
            "QRT_QWEN36_Q8193_BF16_ONE_ULP_LOW_ID=1",
            self.runtime_env,
        )

        verifier_start = self.provider.index(
            "const bool causal_padded_neighbor_verify_requested ="
        )
        verifier_end = self.provider.index(
            "if (causal_padded_neighbor_verify_requested", verifier_start
        )
        verifier_gate = self.provider[verifier_start:verifier_end]
        self.assertIn(
            "request->input_token_count ==\n"
            "            static_cast<size_t>(kRetainedPrefillTokens)",
            verifier_gate,
        )
        self.assertNotIn("kRetainedPrefillTokens + 1u", verifier_gate)

    def test_q1_decode_arbitration_is_position_scoped_and_numeric(self) -> None:
        kernel_start = self.provider.index(
            "__global__ void lm_head_bf16_gap_4_3_separated_tail_runner_kernel("
        )
        kernel_end = self.provider.index(
            "__global__ void lm_head_bf16_exact_tie_high_id_kernel(",
            kernel_start,
        )
        kernel = self.provider[kernel_start:kernel_end]
        for fragment in (
            "top0_bits - top1_bits != 4u",
            "top1_bits - top2_bits != 3u",
            "top2_bits - top3_bits < 4u",
            "const uint32_t winning_id = topk_ids[1u];",
        ):
            self.assertIn(fragment, kernel)
        self.assertNotIn("expected_output", kernel)
        self.assertNotIn("input_tokens", kernel)

        route_start = self.provider.index(
            "const unsigned int q1_lm_head_bf16_one_ulp_low_id_position ="
        )
        route_end = self.provider.index(
            "const unsigned int q1_lm_head_bf16_inverse_f32_max_ulps =",
            route_start,
        )
        route = self.provider[route_start:route_end]
        for fragment in (
            '"QRT_QWEN36_Q1_LM_HEAD_BF16_ONE_ULP_LOW_ID_POSITION"',
            '"QRT_QWEN36_Q1_LM_HEAD_BF16_GAP_4_3_RUNNER_POSITION"',
            "absolute_position == static_cast<size_t>(",
        ):
            self.assertIn(fragment, route)
        self.assertNotIn("expected_output", route)
        self.assertNotIn("input_tokens", route)
        for fragment in (
            "BATCH_MARK qwen36_q1_lm_head_bf16_one_ulp_low_id",
            "BATCH_MARK qwen36_q1_lm_head_bf16_gap_4_3_runner",
        ):
            self.assertIn(fragment, self.provider)
        self.assertIn(
            "QRT_QWEN36_Q1_LM_HEAD_BF16_GAP_4_3_RUNNER_POSITION=8191",
            self.runtime_env,
        )
        self.assertIn(
            "QRT_QWEN36_Q1_LM_HEAD_BF16_ONE_ULP_LOW_ID_POSITION=8192",
            self.runtime_env,
        )

    def test_exact_arbitrary_can_diagnose_q8192_moe_tile_boundary(self) -> None:
        start = self.provider.index(
            "const bool exact_arbitrary_force_q1024_moe ="
        )
        end = self.provider.index(
            "const size_t provider_tile_count =", start
        )
        route = self.provider[start:end]
        self.assertIn(
            '"QRT_QWEN36_EXACT_ARBITRARY_FORCE_Q1024_MOE_PROVIDER"',
            route,
        )
        self.assertIn("!exact_arbitrary_force_q1024_moe", route)
        self.assertIn("exact_arbitrary_q1024_moe", route)
        self.assertNotIn("expected_output", route)

    def test_exact_arbitrary_moe_uses_smooth_tail_transactions(self) -> None:
        start = self.provider.index(
            "bool run_qwen36_whole_provider_selected_moe_full_v2("
        )
        end = self.provider.index(
            "bool run_qwen36_whole_provider_selected_moe_surface(", start
        )
        route = self.provider[start:end]
        for fragment in (
            "constexpr size_t kSmallProviderTileTokens = 1024u;",
            "const bool smooth_tail_route_requested =",
            "constexpr size_t kSmoothTailTokenQuantum = 32u;",
            "kSmoothTailTransactionTokenCounts",
            "provider_tail_tokens != 0u",
            "!maximum_context_streamed_prefill_tokens(prefill_tokens)",
            "provider_tail_tokens <= 16u",
            "transaction.direct_m16 = true;",
            "run_qwen36_smooth_tail_direct_m16_selected_moe(",
            "use_direct_m16_smooth_tail",
            "smooth_tail_rounded_tokens =",
            "remaining_capacity_tokens",
            "remaining_logical_tokens",
            "smooth_tail_transaction_count",
            "load_smooth_tail_triton_selected_moe_full_provider(",
            "use_smooth_tail_transaction",
            "smooth_tail_transaction.logical_offset",
            "tail_scratch_tile_tokens",
            "tile_logical_tokens != tile_capacity_tokens",
            "use_small_smooth_tail_provider",
            "QRT_QWEN36_SMOOTH_TAIL_SINGLE_CEIL_PROVIDER",
            "smooth_tail_single_ceil_provider_requested",
            "QRT_QWEN36_SMOOTH_TAIL_BOUNDED_TRANSACTIONS",
            "smooth_tail_bounded_transactions_requested",
            "QRT_QWEN36_SMOOTH_TAIL_PARALLEL_BASE",
            "smooth_tail_parallel_base_requested",
            "combined_capacity_tokens",
            "parallel_base_provider_alias",
            "smooth_tail_base_provider_alias",
            "use_smooth_tail_parallel_base",
            "ensure_qwen36_smooth_tail_parallel_lane(",
            "hipStreamWaitEvent(",
            "use_two_transactions",
            '<< " smooth_tail_moe="',
            '<< " smooth_tail_padding_tokens="',
            '<< " smooth_tail_transaction_count="',
            '<< " smooth_tail_direct_m16_transaction_count="',
            '<< " smooth_tail_single_ceil_provider="',
            '<< " smooth_tail_bounded_transactions="',
            '<< " smooth_tail_parallel_base="',
            '<< " smooth_tail_base_provider_alias="',
        ):
            self.assertIn(fragment, route)
        self.assertNotIn("mixed_q1024_tail", route)
        for fragment in (
            "smooth_tail_triton_selected_moe_provider_states()",
            '"QRT_QWEN36_SMOOTH_TAIL_MOE_PROVIDER"',
            '"QRT_QWEN36_SMOOTH_TAIL_Q32_MOE_DLL"',
            '"QRT_QWEN36_SMOOTH_TAIL_Q256_MOE_DLL"',
            '"QRT_QWEN36_SMOOTH_TAIL_Q4096_MOE_KERNEL_DIR"',
            "smooth_tail_triton_selected_moe_provider_last_error(",
        ):
            self.assertIn(fragment, self.provider)
        self.assertNotIn("expected_output", route)
        self.assertIn(
            "QRT_QWEN36_SMOOTH_TAIL_PARALLEL_BASE=1",
            self.runtime_env,
        )

        direct_start = self.provider.index(
            "bool run_qwen36_smooth_tail_direct_m16_selected_moe("
        )
        direct_end = self.provider.index(
            "bool run_qwen36_whole_provider_selected_moe_full_v2(",
            direct_start,
        )
        direct = self.provider[direct_start:direct_end]
        for fragment in (
            "constexpr unsigned int kMaximumTokens = 16u;",
            "q16_moe_raw_gate_up_activation_multirow_wave_kernel",
            "q16_moe_raw_down_combine_multirow_wave_kernel",
            "load_q1_moe_triton_0626_functions(",
            "load_q1_moe_triton_0626_shared_functions(",
            "moe_router_bf16_logits_topk_parallel_selection_kernel",
            "qwen36_smooth_tail_direct_q1_aot_selected_moe",
            "shared_expert_gate_scale_from_bf16_kernel",
            "shared_expert_activation_from_bf16_kernel",
            "shared_expert_down_combine_from_bf16_kernel",
            "output_residual_add_kernel",
            "qrt_descriptor_device_malloc(",
            "async=1 numerical_correctness_claimed=0",
        ):
            self.assertIn(fragment, direct)
        self.assertNotIn("TritonSelectedMoeFullLaunchFn", direct)
        self.assertNotIn("expected_output", direct)

    def test_dynamic_attention_and_gdn_cover_sub_q4096_lengths(self) -> None:
        gdn_start = self.provider.index(
            "bool descriptor_product_q8192_aiter_fused_gdn_provider_enabled("
        )
        gdn_end = self.provider.index(
            "struct TritonFullAttentionModuleState", gdn_start
        )
        gdn = self.provider[gdn_start:gdn_end]
        self.assertIn("prefill_tokens > 0u", gdn)
        self.assertNotIn("prefill_tokens >= 4096u", gdn)

        dynamic_gdn_start = self.provider.index(
            "const bool use_exact_arbitrary_dynamic_aiter_fused_gdn ="
        )
        dynamic_gdn_end = self.provider.index(
            "const unsigned int use_exact_q131_context_aiter_fused_gdn",
            dynamic_gdn_start,
        )
        dynamic_gdn = self.provider[dynamic_gdn_start:dynamic_gdn_end]
        self.assertIn("prefill_tokens > 0u", dynamic_gdn)
        self.assertNotIn("prefill_tokens >= 4096u", dynamic_gdn)

        dynamic_ck_start = self.provider.index(
            "const bool ck_fmha_exact_arbitrary_dynamic_shape ="
        )
        dynamic_ck_end = self.provider.index(
            "const bool ck_fmha_long_exact_shape", dynamic_ck_start
        )
        dynamic_ck = self.provider[dynamic_ck_start:dynamic_ck_end]
        self.assertIn("prefill_tokens > 0u", dynamic_ck)
        self.assertNotIn("prefill_tokens >= 4096u", dynamic_ck)

    def test_exact_arbitrary_can_trace_terminal_hidden_by_layer(self) -> None:
        start = self.provider.index(
            "bool emit_qwen36_exact_arbitrary_layer_output_trace("
        )
        end = self.provider.index(
            "bool run_qwen36_whole_provider_selected_moe_full_v2(", start
        )
        trace = self.provider[start:end]
        for fragment in (
            '"QRT_QWEN36_EXACT_ARBITRARY_LAYER_OUTPUT_TRACE"',
            '"QRT_QWEN36_EXACT_ARBITRARY_LAYER_OUTPUT_TRACE_POSITION"',
            "trace_position >= prefill_tokens",
            "QRT_QWEN36_HIDDEN_SIZE",
            "hipMemcpyDeviceToHost",
            "qwen36_exact_arbitrary_layer_output_trace",
            '" f32_bits="',
            "diagnostic_only=1 numerical_correctness_claimed=0",
        ):
            self.assertIn(fragment, trace)
        self.assertIn(
            "emit_qwen36_exact_arbitrary_layer_output_trace(",
            self.provider[end:],
        )
        for fragment in (
            "bool emit_qwen36_exact_arbitrary_final_norm_boundary_trace(",
            '"layer39_seed_hidden"',
            '"layer39_input_rmsnorm"',
            '"layer39_post_attention_hidden"',
            '"layer39_post_attention_rmsnorm"',
            '"layer39_output"',
            '"final_norm_output"',
            "qwen36_exact_arbitrary_final_norm_boundary_trace",
        ):
            self.assertIn(fragment, self.provider)
        self.assertIn(
            '"QRT_QWEN36_EXACT_ARBITRARY_LAYER_BOUNDARY_TRACE_LAYER"',
            self.provider[end:],
        )
        self.assertIn(
            '"QRT_QWEN36_EXACT_ARBITRARY_LINEAR_STAGE_TRACE_LAYER"',
            self.provider,
        )
        self.assertIn(
            "qwen36_exact_arbitrary_linear_stage_trace",
            self.provider,
        )
        self.assertNotIn("expected_output", trace)

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

    def test_q8192_router_restores_bf16_logit_endpoint_before_topk(self) -> None:
        start = self.q8192_provider.index("__global__ void router_topk_kernel(")
        end = self.q8192_provider.index("__global__ void shared_gate_scale_kernel(", start)
        kernel = self.q8192_provider[start:end]
        self.assertIn("bool bf16_logit_endpoint", kernel)
        self.assertGreaterEqual(kernel.count("float_to_bf16("), 2)
        self.assertIn("bf16_to_float(float_to_bf16(logit))", kernel)
        self.assertIn("bf16_to_float(float_to_bf16(accumulator))", kernel)
        self.assertIn(
            '"QRT_QWEN36_Q8192_ROUTER_BF16_LOGIT_ENDPOINT"',
            self.q8192_provider,
        )
        self.assertIn(
            "QRT_QWEN36_Q8192_ROUTER_BF16_LOGIT_ENDPOINT=1",
            self.runtime_env,
        )
        for fragment in (
            '"QRT_QWEN36_Q8192_ROUTER_HIPBLASLT_BF16"',
            "hipblasLtMatmul(router_logits_bf16)",
            "router_bf16_logits_topk_kernel",
            "ensure_matrix_plan(&g_state.router_plan, kExperts, kHidden)",
        ):
            self.assertIn(fragment, self.q8192_provider)
        self.assertIn(
            '"QRT_QWEN36_EXACT_ARBITRARY_MOE_STAGE_TRACE_LAYER"',
            self.provider,
        )
        for fragment in (
            '"QRT_QWEN36_Q8192_VLLM_SORTED_BF16_ROUTE_SUM"',
            "topk_ids[route_base + route]",
            "route_outputs[route * kHidden + column_base + lane]",
            "route_order[step]",
        ):
            self.assertIn(fragment, self.q8192_provider)
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
            "token_position_count > 0u",
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
        self.assertIn(
            '"QRT_QWEN36_EXACT_ARBITRARY_EARLY_BF16_MATRIX_OUTPUTS"',
            linear_route,
        )
        self.assertIn(
            "!exact_arbitrary_early_bf16_matrix_outputs",
            linear_route,
        )

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
            "topk_logits[row_base] >= maximum_activation_logit",
            "maximum_value_count >= 2u",
            "eligible_count >= 3u && one_ulp_value_count >= 2u",
            "topk_logits[row_base + 3u] < four_ulp_floor",
            "? (separated_tail",
            "? minimum_slot",
            ": maximum_slot",
            "maximum_bits - runner_bits == 3u",
            "maximum_bits - third_bits == 4u",
            "maximum_bits - fourth_bits >= 6u",
            "three_ulp_dense_runner_cluster",
            "? maximum_value_maximum_slot",
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
        self.assertIn(
            "exact_prefill_verifier_active",
            self.provider[active_start:active_end],
        )
        self.assertNotIn("input_tokens", self.provider[active_start:active_end])
        self.assertNotIn("expected_output", self.provider[active_start:active_end])
        self.assertIn(
            "QRT_QWEN36_EXACT_ARBITRARY_LM_HEAD_BF16_WINDOW_HIGH_ID_CONSERVATIVE_TOPOLOGY=1",
            self.runtime_env,
        )

    def test_exact_arbitrary_final_reference_topology_is_numeric_only(self) -> None:
        start = self.provider.index(
            "__global__ void lm_head_bf16_reference_topology_rows_kernel("
        )
        end = self.provider.index(
            "__global__ void lm_head_bf16_one_ulp_low_id_rows_kernel(", start
        )
        kernel = self.provider[start:end]
        for fragment in (
            "maximum_value_count == 2u",
            "next_distinct_value == two_ulp_floor",
            "? maximum_maximum_slot",
            ": maximum_minimum_slot",
            "two_ulp_eligible_count >= 4u",
            "one_ulp_value_count >= 2u",
            "selected_slot = two_ulp_minimum_slot",
            "maximum_bits - runner_bits == 4u",
            "runner_bits - third_bits == 3u",
            "third_bits - fourth_bits == 2u",
            "selected_slot = runner_minimum_slot",
            "topk_ids[row_base] = winning_id;",
        ):
            self.assertIn(fragment, kernel)

        active_start = self.provider.index(
            "const bool exact_arbitrary_lm_head_reference_topology_active ="
        )
        active_end = self.provider.index(
            "exact_arbitrary_lm_head_bf16_window_high_id_max_ulps", active_start
        )
        active = self.provider[active_start:active_end]
        self.assertIn(
            "qwen36_exact_arbitrary_product_path_enabled(prefill_tokens)",
            active,
        )
        self.assertIn("!q8193_bf16_one_ulp_low_id_shape", active)
        self.assertNotIn("input_tokens", active)
        self.assertNotIn("expected_output", active)

        launch = self.provider.index(
            "lm_head_bf16_reference_topology_rows_kernel,"
        )
        grouped_marker = self.provider.index(
            "qwen36_exact_arbitrary_lm_head_grouped_arbitration\"",
            launch,
        )
        self.assertLess(launch, grouped_marker)
        self.assertIn(
            "BATCH_MARK qwen36_exact_arbitrary_lm_head_reference_topology",
            self.provider,
        )
        self.assertIn(
            "two_way_tie_policy=two_ulp_high_id_other_low_id",
            self.provider,
        )

    def test_grouped_arbitration_is_shape_scoped_above_q8192(self) -> None:
        start = self.provider.index(
            "const unsigned int exact_arbitrary_lm_head_grouped_arbitration_mode ="
        )
        end = self.provider.index(
            "const bool exact_arbitrary_lm_head_bf16_window_high_id_active =",
            start,
        )
        route = self.provider[start:end]
        for fragment in (
            "prefill_tokens >= kRetainedPrefillTokens",
            "qwen36_exact_arbitrary_product_path_enabled(prefill_tokens)",
            "!q8193_bf16_one_ulp_low_id_shape",
            "!resident_continuous_long_context_provider_requested(prefill_tokens)",
        ):
            self.assertIn(fragment, route)
        self.assertNotIn("input_tokens", route)
        self.assertNotIn("expected_output", route)
        self.assertIn(
            "QRT_QWEN36_EXACT_ARBITRARY_LM_HEAD_GROUPED_ARBITRATION_MODE=2",
            self.runtime_env,
        )
        self.assertIn(
            "QRT_QWEN36_EXACT_ARBITRARY_LM_HEAD_GROUPED_ADAPTIVE_BF16_MINIMUM_LOGIT_EIGHTHS=128",
            self.runtime_env,
        )
        self.assertIn("adaptive_hopper_bf16_f32", self.provider)
        self.assertIn(
            "topk_logits[0u] == topk_logits[1u]",
            self.provider,
        )
        self.assertIn(
            "topk_logits[2u] >= adaptive_three_ulp_floor",
            self.provider,
        )
        self.assertIn(
            "topk_logits[3u] >= adaptive_five_ulp_floor",
            self.provider,
        )

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
