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
        cls.aiter_gdn = (
            ROOT / "native/providers/gdn/qrt_aiter_fused_gdn_q8192_provider.cpp"
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
