from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FlaChunkGdnArbitraryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            ROOT
            / "native/providers/gdn/qrt_fla_chunk_gdn_q8192_provider.cpp"
        ).read_text()
        cls.generator = (
            ROOT
            / "native/generators/compile_q8192_fla_chunk_gdn.py"
        ).read_text()

    def test_arbitrary_lengths_only_stage_a_neutral_tail(self) -> None:
        self.assertIn("int32_t padded_tokens(int32_t tokens)", self.source)
        self.assertIn("const int32_t prefix_tokens", self.source)
        self.assertIn("const int32_t tail_tokens", self.source)
        self.assertIn("tokens > kSegmentTokens ? kSegmentTokens : padded_tokens(tokens)", self.source)
        self.assertIn("g_state.padded_postconv", self.source)
        self.assertIn("g_state.padded_gate", self.source)
        self.assertIn("hipMemsetAsync(padded_postconv)", self.source)
        self.assertIn("hipMemsetAsync(padded_gate)", self.source)
        self.assertIn("hipMemcpyAsync(unpadded_output)", self.source)
        self.assertIn("prefix_tokens == 0", self.source)
        self.assertNotIn("const int32_t launch_tokens = padded_tokens(tokens)", self.source)
        self.assertNotIn("supported multiple-of-64 shape", self.source)

    def test_larger_scratch_allocation_is_reused(self) -> None:
        self.assertIn("g_state.scratch_tokens >= tokens", self.source)
        for surface in (
            "kPaddedPostconvBytesPerToken",
            "kPaddedGateBytesPerToken",
            "kPaddedOutputBytesPerToken",
        ):
            self.assertIn(surface, self.source)
        self.assertIn("kTailPaddingBytes", self.source)
        self.assertIn("static_cast<int32_t>(kChunk)", self.source)

    def test_triton_fla_matches_the_gb10_backend_decomposition(self) -> None:
        for argument in (
            "&q_pointer",
            "&k_pointer",
            "&v_pointer",
            "&inverse_pointer",
            "&g_pointer",
            "&initial_state_pointer",
            "&output_pointer",
            "&final_state_pointer",
        ):
            self.assertIn(argument, self.source)
        self.assertIn("constexpr uint32_t kStateValueTiles = 8u", self.source)
        self.assertIn("kGateAndBetaBytesPerToken = 192u", self.source)
        self.assertIn("uint16_t *beta_pointer = beta_bf16", self.source)
        self.assertIn("&a_pointer,\n        &inverse_pointer", self.source)
        self.assertIn("KernelIndex::kRecomputeWU,", self.source)
        self.assertIn("KernelIndex::kChunkOutput,", self.source)
        self.assertIn("cumulative = tl.cumsum(values, axis=0)", self.generator)
        self.assertNotIn("tl.log2(alpha", self.generator)
        self.assertIn("b_k_beta.to(b_k.dtype)", self.generator)
        self.assertIn("current_v += tl.dot(w2", self.generator)
        self.assertIn("output_f32", self.generator)

    def test_wddm_recurrence_and_scratch_are_segment_bounded(self) -> None:
        self.assertIn("constexpr int32_t kSegmentTokens = 1024", self.source)
        self.assertIn("for (int32_t offset = 0; offset < prefix_tokens;)", self.source)
        self.assertIn("offset += count", self.source)
        self.assertIn("token_offset += segment_tokens", self.source)
        self.assertIn('include "qrt_fla_gdn_kernel_specs.inc"', self.source)


if __name__ == "__main__":
    unittest.main()
