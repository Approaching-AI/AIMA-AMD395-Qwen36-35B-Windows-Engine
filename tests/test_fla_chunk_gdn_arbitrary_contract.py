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
        self.assertIn("const int32_t scratch_tokens = padded_tokens(tokens)", self.source)
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

    def test_flashinfer_order_state_kernel_owns_the_core_output(self) -> None:
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
        self.assertIn("constexpr uint32_t kStateValueTiles = 2u", self.source)
        self.assertIn("kGateAndBetaBytesPerToken = 256u", self.source)
        self.assertIn("float *beta_pointer = beta_f32", self.source)
        self.assertIn("&a_pointer,\n        &beta_pointer,\n        &inverse_pointer", self.source)
        self.assertNotIn("KernelIndex::kRecomputeWU,", self.source)
        self.assertNotIn("KernelIndex::kChunkOutput,", self.source)
        self.assertIn("old_output = tl.dot", self.generator)
        self.assertIn("new_value_bf16", self.generator)
        self.assertIn("output_f32", self.generator)


if __name__ == "__main__":
    unittest.main()
