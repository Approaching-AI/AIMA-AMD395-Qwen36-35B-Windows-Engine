from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlaBlackwellKktContractTests(unittest.TestCase):
    def test_k128_is_one_continuous_blackwell_accumulator(self) -> None:
        source = (ROOT / "native/providers/gdn/blackwell_kkt.h").read_text() + (ROOT / "native/providers/gdn/blackwell_accumulator.h").read_text()
        self.assertIn('q1_moe_hawkeye_bf16_accumulator.h', source)
        self.assertIn("kGroup = 16", source)
        self.assertIn("base < 128; base += kGroup", source)
        self.assertIn("accumulator = accumulate(accumulator, left, right, lane)", source)
        self.assertIn("group_sum<26, kZeroExponent>", source)
        self.assertIn("column >= row", source)

    def test_native_dispatch_is_opt_in_bounded_and_completed_before_reuse(self) -> None:
        source = (ROOT / "native/providers/gdn/qrt_fla_chunk_gdn_q8192_provider.cpp").read_text()
        self.assertIn('std::strcmp(blackwell_kkt, "1") == 0', source)
        self.assertIn("tokens > kSegmentTokens || tokens % kChunk", source)
        start = source.index("bool launch_blackwell_kkt(")
        end = source.index("void release_scratch()", start)
        body = source[start:end]
        self.assertIn("hipEventSynchronize(end.handle)", body)
        self.assertIn("milliseconds <= 100.0f", body)
        self.assertIn("remaining chunks not submitted", body)
        self.assertLess(body.index('"a-dot-f32"'), body.index("qrt_fla_blackwell::gate_kernel"))

    def test_build_provenance_includes_both_native_headers(self) -> None:
        source = (ROOT / "scripts/baiying_build_fla_gdn.ps1").read_text()
        self.assertIn("$blackwellKkt = Join-Path", source)
        self.assertIn("$blackwellAccumulator = Join-Path", source)
        self.assertIn("$generator, $provider, $smoke, $blackwellKkt, $blackwellAccumulator", source)


if __name__ == "__main__":
    unittest.main()
