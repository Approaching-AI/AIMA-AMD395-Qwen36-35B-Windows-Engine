"""Small products must survive the second rounding of packed gate dots."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackedGateMidpointTests(unittest.TestCase):
    def test_actual_header_against_exact_binary_midpoints(self):
        source = r'''
#include "native/providers/moe_accumulator/sm121_packed_dense.h"
#include <array>
#include <cassert>
#include <iostream>
int main() {
    namespace m = qrt_sm121_packed_dense;
    namespace q = qrt_sm121_q1;
    unsigned cases = 0, previous_errors = 0;
    for (unsigned negative = 0; negative < 2; ++negative)
    for (unsigned odd = 0; odd < 2; ++odd)
    for (int side = -1; side <= 1; ++side) {
        const float sign = negative ? -1.0f : 1.0f;
        const uint16_t lower = uint16_t((negative ? 0xbf80u : 0x3f80u) + odd);
        std::array<uint16_t,2048> x{},w{};
        x[0] = lower; w[0] = q::bf16(1.0f);
        x[16] = q::bf16(sign * 0x1p-8f); w[16] = q::bf16(1.0f);
        x[32] = q::bf16(sign * float(side) * 0x1p-26f); w[32] = q::bf16(1.0f);
        // These exact dyadic products give lower + half a BF16 step,
        // plus zero or a signed2^-26 perturbation. No float oracle is used.
        const uint16_t expected = uint16_t(lower + (side > 0 || (side == 0 && odd)));
        float sum = m::lane_dot<2048>(x.data(),w.data(),0);
        assert((qrt_sm121_exp2::bits(sum)&0xffffu)==0x8000u);
        previous_errors += q::bf16(sum) != expected;
        assert(m::gate_midpoint(sum,x.data(),w.data()) == expected); ++cases;
        std::array<float,2048> widened{};
        for (unsigned i=0;i<2048;++i) widened[i]=q::widen(x[i]);
        assert(m::gate_midpoint(sum,widened.data(),w.data()) == expected); ++cases;
        // A settled endpoint must not read or reinterpret any operand.
        const float settled = sign * 1.0f;
        assert(m::gate_midpoint(settled,static_cast<const uint16_t*>(nullptr),nullptr)==q::bf16(settled)); ++cases;
    }
    assert(cases==36 && previous_errors==4);
    // Double cancellation needs compensation: the exact sum is just above
    // 1 + 2^-8 despite an intervening positive/negative2^80 pair.
    std::array<uint16_t,2048> x{},w{};
    const float terms[]={1.0f,0x1p-8f,0x1p80f,0x1p-26f,-0x1p80f};
    for(unsigned i=0;i<5;++i){x[i]=q::bf16(terms[i]);w[i]=q::bf16(1.0f);}
    assert(m::gate_midpoint(1.0f+0x1p-8f,x.data(),w.data())==0x3f81u);
    std::cout << "37 exact midpoint/carrier/cancellation controls; four old-rounding failures\n";
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cpp, executable = directory / "midpoint.cpp", directory / "midpoint"
            cpp.write_text(source)
            subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off",
                "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer", "-I", str(ROOT), str(cpp), "-o", str(executable)],
                check=True, capture_output=True, text=True, timeout=45)
            result = subprocess.run([str(executable)], check=True, capture_output=True,
                                    text=True, timeout=20)
            self.assertIn("four old-rounding failures", result.stdout)


if __name__ == "__main__":
    unittest.main()
