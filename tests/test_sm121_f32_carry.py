"""Check the single-register K16 carry against the independent wide sum."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class F32CarryTests(unittest.TestCase):
    def test_normalization_and_ordered_groups(self):
        code = r'''
#include "f32_carry_cases.h"
#include <cstdio>
namespace fast = qrt_sm121_f32_carry;
namespace cases = qrt_float_alignment_cases;
namespace candidate = qrt_f32_carry_cases;
int main() {
    unsigned normalizations = 0u, accepted = 0u, fallback = 0u;
    const uint32_t edges[] = {0u,1u,2u,3u,0x7fffffu,0x800000u,0xffffffu,
        0x1000000u,0x1000001u,0x7fffffffu,0x80000000u,
        qrt_sm121_group16::kMinNegativeModulo,qrt_sm121_group16::kMaxMagnitude};
    for (int maximum = -160; maximum <= 150; ++maximum) {
        for (unsigned i = 0u; i < 65536u; ++i) {
            const uint32_t magnitude = i < sizeof(edges)/sizeof(edges[0]) ? edges[i] :
                cases::random(i * 7919u + unsigned(maximum + 160)) % (qrt_sm121_group16::kMaxMagnitude + 1u);
            for (unsigned sign = 0u; sign < 2u; ++sign) {
                const auto value = qrt_sm121_canonical::normalize(magnitude, sign != 0u, maximum);
                const auto expected = cases::output_bits(value);
                float a = 123.0f, b = 123.0f;
                const bool pa = fast::normalize<0u>(magnitude, sign != 0u, maximum, &a);
                const bool pb = fast::normalize<1u>(magnitude, sign != 0u, maximum, &b);
                const bool admissible = !magnitude || (value.significand >= 0x800000u && value.exponent >= -126 && value.exponent <= 127);
                if (pa != pb || pa != admissible || (pa && (fast::bits(a) != expected || fast::bits(b) != expected))) {
                    std::printf("NORMALIZE_DIFF magnitude=%u maximum=%d sign=%u accepted=%u/%u expected=%08x actual=%08x/%08x\n",magnitude,maximum,sign,pa,pb,expected,fast::bits(a),fast::bits(b));
                    return 1;
                }
                if (!pa && (a != 123.0f || b != 123.0f)) return 2;
                ++normalizations;
            }
        }
    }
    for (unsigned row = 0u; row < 65536u; ++row) {
        uint32_t a[16],b[16],expected[16];
        cases::reference(row,expected);
        const auto ra=candidate::candidate<0u>(row,a), rb=candidate::candidate<1u>(row,b);
        for(unsigned g=0u;g<16u;++g)if(a[g]!=expected[g] || b[g]!=expected[g]) {
            std::printf("GROUP_DIFF row=%u group=%u expected=%08x actual=%08x/%08x\n",row,g,expected[g],a[g],b[g]);return 3;
        }
        if(ra.accepted!=rb.accepted || ra.fallback!=rb.fallback || ra.accepted+ra.fallback!=16u)return 4;
        accepted+=ra.accepted;fallback+=ra.fallback;
    }
    std::printf("{\"normalization_cases\":%u,\"variants\":2,\"ordered_groups_per_variant\":1048576,\"float_groups\":%u,\"fallback_groups\":%u,\"raw_mismatches\":0}\n",normalizations,accepted,fallback);
    return !accepted || !fallback;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-f32-carry-") as directory:
            exe = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                            "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                            "-I", str(ROOT / "tests/native"), "-x", "c++", "-", "-o", exe],
                           input=code, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
