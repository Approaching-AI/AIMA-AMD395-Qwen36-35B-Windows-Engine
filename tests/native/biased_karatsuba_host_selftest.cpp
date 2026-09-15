#include "../../native/providers/moe_accumulator/sm121_biased_karatsuba_core.h"
#include <cassert>
#include <cmath>
#include <cstdio>
namespace fast = qrt_sm121_biased_karatsuba;
struct Row { int high[4]{}, low[4]{}; };
int signed_core(unsigned v) { return v & 32768u ? int(v) - 65536 : int(v); }
float half(uint16_t bits) {
    const unsigned e = (bits >> 10u) & 31u;
    return e ? ((bits & 0x8000u) ? -1.0f : 1.0f) * std::ldexp(float(1024u + (bits & 1023u)), int(e) - 25) : 0.0f;
}
int main() {
    for (unsigned value = 0u; value < 65536u; ++value) {
        Row row;
        for (unsigned word = 0u; word < 4u; ++word) {
            row.high[word] = int((value >> 8u) * 0x01010101u);
            row.low[word] = int((value & 255u) * 0x01010101u);
        }
        const auto packed = fast::prepare(row);
        const int low = int((value + 128u) & 255u) - 128;
        const int high = (signed_core(value) - low) / 256;
        assert(high >= -128 && high <= 128 && low >= -128 && low <= 127);
        assert(high + low >= -256 && high + low <= 256);
        for (unsigned i = 0u; i < 16u; ++i) {
            assert(fast::core(row, i) == value);
            assert(half(packed.high[i]) == float(high) && half(packed.low[i]) == float(low));
        }
    }
    for (int sum = -1048576; sum <= 1048576; ++sum) {
        const float biased = fast::bias + float(sum);
        assert(biased >= 8388608.0f && biased < 16777216.0f);
        assert(double(biased) == double(sum) + double(fast::bias));
        assert(biased - fast::bias == float(sum));
    }
    uint32_t state = 0x3958192u;
    auto random = [&]() { state ^= state << 13u; state ^= state >> 17u; state ^= state << 5u; return state; };
    for (unsigned dot = 0u; dot < 262144u; ++dot) {
        int64_t expected = 0; int high = 0, low = 0, combined = 0;
        for (unsigned k = 0u; k < 16u; ++k) {
            const unsigned a = random() & 65535u, b = random() & 65535u;
            const int al = int((a + 128u) & 255u) - 128, bl = int((b + 128u) & 255u) - 128;
            const int ah = (signed_core(a) - al) / 256, bh = (signed_core(b) - bl) / 256;
            high += ah * bh; low += al * bl; combined += (ah + al) * (bh + bl);
            expected += int64_t(signed_core(a)) * signed_core(b);
        }
        assert(fast::reconstruct(high, low, combined) == expected);
    }
    std::puts("{\"kind\":\"biased_karatsuba_host\",\"core_encodings\":65536,\"exact_bias_subtractions\":2097153,\"independent_wide_dots\":262144,\"raw_mismatches\":0,\"native_wmma_checked\":false}");
}
