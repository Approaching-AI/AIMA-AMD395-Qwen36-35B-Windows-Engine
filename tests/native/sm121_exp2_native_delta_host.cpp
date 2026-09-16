#include "../../native/providers/gdn/sm121_exp2_native_delta.h"
#include <cstdio>
#include <cstdlib>
#include <limits>

int main() {
    namespace delta = qrt_sm121_exp2_native_delta;
    uint32_t seed = 0x714932a5u;
    size_t checked = 0u;
    for (unsigned sample = 0u; sample < 1048576u; ++sample) {
        seed = seed * 1664525u + 1013904223u;
        const uint32_t input = sample < 8u ? (sample < 4u ? sample : UINT32_MAX - (sample - 4u)) : seed;
        unsigned char packed[4]{};
        for (unsigned i = 0u; i < 16u; ++i) {
            const int64_t expected = int64_t(input) + int(i) - 8;
            if (expected < 0 || expected > UINT32_MAX) continue;
            const unsigned code = delta::encode(input, uint32_t(expected));
            packed[i >> 2u] |= static_cast<unsigned char>(code << (2u * (i & 3u)));
            if (delta::code(packed, i) != code) std::abort();
            if (i >= 7u && i <= 9u) {
                if (code == 3u || delta::apply(input, code) != uint32_t(expected)) std::abort();
            } else if (code != 3u) std::abort();
            ++checked;
        }
    }
    if (delta::cells != 328728576u || delta::packed_bytes != 82182144u) std::abort();
    std::printf("native_exp2_delta_host_cells=%zu signed_edges_and_escape=pass\n", checked);
}
