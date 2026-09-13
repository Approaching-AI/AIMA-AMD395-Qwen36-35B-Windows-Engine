#include "../../native/providers/moe_accumulator/sm121_row_maximum.h"
#include <algorithm>
#include <cstdio>

namespace rm = qrt_sm121_row_maximum;
int main() {
    for (unsigned bits = 0u; bits < 65536u; ++bits) {
        const unsigned e = (bits >> 7u) & 255u;
        const bool zero = (bits & 0x7fffu) == 0u;
        const unsigned expected = zero ? 0u : e < 64u || e > 190u ? 256u : e;
        if (rm::word(uint16_t(bits)) != expected) return 1;
        for (unsigned current : {0u, 64u, 120u, 190u, 256u})
            if (rm::append(current, uint16_t(bits)) != std::max(current, expected)) return 2;
    }
    unsigned accepted = 0u, declined = 0u;
    // Independent original product exponents, including zero sentinels and
    // every supported exponent pair. A complete-row maximum may overestimate
    // any selected K16 slice; admission must remain conservative.
    const int carries[] = {-134,-133,-127,-126,-102,-101,-1,0,63,126,127,151,152,255};
    for (unsigned ae = 0u; ae <= 190u; ++ae) for (unsigned be = 0u; be <= 190u; ++be) {
        if ((ae && ae < 64u) || (be && be < 64u)) continue;
        for (unsigned aextra : {0u, 1u, 7u, 31u}) for (unsigned bextra : {0u, 1u, 7u, 31u}) {
            const unsigned am = std::min(190u, ae + aextra), bm = std::min(190u, be + bextra);
            for (int carry : carries) {
                const bool admit = rm::eligible(am,bm) && rm::carry_dominates(carry,rm::product_upper(am,bm));
                if (!admit) { ++declined; continue; }
                ++accepted;
                const int actual_product = ae && be ? int(ae + be) - 254 : -133;
                if (std::max({carry,actual_product,-133}) != carry) return 3;
            }
        }
    }
    if (rm::eligible(256u,127u) || rm::eligible(127u,256u)) return 4;
    std::printf("{\"kind\":\"row_maximum_cpu\",\"source_encodings\":65536,\"admitted_certificates\":%u,\"declined_certificates\":%u,\"incorrect_certificates\":0}\n",accepted,declined);
    return !accepted || !declined;
}
