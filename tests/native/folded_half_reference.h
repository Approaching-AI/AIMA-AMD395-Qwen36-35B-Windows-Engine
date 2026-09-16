#pragma once
#include "strong_float_replay_cases.h"
namespace qrt_folded_half_reference {
using Value = qrt_q1_moe_hawkeye::Value;
struct Expected { bool allowed, left; int maximum, power; };
// Independently derive applicability from original BF16 exponents and the
// original carry. No compressed metadata or production plan is consulted.
inline Expected expected(Value carry, const uint16_t* left, const uint16_t* right) {
    unsigned lo[2]{255u,255u}, hi[2]{};
    bool valid = true;
    int maximum = std::max(-133, int(carry.exponent));
    for (unsigned i = 0u; i < 16u; ++i) {
        const unsigned a = (left[i] >> 7u) & 255u, b = (right[i] >> 7u) & 255u;
        valid = valid && a && b && a < 255u && b < 255u;
        lo[0] = std::min(lo[0], a); hi[0] = std::max(hi[0], a);
        lo[1] = std::min(lo[1], b); hi[1] = std::max(hi[1], b);
        maximum = std::max(maximum, int(a) + int(b) - 254);
    }
    const int power = 25 - maximum + int(hi[0]) + int(hi[1]) - 284;
    const unsigned spread0 = hi[0] - lo[0], spread1 = hi[1] - lo[1];
    valid = valid && spread0 <= 29u && spread1 <= 29u;
    return {valid && power <= 0 && 30 - int(std::min(spread0,spread1)) + power >= 1,
        spread0 <= spread1, maximum, power};
}
}
