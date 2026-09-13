#ifndef QRT_RANGE_PROJECTION_CASES_H
#define QRT_RANGE_PROJECTION_CASES_H
#include "float_alignment_cases.h"
namespace qrt_range_projection_cases {
inline qrt_float_alignment_cases::Pair input(unsigned row, unsigned group, unsigned i) {
    using namespace qrt_float_alignment_cases;
    const uint32_t a = random(row * 7919u + group * 997u + i * 17u + 0x395u);
    const uint32_t b = random(a ^ 0x8192u);
    switch (row % 8u) {
    case 0u: return qrt_float_alignment_cases::input(row / 8u, group, i);
    case 1u: return {uint16_t((a & 0x807fu) | ((77u + a % 103u) << 7u)),
                    uint16_t((b & 0x807fu) | ((77u + b % 103u) << 7u))};
    case 2u: return {uint16_t(a & 0x8000u), uint16_t(b & 0x8000u)};
    case 3u:
        // Form a small cancellation carry at the minimum exponent, then
        // carry it through zero-product K16 groups without clamping it.
        if (group) return {0u, 0u};
        if (i == 0u) return {uint16_t((77u << 7u) | 1u), uint16_t((77u << 7u) | 1u)};
        if (i == 1u) return {uint16_t(0x8000u | (77u << 7u)), uint16_t((77u << 7u) | 2u)};
        return {0u, 0u};
    case 4u: return {uint16_t((179u << 7u) | 127u), uint16_t((179u << 7u) | 127u)};
    case 5u: return {uint16_t((a & 0x807fu) | (76u << 7u)), uint16_t((b & 0x807fu) | (77u << 7u))};
    case 6u: return {uint16_t((a & 0x807fu) | (180u << 7u)), uint16_t((b & 0x807fu) | (179u << 7u))};
    default: return {uint16_t((a & 0x807fu) | (190u << 7u)), uint16_t((b & 0x807fu) | (64u << 7u))};
    }
}
} // namespace qrt_range_projection_cases
#endif
