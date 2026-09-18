#pragma once
#include "sm121_f32_carry.h"
#include <cmath>
#if defined(__HIPCC__)
#define QRT_RZ_TREE_SPEC_INLINE __host__ __device__ __forceinline__
#else
#define QRT_RZ_TREE_SPEC_INLINE inline
#endif
namespace qrt_sm121_rz_tree {
QRT_RZ_TREE_SPEC_INLINE bool eligible(uint16_t value) {
    const unsigned exponent = (value >> 7u) & 255u;
    return !(value & 0x7fffu) || (exponent >= 96u && exponent <= 158u);
}

// Independent host specification for one IEEE FP32 round-toward-zero add.
// At exponent gaps <=29, double holds the exact sum of both FP32 terms.
// At larger gaps the small nonzero term is below one FP32 ulp: like signs
// retain the larger term, opposite signs move it one representable step to0.
inline float add_spec(float a, float b) {
    uint32_t x = qrt_sm121_f32_carry::bits(a), y = qrt_sm121_f32_carry::bits(b);
    const unsigned ax = x & 0x7fffffffu, ay = y & 0x7fffffffu;
    if (!ax && !ay) return qrt_sm121_float_alignment::from_bits((x & y) & 0x80000000u);
    if (!ax) return b;
    if (!ay) return a;
    if (ax < ay) { const auto word = x; x = y; y = word; const auto value = a; a = b; b = value; }
    const unsigned gap = ((x & 0x7fffffffu) >> 23u) - ((y & 0x7fffffffu) >> 23u);
    if (gap > 29u) return qrt_sm121_float_alignment::from_bits(x - (((x ^ y) >> 31u) ? 1u : 0u));
    const double exact = double(a) + double(b);
    if (exact == 0.0) return 0.0f;
    const float rounded = float(exact);
    const uint32_t bits = qrt_sm121_f32_carry::bits(rounded);
    return std::fabs(double(rounded)) > std::fabs(exact)
        ? qrt_sm121_float_alignment::from_bits(bits - 1u) : rounded;
}

inline float group_spec(float carry, const float* p) {
    const float a = add_spec(add_spec(p[0], p[1]), add_spec(p[2], p[3]));
    const float b = add_spec(add_spec(p[4], p[5]), add_spec(p[6], p[7]));
    const float c = add_spec(add_spec(p[8], p[9]), add_spec(p[10], p[11]));
    const float d = add_spec(add_spec(p[12], p[13]), add_spec(p[14], p[15]));
    return add_spec(carry, add_spec(add_spec(a, b), add_spec(c, d)));
}
} // namespace qrt_sm121_rz_tree
#undef QRT_RZ_TREE_SPEC_INLINE
