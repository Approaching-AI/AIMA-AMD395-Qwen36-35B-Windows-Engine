#pragma once
#include "sm121_float_alignment.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PV_GROUP_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PV_GROUP_INLINE inline
#endif
namespace qrt_sm121_pv_group_plan {
constexpr uint32_t float_eligible = 0x80000000u;
QRT_PV_GROUP_INLINE uint32_t metadata(const uint16_t* values, unsigned count) {
    unsigned maximum = 0u; bool eligible = true;
    for (unsigned i = 0u; i < count; ++i) {
        const unsigned magnitude = values[i] & 0x7fffu;
        maximum = magnitude > maximum ? magnitude : maximum;
        eligible = eligible && qrt_sm121_float_alignment::eligible(values[i]);
    }
    return maximum | (eligible ? float_eligible : 0u);
}
// Product significands are at most (255*255)<<11 in the original aligned
// representation. At an exponent gap of27 every term truncates to zero.
// Zero operands also annihilate the original product, independently of the
// other operand. Callers still retain K32 alpha and K16 rounding boundaries.
QRT_PV_GROUP_INLINE unsigned skip_kind(int carry_exponent, uint32_t probability, uint32_t value) {
    const unsigned p = probability & 0x7fffu, v = value & 0x7fffu;
    if (!p || !v) return 1u;
    unsigned pe = p >> 7u, ve = v >> 7u;
    if (pe == 255u || ve == 255u) return 0u;
    pe = pe ? pe : 1u; ve = ve ? ve : 1u;
    const int upper = int(pe) + int(ve) - 254;
    return carry_exponent >= upper + 27 ? 2u : 0u;
}
QRT_PV_GROUP_INLINE bool use_float(uint32_t probability, uint32_t value) {
    return (probability & value & float_eligible) != 0u;
}
} // namespace qrt_sm121_pv_group_plan
#undef QRT_PV_GROUP_INLINE
