#ifndef QRT_SM121_CANONICAL_NORMALIZE_H
#define QRT_SM121_CANONICAL_NORMALIZE_H
#include "q1_moe_hawkeye_bf16_accumulator.h"
#ifndef QRT_SM121_COMPACT_NORMALIZE
#define QRT_SM121_COMPACT_NORMALIZE 0
#endif
static_assert(QRT_SM121_COMPACT_NORMALIZE == 0 || QRT_SM121_COMPACT_NORMALIZE == 1);

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_CANONICAL_INLINE __host__ __device__ __forceinline__
#else
#define QRT_CANONICAL_INLINE inline
#endif

namespace qrt_sm121_canonical {
// Align the integer magnitude once, retaining the original internal26 bits
// before the final FP32 truncation. Folding the two truncating right shifts
// also preserves subnormal/zero results. OR1 makes the leading-zero operation
// defined at zero; the resulting zero significand restores exponent-133.
QRT_CANONICAL_INLINE qrt_q1_moe_hawkeye::Value normalize(
    uint32_t magnitude, bool negative, int maximum) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    const unsigned leading = __clz(magnitude | 1u);
#else
    const unsigned leading = 32u - qrt_q1_moe_hawkeye::bit_width_u64(magnitude | 1u);
#endif
    const int exponent = maximum + 6 - int(leading);
    const unsigned extra = exponent < -126 ? unsigned(-126 - exponent) : 0u;
    const uint32_t normalized = (magnitude << leading) >> 6u;
    const uint32_t significand = extra + 2u >= 32u ? 0u : normalized >> (extra + 2u);
    return {significand, int16_t(significand ? (exponent < -126 ? -126 : exponent) : -133), negative};
}
}
#undef QRT_CANONICAL_INLINE
#endif
