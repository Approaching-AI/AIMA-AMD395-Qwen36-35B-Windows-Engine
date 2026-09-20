#pragma once
#include "sm121_mtp_math.h"
#include "sm121_bf16_fma.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MTP_KV_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MTP_KV_INLINE inline
#endif

namespace qrt_sm121_mtp {
QRT_MTP_KV_INLINE float key_inverse(const float* warp_sums, const unsigned char* table) {
    using namespace qrt_sm121_q1;
    return qrt_sm121_rsqrt::evaluate(table,
        add(multiply(head_norm_warp_sum(warp_sums, true), 1.0f / 256.0f), 1.0e-6f));
}

// The original text MRoPE rotates the first 64 channels of each 256-wide
// head. Its sine product rounds to BF16 before a single-round BF16 FMA.
QRT_MTP_KV_INLINE uint16_t key_rotated(const uint16_t* normalized_head,
                                     unsigned int channel, const uint16_t* coefficients) {
    if (channel >= 64u) return normalized_head[channel];
    using namespace qrt_sm121_q1;
    const unsigned int pair = channel & 31u;
    const uint16_t sine_product = bf16(multiply(
        widen(normalized_head[channel < 32u ? pair + 32u : pair]),
        widen(coefficients[32u + pair])));
    return qrt_sm121_bf16_fma::round(normalized_head[channel], coefficients[pair],
        uint16_t(sine_product ^ (channel < 32u ? 0x8000u : 0u)));
}
} // namespace qrt_sm121_mtp
#undef QRT_MTP_KV_INLINE
