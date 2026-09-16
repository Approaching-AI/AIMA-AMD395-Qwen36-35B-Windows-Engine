#pragma once
#include "sm121_exponent_mask.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PACKED_MASK_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PACKED_MASK_INLINE inline
#endif

namespace qrt_sm121_packed_exponent_masks {
using Metadata = qrt_sm121_exponent_mask::Metadata;
namespace decoded = qrt_sm121_decoded_bf16;

// Preserve both original BF16 words in each input word. The caller owns one
// separate eight-byte metadata record per original K16 row and refreshes it
// after all row writers complete, before any dot consumer starts.
template<unsigned Stride>
QRT_PACKED_MASK_INLINE Metadata prepare(const uint32_t* words) {
    int maximum = -512;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 8u; ++i) {
        const uint32_t x = words[i * Stride];
        const int a = decoded::exponent(uint16_t(x)), b = decoded::exponent(uint16_t(x >> 16u));
        maximum = a > maximum ? a : maximum;
        maximum = b > maximum ? b : maximum;
    }
    Metadata result{int16_t(maximum), {0u, 0u, 0u}};
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint16_t x = uint16_t(words[(i / 2u) * Stride] >> ((i & 1u) * 16u));
        const int rank = maximum - decoded::exponent(x);
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (unsigned r = 0u; r < 3u; ++r)
            if ((x & 0x7fffu) && rank == int(r)) result.mask[r] |= uint16_t(1u << i);
    }
    return result;
}

// The original caller's eligibility check still owns the numerical domain.
// The carried maximum is a certificate input only; the original Value and
// its normalization remain unchanged, including extended carried exponents.
template<unsigned RightStride>
QRT_PACKED_MASK_INLINE void group(qrt_sm121_float_alignment::Group& result,
    const uint32_t* left, const uint32_t* right, const Metadata& lm,
    const Metadata& rm, int carry_exponent) {
    const int carried = carry_exponent > -133 ? carry_exponent : -133;
    const bool certified = qrt_sm121_exponent_mask::maximum(lm, rm, carried, &result.maximum);
    result.first_negative = ((left[0] ^ right[0]) & 0x8000u) != 0u;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 8u; ++i) {
        const uint32_t a = left[i], b = right[i * RightStride];
        result.products[2u * i] = qrt_sm121_float_alignment::from_bits(a << 16u) *
            qrt_sm121_float_alignment::from_bits(b << 16u);
        result.products[2u * i + 1u] = qrt_sm121_float_alignment::from_bits(a & 0xffff0000u) *
            qrt_sm121_float_alignment::from_bits(b & 0xffff0000u);
    }
    if (certified) return;
    result.maximum = carried;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint16_t a = uint16_t(left[i / 2u] >> ((i & 1u) * 16u));
        const uint16_t b = uint16_t(right[(i / 2u) * RightStride] >> ((i & 1u) * 16u));
        const int e = int(decoded::exponent(a)) + int(decoded::exponent(b));
        result.maximum = e > result.maximum ? e : result.maximum;
    }
}
} // namespace qrt_sm121_packed_exponent_masks
#undef QRT_PACKED_MASK_INLINE
