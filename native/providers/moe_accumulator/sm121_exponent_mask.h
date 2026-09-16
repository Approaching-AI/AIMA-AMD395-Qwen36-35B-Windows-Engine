#pragma once
#include "sm121_decoded_bf16.h"
#include "sm121_f32_carry.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_EXPONENT_MASK_INLINE __host__ __device__ __forceinline__
#else
#define QRT_EXPONENT_MASK_INLINE inline
#endif

namespace qrt_sm121_exponent_mask {
// Isolated representation. Every high half retains the original BF16. The
// low halves of slots 0..3 hold the row maximum and its first three position
// masks; remaining low halves retain decoded exponents. Existing prepared
// buffers must never be passed to this experiment's consumer.
struct Metadata { int16_t maximum; uint16_t mask[3]; };
static_assert(sizeof(Metadata) == 8u);

QRT_EXPONENT_MASK_INLINE void pack(const uint16_t* source, uint32_t* output) {
    int maximum = qrt_sm121_decoded_bf16::zero_exponent;
    for (unsigned i = 0u; i < 16u; ++i) {
        output[i] = qrt_sm121_decoded_bf16::pack(source[i]);
        const int e = qrt_sm121_decoded_bf16::exponent(source[i]);
        maximum = e > maximum ? e : maximum;
    }
    uint16_t masks[3]{};
    for (unsigned i = 0u; i < 16u; ++i) {
        const int rank = maximum - qrt_sm121_decoded_bf16::exponent(source[i]);
        if ((source[i] & 0x7fffu) && rank >= 0 && rank < 3)
            masks[rank] |= uint16_t(1u << i);
    }
    output[0] = (output[0] & 0xffff0000u) | uint16_t(maximum);
    for (unsigned rank = 0u; rank < 3u; ++rank)
        output[rank + 1u] = (output[rank + 1u] & 0xffff0000u) | masks[rank];
}

template<unsigned Stride>
QRT_EXPONENT_MASK_INLINE Metadata metadata(const uint32_t* input) {
    return {int16_t(input[0]), {uint16_t(input[Stride]),
        uint16_t(input[2u * Stride]), uint16_t(input[3u * Stride])}};
}

// Exact certificate for max(carry exponent, paired product exponents, -133).
// Both row maxima upper-bound every paired exponent. A matching rank-sum
// mask establishes that the bound (or bound-1/-2) is attained. A miss leaves
// output untouched and requires the original sixteen-pair maximum scan.
QRT_EXPONENT_MASK_INLINE bool maximum(const Metadata& left, const Metadata& right,
    int carry_exponent, int* output) {
    int upper = int(left.maximum) + int(right.maximum);
    if (upper < -133) upper = -133;
    if (carry_exponent >= upper) { *output = carry_exponent; return true; }
    int value;
    if (left.mask[0] & right.mask[0]) value = upper;
    else if ((left.mask[0] & right.mask[1]) | (left.mask[1] & right.mask[0])) value = upper - 1;
    else if ((left.mask[0] & right.mask[2]) | (left.mask[1] & right.mask[1]) |
             (left.mask[2] & right.mask[0])) value = upper - 2;
    else return false;
    *output = value > carry_exponent ? value : carry_exponent;
    return true;
}

// Precondition: both complete rows passed the original BF16 eligibility
// check. Product formation, integer reduction and normalization are unchanged.
template<unsigned LeftStride, unsigned RightStride, bool CarryAware>
QRT_EXPONENT_MASK_INLINE void group(qrt_sm121_float_alignment::Group& result,
    const uint32_t* left, const uint32_t* right, float carry) {
    result.first_negative = ((left[0] ^ right[0]) & 0x80000000u) != 0u;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 16u; ++i)
        result.products[i] = qrt_sm121_float_alignment::from_bits(left[i * LeftStride] & 0xffff0000u) *
            qrt_sm121_float_alignment::from_bits(right[i * RightStride] & 0xffff0000u);
    int carry_exponent = -133;
    if constexpr (CarryAware) {
        const uint32_t absolute = qrt_sm121_f32_carry::bits(carry) & 0x7fffffffu;
        if (absolute) carry_exponent = int(absolute >> 23u) - 127;
    }
    if (maximum(metadata<LeftStride>(left), metadata<RightStride>(right), carry_exponent, &result.maximum)) return;
    result.maximum = carry_exponent;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint32_t a = left[i * LeftStride], b = right[i * RightStride];
        const int ae = i < 4u ? qrt_sm121_decoded_bf16::exponent(uint16_t(a >> 16u)) : int(int16_t(a));
        const int be = i < 4u ? qrt_sm121_decoded_bf16::exponent(uint16_t(b >> 16u)) : int(int16_t(b));
        const int e = ae + be;
        result.maximum = e > result.maximum ? e : result.maximum;
    }
}
} // namespace qrt_sm121_exponent_mask
#undef QRT_EXPONENT_MASK_INLINE
