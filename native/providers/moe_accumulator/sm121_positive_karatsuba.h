#ifndef QRT_SM121_POSITIVE_KARATSUBA_H
#define QRT_SM121_POSITIVE_KARATSUBA_H
#include <cstdint>
#if defined(__HIPCC__)
#define QRT_POSITIVE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_POSITIVE_INLINE inline
#endif

namespace qrt_sm121_positive_karatsuba {
struct Digits { int high, low; };
QRT_POSITIVE_INLINE Digits split(uint16_t core) {
    return {int(core >> 8u) - ((core & 0x8000u) ? 256 : 0), int(core & 255u)};
}
QRT_POSITIVE_INLINE unsigned sum_digit(uint16_t core) {
    return ((unsigned(core) >> 8u) ^ 128u) + (core & 255u);
}
// The caller supplies an integer0..510, so this exact FP16 encoding needs
// neither floating conversion nor any rounding. Zero keeps its ordinary bits.
QRT_POSITIVE_INLINE uint16_t positive_half_bits(unsigned value) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned exponent = 31u - __clz(value | 1u);
#else
    unsigned exponent = 0u;
    for (unsigned copy = value; copy > 1u; copy >>= 1u) ++exponent;
#endif
    return value ? uint16_t(((exponent + 15u) << 10u) | ((value << (10u - exponent)) & 1023u)) : 0u;
}
// Let s=h+l+128, with signed high byte h and unsigned low byte l.
// SS-HH-LL-128*(sum(sA)+sum(sB))+16*128^2 is the exact cross term.
// The floating instruction sees only nonnegative integers0..510. HH and LL
// still use the original exact IU8 instructions. No floating cancellation or
// nearest-integer repair participates in the integer reconstruction.
QRT_POSITIVE_INLINE int64_t reconstruct(int high, int low, int combined,
                                       unsigned left_sum, unsigned right_sum) {
    const int64_t cross = int64_t(combined) - high - low -
        int64_t(left_sum + right_sum) * 128 + 262144;
    return int64_t(high) * 65536 + cross * 256 + low;
}

#if defined(__HIPCC__)
using F16x16 = _Float16 __attribute__((ext_vector_type(16)));
using F32x8 = float __attribute__((ext_vector_type(8)));
using I32x4 = int __attribute__((ext_vector_type(4)));
using I32x8 = int __attribute__((ext_vector_type(8)));
struct Parts { I32x8 high, low; F32x8 combined; };
template<class Row>
__device__ __forceinline__ Parts products(const Row& left, const Row& right) {
    I32x4 ah{}, al{}, bh{}, bl{}; F16x16 as{}, bs{};
#pragma unroll
    for (unsigned word = 0u; word < 4u; ++word) {
        ah[word] = left.high[word]; al[word] = left.low[word];
        bh[word] = right.high[word]; bl[word] = right.low[word];
#pragma unroll
        for (unsigned byte = 0u; byte < 4u; ++byte) {
            const unsigned shift = byte * 8u, i = word * 4u + byte;
            const unsigned a = ((uint32_t(ah[word]) >> shift) & 255u) ^ 128u;
            const unsigned b = ((uint32_t(bh[word]) >> shift) & 255u) ^ 128u;
            as[i] = _Float16(a + ((uint32_t(al[word]) >> shift) & 255u));
            bs[i] = _Float16(b + ((uint32_t(bl[word]) >> shift) & 255u));
        }
    }
    const I32x8 zero{}; const F32x8 fzero{};
    return {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ah, true, bh, zero, false),
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, al, false, bl, zero, false),
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(as, bs, fzero)};
}
#endif
}  // namespace qrt_sm121_positive_karatsuba
#undef QRT_POSITIVE_INLINE
#endif
