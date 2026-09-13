#ifndef QRT_SM121_KARATSUBA_CORE_H
#define QRT_SM121_KARATSUBA_CORE_H

#include <cstdint>
#include <cstring>

#if defined(__HIPCC__)
#define QRT_KARATSUBA_INLINE __host__ __device__ __forceinline__
#else
#define QRT_KARATSUBA_INLINE inline
#endif

namespace qrt_sm121_karatsuba {
struct Digits { int high, low; };

// Signed base256 with a balanced low digit keeps both digits and their sum
// in [-256,256]. Every such integer has an exact BF16 representation. This
// decomposes the existing signed16 core; original BF16 exception correction
// and ordered canonical carries still belong to the caller.
QRT_KARATSUBA_INLINE Digits split(uint16_t core) {
    const int value = core & 0x8000u ? int(core) - 65536 : int(core);
    const int byte = core & 255u;
    const int low = byte >= 128 ? byte - 256 : byte;
    return {(value - low) / 256, low};
}

QRT_KARATSUBA_INLINE uint16_t small_bf16(int integer) {
    const float value = float(integer);
#if defined(__HIP_DEVICE_COMPILE__)
    return uint16_t(__float_as_uint(value) >> 16u);
#else
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    return uint16_t(bits >> 16u);
#endif
}

QRT_KARATSUBA_INLINE int64_t reconstruct(int high, int low, int combined) {
    return int64_t(high) * 65536 + (int64_t(combined) - high - low) * 256 + low;
}

#if defined(__HIPCC__)
using Bf16x16 = unsigned short __attribute__((ext_vector_type(16)));
using F32x8 = float __attribute__((ext_vector_type(8)));
struct Parts { F32x8 high, low, combined; };

// Diagnostic until native comparisons and a correctness-attached product
// measurement qualify it. No snapping or approximate integer conversion is
// used: tests require the raw floating partials to equal independent integer
// sums. The absolute sum of16 products is at most2^20 for each matrix call.
template<class OperandRow>
__device__ __forceinline__ Parts products(const OperandRow& left, const OperandRow& right) {
    Bf16x16 ah{}, al{}, as{}, bh{}, bl{}, bs{};
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        const unsigned word = i / 4u, shift = (i % 4u) * 8u;
        const uint16_t a = uint16_t(((uint32_t(left.high[word]) >> shift) & 255u) * 256u |
                                   ((uint32_t(left.low[word]) >> shift) & 255u));
        const uint16_t b = uint16_t(((uint32_t(right.high[word]) >> shift) & 255u) * 256u |
                                   ((uint32_t(right.low[word]) >> shift) & 255u));
        const Digits x = split(a), y = split(b);
        ah[i] = small_bf16(x.high); al[i] = small_bf16(x.low); as[i] = small_bf16(x.high + x.low);
        bh[i] = small_bf16(y.high); bl[i] = small_bf16(y.low); bs[i] = small_bf16(y.high + y.low);
    }
    const F32x8 zero{};
    return {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bh, zero),
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al, bl, zero),
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as, bs, zero)};
}
#endif
}  // namespace qrt_sm121_karatsuba
#undef QRT_KARATSUBA_INLINE
#endif
