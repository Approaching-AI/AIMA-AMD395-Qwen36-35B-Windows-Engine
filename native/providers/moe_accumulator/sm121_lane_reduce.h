#ifndef QRT_SM121_LANE_REDUCE_H
#define QRT_SM121_LANE_REDUCE_H
#include <cstdint>
#include <hip/hip_runtime.h>
#ifndef QRT_SM121_DPP_REDUCTION
#define QRT_SM121_DPP_REDUCTION 0
#endif

namespace qrt_sm121_lane_reduce {
// Quad swaps combine pairs and quads; the two mirrors then combine adjacent
// quads and half rows. Every intermediate is replicated within its completed
// group. This changes only integer reduction transport, not K16 arithmetic.
// Encoding: https://llvm.org/doxygen/namespacellvm_1_1AMDGPU_1_1DPP.html
template<unsigned Lanes, bool Dpp = (QRT_SM121_DPP_REDUCTION != 0)>
__device__ __forceinline__ int maximum(int value) {
    static_assert(Lanes == 4 || Lanes == 8 || Lanes == 16);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    if constexpr (Dpp) {
        int other = __builtin_amdgcn_mov_dpp(value, 0xb1, 0xf, 0xf, true);
        value = other > value ? other : value;
        other = __builtin_amdgcn_mov_dpp(value, 0x4e, 0xf, 0xf, true);
        value = other > value ? other : value;
        if constexpr (Lanes >= 8) {
            other = __builtin_amdgcn_mov_dpp(value, 0x141, 0xf, 0xf, true);
            value = other > value ? other : value;
        }
        if constexpr (Lanes >= 16) {
            other = __builtin_amdgcn_mov_dpp(value, 0x140, 0xf, 0xf, true);
            value = other > value ? other : value;
        }
        return value;
    }
#endif
#pragma unroll
    for (unsigned mask = Lanes / 2; mask; mask >>= 1) {
        const int other = __shfl_xor(value, mask, Lanes);
        value = other > value ? other : value;
    }
    return value;
}

template<unsigned Lanes, bool Dpp = (QRT_SM121_DPP_REDUCTION != 0)>
__device__ __forceinline__ uint32_t sum(uint32_t value) {
    static_assert(Lanes == 4 || Lanes == 8 || Lanes == 16);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    if constexpr (Dpp) {
        value += uint32_t(__builtin_amdgcn_mov_dpp(value, 0xb1, 0xf, 0xf, true));
        value += uint32_t(__builtin_amdgcn_mov_dpp(value, 0x4e, 0xf, 0xf, true));
        if constexpr (Lanes >= 8)
            value += uint32_t(__builtin_amdgcn_mov_dpp(value, 0x141, 0xf, 0xf, true));
        if constexpr (Lanes >= 16)
            value += uint32_t(__builtin_amdgcn_mov_dpp(value, 0x140, 0xf, 0xf, true));
        return value;
    }
#endif
#pragma unroll
    for (unsigned mask = Lanes / 2; mask; mask >>= 1)
        value += __shfl_xor(value, mask, Lanes);
    return value;
}
}  // namespace qrt_sm121_lane_reduce
#endif
