#pragma once
#include "sm121_scaled_projection.h"

namespace qrt_sm121_scaled_fallback {
// Two independent predicates: the retained scalar-float range and the full
// normal-BF16 range supported by exact scaled-significand alignment.
__host__ __device__ __forceinline__ unsigned invalid_bits(uint16_t value) {
    return unsigned(!qrt_sm121_float_alignment::eligible(value)) |
        (unsigned(!qrt_sm121_scaled_significand::eligible(value)) << 1u);
}
__global__ void classify_rows(const uint16_t* source, unsigned* flags,
    unsigned rows, unsigned width) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned invalid;
    if (!threadIdx.x) invalid = 0u;
    __syncthreads();
    unsigned bad = 0u;
    for (unsigned k = threadIdx.x; k < width; k += blockDim.x)
        bad |= invalid_bits(source[size_t(row) * width + k]);
    if (bad) atomicOr(&invalid, bad);
    __syncthreads();
    if (!threadIdx.x) flags[row] = (~invalid) & 3u;
}
template<unsigned Lanes, unsigned Staging = 1u>
__device__ __forceinline__ float dot(const uint16_t* input, const uint16_t* weight,
    unsigned width, unsigned common_flags) {
    if constexpr (Lanes == 4u && Staging == 1u) {
        if (!(common_flags & 1u) && (common_flags & 2u))
            return qrt_sm121_scaled_projection::dot(input, weight, width, true);
    }
    return qrt_sm121_scalar_projection::validated_dot<Lanes, Staging>(
        input, weight, width, (common_flags & 1u) != 0u);
}
} // namespace qrt_sm121_scaled_fallback
