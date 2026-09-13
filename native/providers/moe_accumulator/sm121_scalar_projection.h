#ifndef QRT_SM121_SCALAR_PROJECTION_H
#define QRT_SM121_SCALAR_PROJECTION_H
#include "sm121_float_subgroup.h"

// Component candidate only. Preflight complete rows once, then compare
// independent scalar output ownership and cooperative replay without repeating
// BF16 range checks for every pair in every selected output dot.
namespace qrt_sm121_scalar_projection {
namespace alignment = qrt_sm121_float_alignment;
using Value = qrt_q1_moe_hawkeye::Value;

__global__ void eligible_rows_kernel(const uint16_t* source, unsigned* flags,
    unsigned rows, unsigned width) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned invalid;
    if (!threadIdx.x) invalid = 0u;
    __syncthreads();
    bool bad = false;
    for (unsigned k = threadIdx.x; k < width; k += blockDim.x)
        bad |= !alignment::eligible(source[size_t(row) * width + k]);
    if (bad) atomicOr(&invalid, 1u);
    __syncthreads();
    if (!threadIdx.x) flags[row] = invalid == 0u;
}

template<bool Transposed>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
    unsigned right_rows, unsigned row, unsigned count, bool eligible) {
    Value carry{0u, -133, false};
#pragma unroll 1
    for (unsigned base = 0u; base < count; base += 16u) {
        qrt_sm121_group16::AlignedSum sum;
        bool accepted = false;
        if (eligible) {
            alignment::Group group;
#pragma unroll
            for (unsigned i = 0u; i < 16u; ++i) {
                const size_t index = Transposed ? size_t(base + i) * right_rows + row
                                                : size_t(row) * count + base + i;
                group.set(i, left[base + i], right[index]);
            }
            accepted = alignment::sum(carry, group, &sum);
        }
        if (!accepted) {
            uint32_t products[16];
#pragma unroll
            for (unsigned i = 0u; i < 16u; ++i) {
                const size_t index = Transposed ? size_t(base + i) * right_rows + row
                                                : size_t(row) * count + base + i;
                products[i] = qrt_sm121_group16::pack_product(
                    qrt_q1_moe_hawkeye::multiply_bf16(left[base + i], right[index], -133));
            }
            sum = qrt_sm121_group16::sum_packed(carry, products);
        }
        carry = qrt_sm121_wave16::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
    }
    return qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}

__device__ __forceinline__ float cooperative_dot(const uint16_t* left,
    const uint16_t* right, unsigned count, bool eligible) {
    if (!eligible) return qrt_sm121_subgroup::dot<4u>(left, right, count);
    const unsigned lane = threadIdx.x & 3u;
    Value carry{0u, -133, false};
#pragma unroll 1
    for (unsigned base = 0u; base < count; base += 16u) {
        uint64_t a, b;
        __builtin_memcpy(&a, left + base + lane * 4u, 8u);
        __builtin_memcpy(&b, right + base + lane * 4u, 8u);
        qrt_sm121_float_subgroup::Product products[4];
#pragma unroll
        for (unsigned i = 0u; i < 4u; ++i) {
            const uint16_t x = uint16_t(a >> (i * 16u)), y = uint16_t(b >> (i * 16u));
            const bool zero = !(x & 0x7fffu) || !(y & 0x7fffu);
            products[i] = {
                alignment::from_bits(uint32_t(x) << 16u) * alignment::from_bits(uint32_t(y) << 16u),
                uint32_t(x) | (uint32_t(y) << 16u),
                zero ? -133 : int((x >> 7u) & 255u) + int((y >> 7u) & 255u) - 254};
        }
        carry = qrt_sm121_float_subgroup::accumulate<4u>(carry, products);
    }
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_scalar_projection
#endif
