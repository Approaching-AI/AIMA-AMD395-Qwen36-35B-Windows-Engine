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

template<unsigned Lanes, unsigned Staging = 1u>
__device__ __forceinline__ float validated_dot(const uint16_t* left,
    const uint16_t* right, unsigned count, bool eligible) {
    static_assert(Lanes == 4u || Lanes == 8u || Lanes == 16u);
    static_assert(Staging == 1u || Staging == 4u || Staging == 8u);
    if (!eligible) return qrt_sm121_subgroup::dot<Lanes, Staging>(left, right, count);
    constexpr unsigned items = 16u / Lanes;
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    Value carry{0u, -133, false};
#pragma unroll 1
    for (unsigned base = 0u; base < count; base += 16u * Staging) {
        qrt_sm121_float_subgroup::Product products[Staging][items];
#pragma unroll
        for (unsigned group = 0u; group < Staging; ++group) if (base + group * 16u < count) {
            using Packed = typename std::conditional<Lanes == 4u, uint64_t,
                typename std::conditional<Lanes == 8u, uint32_t, uint16_t>::type>::type;
            Packed a, b;
            __builtin_memcpy(&a, left + base + group * 16u + lane * items, sizeof(a));
            __builtin_memcpy(&b, right + base + group * 16u + lane * items, sizeof(b));
#pragma unroll
            for (unsigned i = 0u; i < items; ++i) {
            const uint16_t x = uint16_t(a >> (i * 16u)), y = uint16_t(b >> (i * 16u));
            const bool zero = !(x & 0x7fffu) || !(y & 0x7fffu);
            products[group][i] = {
                alignment::from_bits(uint32_t(x) << 16u) * alignment::from_bits(uint32_t(y) << 16u),
                uint32_t(x) | (uint32_t(y) << 16u),
                zero ? -133 : int((x >> 7u) & 255u) + int((y >> 7u) & 255u) - 254};
            }
        }
#pragma unroll
        for (unsigned group = 0u; group < Staging; ++group) if (base + group * 16u < count)
            carry = qrt_sm121_float_subgroup::accumulate<Lanes>(carry, products[group]);
    }
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
__device__ __forceinline__ float cooperative_dot(const uint16_t* left,
    const uint16_t* right, unsigned count, bool eligible) {
    return validated_dot<4u>(left, right, count, eligible);
}
} // namespace qrt_sm121_scalar_projection
#endif
