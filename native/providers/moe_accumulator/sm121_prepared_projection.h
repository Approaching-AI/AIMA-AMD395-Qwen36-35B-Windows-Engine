#ifndef QRT_SM121_PREPARED_PROJECTION_H
#define QRT_SM121_PREPARED_PROJECTION_H
#include "sm121_subgroup.h"
#include "sm121_prepared_bf16.h"

namespace qrt_sm121_prepared_projection {
// Each original row stays available for both the matrix producer and the
// exact fallback. A mixed row is never interpreted as a prepared operand.
__global__ void prepare_rows_kernel(const uint16_t* source, uint16_t* encoded,
    unsigned* eligible_rows, unsigned rows, unsigned width) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned invalid;
    if (threadIdx.x == 0u) invalid = 0u;
    __syncthreads();
    bool bad = false;
    for (unsigned k = threadIdx.x; k < width; k += blockDim.x) {
        const size_t index = size_t(row) * width + k;
        const uint16_t value = source[index];
        const bool valid = qrt_sm121_prepared_bf16::eligible(value);
        encoded[index] = valid ? qrt_sm121_prepared_bf16::encode(value) : 0u;
        bad |= !valid;
    }
    if (bad) atomicOr(&invalid, 1u);
    __syncthreads();
    if (threadIdx.x == 0u) eligible_rows[row] = invalid == 0u;
}

template<unsigned Lanes>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right, unsigned count) {
    static_assert(Lanes == 4u || Lanes == 8u || Lanes == 16u);
    constexpr unsigned items = 16u / Lanes;
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    qrt_q1_moe_hawkeye::Value carry{0u, -133, false};
#pragma unroll 1
    for (unsigned base = 0u; base < count; base += 16u) {
        using Packed = typename std::conditional<Lanes == 4u, uint64_t,
            typename std::conditional<Lanes == 8u, uint32_t, uint16_t>::type>::type;
        Packed a, b;
        __builtin_memcpy(&a, left + base + lane * items, sizeof(a));
        __builtin_memcpy(&b, right + base + lane * items, sizeof(b));
        uint32_t products[items];
#pragma unroll
        for (unsigned i = 0u; i < items; ++i)
            products[i] = qrt_sm121_prepared_bf16::multiply(uint16_t(a >> (i * 16u)), uint16_t(b >> (i * 16u)));
        if constexpr (Lanes == 16u) {
            const uint32_t p = products[0];
            carry = qrt_sm121_wave16::accumulate_product(carry,
                {(p & 0xffffu) << 9u, int16_t(qrt_sm121_group16::packed_exponent(p)), (p & 0x80000000u) != 0u});
        } else {
            carry = qrt_sm121_subgroup::accumulate_products<Lanes>(carry, products);
        }
    }
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
}
#endif
