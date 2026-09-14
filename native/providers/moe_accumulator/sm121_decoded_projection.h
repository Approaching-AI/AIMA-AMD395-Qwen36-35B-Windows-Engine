#pragma once
#include "sm121_scalar_projection.h"
#include "sm121_decoded_bf16.h"

// Component experiment: share lossless BF16 exponent decoding across selected
// output dots. Original operands remain available for every exceptional row.
namespace qrt_sm121_decoded_projection {
namespace decoded = qrt_sm121_decoded_bf16;
namespace floating = qrt_sm121_float_subgroup;
struct Stats { unsigned floating = 0u, integer = 0u; };

__global__ void prepare_rows(const uint16_t* source, uint32_t* packed,
    unsigned* flags, unsigned rows, unsigned width) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned invalid;
    if (!threadIdx.x) invalid = 0u;
    __syncthreads();
    bool bad = false;
    for (unsigned k = threadIdx.x; k < width; k += blockDim.x) {
        const uint16_t x = source[size_t(row) * width + k];
        packed[size_t(row) * width + k] = decoded::pack(x);
        bad |= !qrt_sm121_float_alignment::eligible(x);
    }
    if (bad) atomicOr(&invalid, 1u);
    __syncthreads();
    if (!threadIdx.x) flags[row] = invalid == 0u;
}

template<bool PackedWeights, bool Audit = false>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
    const uint32_t* packed_left, const uint32_t* packed_right,
    unsigned width, bool eligible, uint32_t* trace = nullptr, Stats* stats = nullptr) {
    if constexpr (!Audit) if (!eligible) return qrt_sm121_subgroup::dot<4u>(left, right, width);
    const unsigned lane = threadIdx.x & 3u;
    qrt_q1_moe_hawkeye::Value carry{0u, -133, false};
    Stats counts;
#pragma unroll 1
    for (unsigned base = 0u; base < width; base += 16u) {
        if (eligible) {
            uint32_t a[4], b[4];
            __builtin_memcpy(a, packed_left + base + lane * 4u, sizeof(a));
            if constexpr (PackedWeights)
                __builtin_memcpy(b, packed_right + base + lane * 4u, sizeof(b));
            else {
                uint64_t raw;
                __builtin_memcpy(&raw, right + base + lane * 4u, sizeof(raw));
#pragma unroll
                for (unsigned i = 0u; i < 4u; ++i) b[i] = decoded::pack(uint16_t(raw >> (i * 16u)));
            }
            floating::Product products[4];
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) {
                products[i] = {
                    floating::alignment::from_bits(a[i] & 0xffff0000u) *
                        floating::alignment::from_bits(b[i] & 0xffff0000u),
                    (a[i] >> 16u) | (b[i] & 0xffff0000u),
                    int(int16_t(a[i])) + int(int16_t(b[i]))};
            }
            bool accepted = false;
            carry = floating::accumulate<4u>(carry, products, Audit ? &accepted : nullptr);
            if constexpr (Audit) { counts.floating += accepted; counts.integer += !accepted; }
        } else {
            carry = qrt_sm121_subgroup::accumulate<4u>(carry, left + base, right + base);
            if constexpr (Audit) ++counts.integer;
        }
        if constexpr (Audit) if (!lane && trace) {
            const float value = qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
            __builtin_memcpy(trace + base / 16u, &value, sizeof(value));
        }
    }
    if constexpr (Audit) if (!lane && stats) *stats = counts;
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_decoded_projection
