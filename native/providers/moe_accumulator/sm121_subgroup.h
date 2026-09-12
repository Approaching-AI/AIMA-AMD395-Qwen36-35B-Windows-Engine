#ifndef QRT_SM121_SUBGROUP_H
#define QRT_SM121_SUBGROUP_H
#include "sm121_wave16.h"
#include "sm121_paired_products.h"
#include <cstring>
#include <type_traits>

// Distribute the same sixteen products over fewer lanes. Products and their
// K16 carry keep the original exponent alignment, unsigned sum and normalizer.
// A smaller subgroup exposes more independent candidate dots per wave.
namespace qrt_sm121_subgroup {
// Advance one K16 group with a uniformly replicated FP32 carry. The caller
// may finish that carry between groups when its original operator does so.
template<unsigned Lanes>
__device__ __forceinline__ qrt_q1_moe_hawkeye::Value accumulate(
    qrt_q1_moe_hawkeye::Value carry, const uint16_t* left, const uint16_t* right) {
    static_assert(Lanes == 4u || Lanes == 8u || Lanes == 16u);
    constexpr unsigned items = 16u / Lanes;
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    if constexpr (Lanes == 16u) {
        carry = qrt_sm121_wave16::accumulate(carry, left[lane], right[lane], lane);
    } else {
        // memcpy permits a vector load without imposing stronger pointer
        // alignment or aliasing requirements than the original BF16 API.
        using Packed = typename std::conditional<Lanes == 4u, uint64_t, uint32_t>::type;
        Packed a, b;
        __builtin_memcpy(&a, left + lane * items, sizeof(Packed));
        __builtin_memcpy(&b, right + lane * items, sizeof(Packed));
        uint32_t products[items];
        int maximum = carry.exponent > -133 ? carry.exponent : -133;
        if constexpr (QRT_SM121_PAIRED_PRODUCTS) {
#pragma unroll
            for (unsigned i = 0u; i < items; i += 2u) {
                const auto pair = qrt_sm121_paired_products::multiply(uint32_t(a >> (i * 16u)), uint32_t(b >> (i * 16u)));
                products[i] = pair.low; products[i + 1u] = pair.high;
            }
        } else {
#pragma unroll
            for (unsigned i = 0u; i < items; ++i) {
                products[i] = qrt_sm121_group16::pack_product(
                    qrt_q1_moe_hawkeye::multiply_bf16(uint16_t(a >> (i * 16u)), uint16_t(b >> (i * 16u)), -133));
            }
        }
#pragma unroll
        for (unsigned i = 0u; i < items; ++i) {
            const int exponent = qrt_sm121_group16::packed_exponent(products[i]);
            maximum = exponent > maximum ? exponent : maximum;
        }
        maximum = qrt_sm121_lane_reduce::maximum<Lanes>(maximum);
        uint32_t modulo = 0u;
#pragma unroll
        for (unsigned i = 0u; i < items; ++i) {
            const unsigned shift = unsigned(maximum - qrt_sm121_group16::packed_exponent(products[i]));
            const uint32_t magnitude = shift >= 32u ? 0u : ((products[i] & 0xffffu) << 11u) >> shift;
            modulo += (products[i] & 0x80000000u) ? 0u - magnitude : magnitude;
        }
        modulo = qrt_sm121_lane_reduce::sum<Lanes>(modulo);
        const unsigned shift = unsigned(maximum - carry.exponent);
        const uint32_t aligned = shift >= 32u ? 0u : (carry.significand << 2u) >> shift;
        modulo += carry.negative ? 0u - aligned : aligned;
        const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo, (products[0] & 0x80000000u) != 0u);
        carry = qrt_sm121_wave16::normalize(sum.magnitude, sum.negative, maximum);
    }
    return carry;
}

template<unsigned Lanes>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
                                   unsigned reduction_size) {
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    qrt_q1_moe_hawkeye::Value carry{0u, -133, false};
    // Callers supply complete K16 groups, retaining ascending K order.
#pragma unroll 1
    for (unsigned base = 0u; base < reduction_size; base += 16u)
        carry = accumulate<Lanes>(carry, left + base, right + base);
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}

} // namespace qrt_sm121_subgroup
#endif
