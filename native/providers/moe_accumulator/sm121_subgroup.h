#ifndef QRT_SM121_SUBGROUP_H
#define QRT_SM121_SUBGROUP_H
#include "sm121_wave16.h"
#include "sm121_paired_products.h"
#include "sm121_dot_certificate.h"
#include <cstring>
#include <type_traits>
#ifndef QRT_SM121_DOT_STAGING_GROUPS
#define QRT_SM121_DOT_STAGING_GROUPS 1
#endif
static_assert(QRT_SM121_DOT_STAGING_GROUPS == 1 || QRT_SM121_DOT_STAGING_GROUPS == 4 ||
              QRT_SM121_DOT_STAGING_GROUPS == 8);

// Distribute the same sixteen products over fewer lanes. Products and their
// K16 carry keep the original exponent alignment, unsigned sum and normalizer.
// A smaller subgroup exposes more independent candidate dots per wave.
namespace qrt_sm121_subgroup {
// Lane-private products support strided operators without pretending they
// are a complete contiguous K16 input array. Each subgroup still owns all16
// products and one replicated carry.
template<unsigned Lanes>
__device__ __forceinline__ qrt_q1_moe_hawkeye::Value accumulate_products(
    qrt_q1_moe_hawkeye::Value carry, const uint32_t* products) {
    static_assert(Lanes == 4u || Lanes == 8u);
    constexpr unsigned items = 16u / Lanes;
    int maximum = carry.exponent > -133 ? carry.exponent : -133;
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
    return qrt_sm121_wave16::normalize(sum.magnitude, sum.negative, maximum);
}

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
        carry = accumulate_products<Lanes>(carry, products);
    }
    return carry;
}

template<unsigned Groups>
__device__ __forceinline__ bool try_staged_dot_tile(
    qrt_q1_moe_hawkeye::Value& carry,
    const qrt_q1_moe_hawkeye::Value (&products)[Groups], unsigned count) {
    if (!qrt_sm121_dot_certificate::canonical_normal(carry)) return false;
    int maximum = -133;
#pragma unroll
    for (unsigned group = 0u; group < Groups; ++group)
        if (group < count) maximum = products[group].exponent > maximum ? products[group].exponent : maximum;
    maximum = qrt_sm121_lane_reduce::maximum<16u>(maximum);
    if (maximum > carry.exponent) return false;
    uint32_t sums[Groups];
#pragma unroll
    for (unsigned group = 0u; group < Groups; ++group) {
        if (group < count) {
            const auto product = products[group];
            const unsigned shift = unsigned(carry.exponent - product.exponent);
            const uint32_t aligned = shift >= 32u ? 0u : (product.significand << 2u) >> shift;
            sums[group] = qrt_sm121_lane_reduce::sum<16u>(product.negative ? 0u - aligned : aligned);
        }
    }
    uint32_t partial = carry.negative ? 0u - carry.significand : carry.significand;
    bool valid = true;
#pragma unroll
    for (unsigned group = 0u; group < Groups; ++group)
        if (group < count) valid &= qrt_sm121_dot_certificate::advance(partial, sums[group], carry.negative);
    if (!valid) return false;
    carry.significand = carry.negative ? 0u - partial : partial;
    return true;
}

template<unsigned Lanes, unsigned StagingGroups = QRT_SM121_DOT_STAGING_GROUPS,
         bool Certified = QRT_SM121_CERTIFIED_DOT_TILES != 0>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
    unsigned reduction_size, qrt_sm121_dot_certificate::Stats* stats = nullptr) {
    static_assert(StagingGroups == 1u || StagingGroups == 4u || StagingGroups == 8u);
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    qrt_q1_moe_hawkeye::Value carry{0u, -133, false};
    if (stats) *stats = {};
    // Callers supply complete K16 groups, retaining ascending K order.
    if constexpr (Lanes == 16u && StagingGroups > 1u) {
        // Decode a K64/K128 operand tile before its carry-dependent work.
        // Independent loads and products can overlap; every K16 still uses
        // the original integer alignment, reduction and normalization.
#pragma unroll 1
        for (unsigned base = 0u; base < reduction_size; base += 16u * StagingGroups) {
            if (stats) ++stats->attempted;
            qrt_q1_moe_hawkeye::Value products[StagingGroups];
#pragma unroll
            for (unsigned group = 0u; group < StagingGroups; ++group) {
                const unsigned k = base + group * 16u;
                if (k < reduction_size)
                    products[group] = qrt_q1_moe_hawkeye::multiply_bf16(left[k + lane], right[k + lane], -133);
            }
            if constexpr (Certified) {
                const unsigned remaining = (reduction_size - base) / 16u;
                const unsigned count = remaining < StagingGroups ? remaining : StagingGroups;
                if (try_staged_dot_tile(carry, products, count)) {
                    if (stats) ++stats->accepted;
                    continue;
                }
            }
#pragma unroll
            for (unsigned group = 0u; group < StagingGroups; ++group)
                if (base + group * 16u < reduction_size)
                    carry = qrt_sm121_wave16::accumulate_product(carry, products[group]);
        }
    } else {
#pragma unroll 1
        for (unsigned base = 0u; base < reduction_size; base += 16u)
            carry = accumulate<Lanes>(carry, left + base, right + base);
    }
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}

} // namespace qrt_sm121_subgroup
#endif
