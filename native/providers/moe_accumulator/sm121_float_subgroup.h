#ifndef QRT_SM121_FLOAT_SUBGROUP_H
#define QRT_SM121_FLOAT_SUBGROUP_H
#include "sm121_subgroup.h"
#include "sm121_float_alignment.h"

// Candidate shared by dense and routed-MoE replay. Product dispatch remains
// unchanged until full numerical and real-model comparisons are attached.
namespace qrt_sm121_float_subgroup {
namespace alignment = qrt_sm121_float_alignment;
struct Product { float value; uint32_t original; int exponent; };

__device__ __forceinline__ Product prepare(uint16_t left, uint16_t right) {
    const uint32_t original = uint32_t(left) | (uint32_t(right) << 16u);
    if (!(left & 0x7fffu) || !(right & 0x7fffu)) return {0.0f, original, -133};
    if (!alignment::eligible(left) || !alignment::eligible(right)) return {0.0f, original, 512};
    return {alignment::from_bits(uint32_t(left) << 16u) * alignment::from_bits(uint32_t(right) << 16u),
        original, int((left >> 7u) & 255u) + int((right >> 7u) & 255u) - 254};
}

template<unsigned Lanes>
__device__ __forceinline__ qrt_q1_moe_hawkeye::Value accumulate(
    qrt_q1_moe_hawkeye::Value carry, const Product* products, bool* used_float = nullptr) {
    static_assert(Lanes == 4u || Lanes == 8u || Lanes == 16u);
    constexpr unsigned items = 16u / Lanes;
    int maximum = carry.exponent > -133 ? carry.exponent : -133;
#pragma unroll
    for (unsigned i = 0u; i < items; ++i)
        maximum = products[i].exponent > maximum ? products[i].exponent : maximum;
    maximum = qrt_sm121_lane_reduce::maximum<Lanes>(maximum);
    if (maximum == -133 && !carry.significand) {
        if (used_float) *used_float = true;
        return {0u, -133, false};
    }
    if (maximum >= -101 && maximum <= 151) {
        const float scale = alignment::from_bits(uint32_t(127 + 25 - maximum) << 23u);
        uint32_t modulo = 0u;
#pragma unroll
        for (unsigned i = 0u; i < items; ++i)
            modulo += uint32_t(int32_t(products[i].value * scale));
        modulo = qrt_sm121_lane_reduce::sum<Lanes>(modulo);
        // Keep the carried Value in its original integer representation.
        // This also handles carries beyond the finite FP32 exponent range.
        const unsigned shift = unsigned(maximum - carry.exponent);
        const uint32_t aligned = shift >= 32u ? 0u : (carry.significand << 2u) >> shift;
        modulo += carry.negative ? 0u - aligned : aligned;
        const uint32_t original = products[0].original;
        // In the overlapping modulo interval every product has the same
        // sign, so each lane's first product selects the identical result.
        const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo, ((original ^ (original >> 16u)) & 0x8000u) != 0u);
        if (used_float) *used_float = true;
        return qrt_sm121_wave16::normalize(sum.magnitude, sum.negative, maximum);
    }
    if (used_float) *used_float = false;
    if constexpr (Lanes == 16u) {
        const uint32_t original = products[0].original;
        return qrt_sm121_wave16::accumulate(carry, uint16_t(original), uint16_t(original >> 16u), threadIdx.x & 15u);
    } else {
        uint32_t original[items];
#pragma unroll
        for (unsigned i = 0u; i < items; ++i)
            original[i] = qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(
                uint16_t(products[i].original), uint16_t(products[i].original >> 16u), -133));
        return qrt_sm121_subgroup::accumulate_products<Lanes>(carry, original);
    }
}

template<unsigned Lanes, unsigned StagingGroups = 1u>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right, unsigned count,
    uint32_t* trace = nullptr, unsigned* float_groups = nullptr) {
    static_assert(Lanes == 4u || Lanes == 8u || Lanes == 16u);
    static_assert(StagingGroups == 1u || StagingGroups == 4u || StagingGroups == 8u);
    constexpr unsigned items = 16u / Lanes;
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    qrt_q1_moe_hawkeye::Value carry{0u, -133, false}; unsigned accepted = 0u;
#pragma unroll 1
    for (unsigned base = 0u; base < count; base += 16u * StagingGroups) {
        Product products[StagingGroups][items];
#pragma unroll
        for (unsigned group = 0u; group < StagingGroups; ++group) {
            const unsigned k = base + group * 16u;
            if (k < count) {
                using Packed = typename std::conditional<Lanes == 4u, uint64_t,
                    typename std::conditional<Lanes == 8u, uint32_t, uint16_t>::type>::type;
                Packed a, b;
                __builtin_memcpy(&a, left + k + lane * items, sizeof(a));
                __builtin_memcpy(&b, right + k + lane * items, sizeof(b));
#pragma unroll
                for (unsigned i = 0u; i < items; ++i)
                    products[group][i] = prepare(uint16_t(a >> (i * 16u)), uint16_t(b >> (i * 16u)));
            }
        }
#pragma unroll
        for (unsigned group = 0u; group < StagingGroups; ++group) if (base + group * 16u < count) {
            bool used_float; carry = accumulate<Lanes>(carry, products[group], float_groups ? &used_float : nullptr);
            if (float_groups) accepted += unsigned(used_float);
            if (trace && !lane) {
                const float value = qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
                __builtin_memcpy(trace + base / 16u + group, &value, 4u);
            }
        }
    }
    if (float_groups && !lane) *float_groups = accepted;
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
}  // namespace qrt_sm121_float_subgroup
#endif
