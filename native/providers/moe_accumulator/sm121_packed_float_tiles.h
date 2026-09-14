#pragma once
#include "sm121_strong_float_subgroup.h"
#include "sm121_dot_certificate.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PACKED_TILE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PACKED_TILE_INLINE inline
#endif

// Component experiment. Callers certify complete rows with the existing
// exponent84..174 predicate and K<=4096; other rows retain original replay.
namespace qrt_sm121_packed_tiles {
namespace strong = qrt_sm121_strong_float;
namespace certificate = qrt_sm121_dot_certificate;
using Value = qrt_q1_moe_hawkeye::Value;
struct Stats { unsigned attempted = 0u, accepted = 0u; };

// A product of two certified nonzero BF16 values is an exact, normal FP32
// value with at most16 significant bits. Its low8 mantissa bits are zero.
// The original (unnormalized) product exponent is in[-86,94], or-133 for
// zero, so exponent+133 fits those8 bits. Sign, value and exponent are all
// recoverable without retaining either source operand in the register tile.
QRT_PACKED_TILE_INLINE uint32_t prepare(uint16_t a, uint16_t b) {
    const auto p = strong::prepare(a, b);
    uint32_t bits; __builtin_memcpy(&bits, &p.value, 4u);
    return bits | unsigned(p.exponent + 133);
}
QRT_PACKED_TILE_INLINE strong::Product unpack(uint32_t word) {
    return {strong::alignment::from_bits(word & 0xffffff00u), int(word & 255u) - 133};
}
QRT_PACKED_TILE_INLINE Value fixed_value(uint32_t partial, Value original) {
    original.significand = original.negative ? 0u - partial : partial;
    return original;
}

// Host/device serial certificate for independent wide-reference tests.
// An unsuccessful certificate does not modify the caller's carry.
template<unsigned Groups>
QRT_PACKED_TILE_INLINE bool try_tile(Value carry, const uint32_t (&packed)[Groups][16],
    unsigned count, Value* output, Value* prefixes = nullptr) {
    if (!count || count > Groups || !certificate::canonical_normal(carry)) return false;
    int maximum = -133;
    for (unsigned g = 0u; g < count; ++g) for (unsigned i = 0u; i < 16u; ++i)
        if (unpack(packed[g][i]).exponent > maximum) maximum = unpack(packed[g][i]).exponent;
    if (maximum > carry.exponent) return false;
    const float scale = strong::alignment::from_bits(uint32_t(152 - carry.exponent) << 23u);
    uint32_t partial = carry.negative ? 0u - carry.significand : carry.significand;
    bool valid = true;
    for (unsigned g = 0u; g < count; ++g) {
        uint32_t sum = 0u;
        for (unsigned i = 0u; i < 16u; ++i) sum += uint32_t(int32_t(unpack(packed[g][i]).value * scale));
        valid &= certificate::advance(partial, sum, carry.negative);
        if (prefixes) prefixes[g] = fixed_value(partial, carry);
    }
    if (valid) *output = fixed_value(partial, carry);
    return valid;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
template<unsigned Lanes, unsigned Groups, bool Certify = true, bool Trace = false>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
    unsigned count, bool certified_rows, uint32_t* trace = nullptr, Stats* stats = nullptr) {
    static_assert(Lanes == 4u || Lanes == 8u || Lanes == 16u);
    static_assert(Groups == 1u || Groups == 4u || Groups == 8u);
    if (!certified_rows || count > 4096u) {
        if (stats) *stats = {};
        return qrt_sm121_float_subgroup::dot<Lanes, 1u>(left, right, count, Trace ? trace : nullptr);
    }
    constexpr unsigned items = 16u / Lanes;
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    Value carry{0u, -133, false}; Stats totals;
#pragma unroll 1
    for (unsigned base = 0u; base < count; base += Groups * 16u) {
        uint32_t packed[Groups][items];
        int product_maximum = -133;
#pragma unroll
        for (unsigned g = 0u; g < Groups; ++g) if (base + g * 16u < count) {
            using Packed = typename std::conditional<Lanes == 4u, uint64_t,
                typename std::conditional<Lanes == 8u, uint32_t, uint16_t>::type>::type;
            Packed a, b;
            __builtin_memcpy(&a, left + base + g * 16u + lane * items, sizeof(a));
            __builtin_memcpy(&b, right + base + g * 16u + lane * items, sizeof(b));
#pragma unroll
            for (unsigned i = 0u; i < items; ++i) {
                packed[g][i] = prepare(uint16_t(a >> (i * 16u)), uint16_t(b >> (i * 16u)));
                if constexpr (Certify) product_maximum = max(product_maximum, unpack(packed[g][i]).exponent);
            }
        }
        bool accepted = false;
        if constexpr (Certify) {
            if (stats) ++totals.attempted;
            product_maximum = qrt_sm121_lane_reduce::maximum<Lanes>(product_maximum);
            if (certificate::canonical_normal(carry) && product_maximum <= carry.exponent) {
                const float scale = strong::alignment::from_bits(uint32_t(152 - carry.exponent) << 23u);
                uint32_t partial = carry.negative ? 0u - carry.significand : carry.significand;
                uint32_t prefixes[Trace ? Groups : 1u];
                accepted = true;
#pragma unroll
                for (unsigned g = 0u; g < Groups; ++g) if (base + g * 16u < count) {
                    uint32_t sum = 0u;
#pragma unroll
                    for (unsigned i = 0u; i < items; ++i)
                        sum += uint32_t(int32_t(unpack(packed[g][i]).value * scale));
                    sum = qrt_sm121_lane_reduce::sum<Lanes>(sum);
                    accepted &= certificate::advance(partial, sum, carry.negative);
                    if constexpr (Trace) prefixes[g] = partial;
                }
                if (accepted) {
                    if constexpr (Trace) if (!lane) {
#pragma unroll
                        for (unsigned g = 0u; g < Groups; ++g) if (base + g * 16u < count) {
                            const float value = qrt_q1_moe_hawkeye::value_to_float(fixed_value(prefixes[g], carry));
                            __builtin_memcpy(trace + base / 16u + g, &value, 4u);
                        }
                    }
                    carry = fixed_value(partial, carry);
                    if (stats) ++totals.accepted;
                }
            }
        }
        if (!accepted) {
#pragma unroll
            for (unsigned g = 0u; g < Groups; ++g) if (base + g * 16u < count) {
                strong::Product products[items];
#pragma unroll
                for (unsigned i = 0u; i < items; ++i) products[i] = unpack(packed[g][i]);
                carry = strong::accumulate<Lanes>(carry, products);
                if constexpr (Trace) if (!lane) {
                    const float value = qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
                    __builtin_memcpy(trace + base / 16u + g, &value, 4u);
                }
            }
        }
    }
    if (stats) *stats = totals;
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
#endif
} // namespace qrt_sm121_packed_tiles
#undef QRT_PACKED_TILE_INLINE
