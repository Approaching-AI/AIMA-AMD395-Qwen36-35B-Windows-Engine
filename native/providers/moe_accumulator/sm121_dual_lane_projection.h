#pragma once
#include "sm121_scalar_projection.h"

// Component-only two-lane ownership. Each lane contributes eight original
// products; one adjacent-pair exchange replaces the four-lane quad reduction.
// The unsigned sum, carried Value and canonical normalization are unchanged.
namespace qrt_sm121_dual_lane {
namespace original = qrt_q1_moe_hawkeye;
namespace alignment = qrt_sm121_float_alignment;
using Value = original::Value;
struct Stats { unsigned floating = 0u, integer = 0u; };

__device__ __forceinline__ uint32_t other(uint32_t value) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) && QRT_SM121_DPP_REDUCTION
    return uint32_t(__builtin_amdgcn_mov_dpp(value, 0xb1, 0xf, 0xf, true));
#else
    return __shfl_xor(value, 1u, 2u);
#endif
}
__device__ __forceinline__ int maximum(int value) {
    const int adjacent = int(other(uint32_t(value)));
    return adjacent > value ? adjacent : value;
}
__device__ __forceinline__ Value finish(Value carry, uint32_t partial,
    int exponent, bool negative) {
    uint32_t modulo = partial + other(partial);
    const unsigned shift = unsigned(exponent - carry.exponent);
    const uint32_t aligned = shift >= 32u ? 0u : (carry.significand << 2u) >> shift;
    modulo += carry.negative ? 0u - aligned : aligned;
    const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo, negative);
    return qrt_sm121_wave16::normalize(sum.magnitude, sum.negative, exponent);
}
__device__ __forceinline__ Value integer_group(Value carry, const uint32_t* pairs) {
    uint32_t products[8];
    int exponent = carry.exponent > -133 ? carry.exponent : -133;
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) {
        products[i] = qrt_sm121_group16::pack_product(original::multiply_bf16(
            uint16_t(pairs[i]), uint16_t(pairs[i] >> 16u), -133));
        const int e = qrt_sm121_group16::packed_exponent(products[i]);
        exponent = e > exponent ? e : exponent;
    }
    exponent = maximum(exponent);
    uint32_t partial = 0u;
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) {
        const unsigned shift = unsigned(exponent - qrt_sm121_group16::packed_exponent(products[i]));
        const uint32_t magnitude = shift >= 32u ? 0u : ((products[i] & 65535u) << 11u) >> shift;
        partial += products[i] & 0x80000000u ? 0u - magnitude : magnitude;
    }
    return finish(carry, partial, exponent, (products[0] >> 31u) != 0u);
}

template<bool Trace = false>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
    unsigned count, bool eligible_rows, uint32_t* trace = nullptr, Stats* stats = nullptr) {
    const unsigned lane = threadIdx.x & 1u;
    Value carry{0u, -133, false}; Stats totals;
#pragma unroll 1
    for (unsigned base = 0u; base < count; base += 16u) {
        uint32_t pairs[8]; float products[8];
        int exponent = carry.exponent > -133 ? carry.exponent : -133;
#pragma unroll
        for (unsigned part = 0u; part < 2u; ++part) {
            uint64_t a, b;
            __builtin_memcpy(&a, left + base + lane * 8u + part * 4u, sizeof(a));
            __builtin_memcpy(&b, right + base + lane * 8u + part * 4u, sizeof(b));
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) {
                const unsigned item = part * 4u + i;
                const uint16_t x = uint16_t(a >> (i * 16u)), y = uint16_t(b >> (i * 16u));
                pairs[item] = uint32_t(x) | (uint32_t(y) << 16u);
                if (eligible_rows) {
                    products[item] = alignment::from_bits(uint32_t(x) << 16u) * alignment::from_bits(uint32_t(y) << 16u);
                    const int e = (x & 0x7fffu) && (y & 0x7fffu)
                        ? int((x >> 7u) & 255u) + int((y >> 7u) & 255u) - 254 : -133;
                    exponent = e > exponent ? e : exponent;
                }
            }
        }
        bool floating = false;
        if (eligible_rows) {
            exponent = maximum(exponent);
            if (exponent == -133 && !carry.significand) {
                carry = {0u, -133, false}; floating = true;
            } else if (exponent >= -101 && exponent <= 151) {
                const float scale = alignment::from_bits(uint32_t(152 - exponent) << 23u);
                uint32_t partial = 0u;
#pragma unroll
                for (unsigned i = 0u; i < 8u; ++i) partial += uint32_t(int32_t(products[i] * scale));
                carry = finish(carry, partial, exponent, ((pairs[0] ^ (pairs[0] >> 16u)) & 0x8000u) != 0u);
                floating = true;
            }
        }
        if (!floating) carry = integer_group(carry, pairs);
        if (stats) { totals.floating += unsigned(floating); totals.integer += unsigned(!floating); }
        if constexpr (Trace) if (!lane) {
            const float value = original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
            __builtin_memcpy(trace + base / 16u, &value, 4u);
        }
    }
    if (stats) *stats = totals;
    return lane ? 0.0f : original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_dual_lane
