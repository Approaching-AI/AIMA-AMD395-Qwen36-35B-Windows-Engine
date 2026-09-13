#ifndef QRT_SM121_RANGE_PROJECTION_H
#define QRT_SM121_RANGE_PROJECTION_H
#include "sm121_scalar_projection.h"
#include "sm121_range_normalize.h"

// Component-only exact specialization. Bit0 retains the original float
// eligibility; bit1 certifies finite normal K16 normalization for this width.
// Neither existing provider dispatch nor existing metadata is changed.
namespace qrt_sm121_range_projection {
namespace range = qrt_sm121_range;
namespace alignment = qrt_sm121_float_alignment;
using Value = qrt_q1_moe_hawkeye::Value;

__global__ void classify_rows_kernel(const uint16_t* source, unsigned* flags,
    unsigned rows, unsigned columns) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned invalid;
    if (!threadIdx.x) invalid = 0u;
    __syncthreads();
    unsigned bad = range::bounded_shape(columns) ? 0u : 2u;
    for (unsigned k = threadIdx.x; k < columns; k += blockDim.x) {
        const uint16_t value = source[size_t(row) * columns + k];
        bad |= unsigned(!alignment::eligible(value));
        bad |= unsigned(!range::bounded_operand(value)) << 1u;
    }
    if (bad) atomicOr(&invalid, bad);
    __syncthreads();
    if (!threadIdx.x) flags[row] = (~invalid) & 3u;
}

template<unsigned Lanes, unsigned Staging = 1u>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
    unsigned count, unsigned common_flags) {
    static_assert(Lanes == 4u || Lanes == 16u);
    static_assert(Staging == 1u || Staging == 4u);
    if (!(common_flags & 2u) || !range::bounded_shape(count))
        return qrt_sm121_scalar_projection::validated_dot<Lanes, Staging>(
            left, right, count, (common_flags & 1u) != 0u);
    constexpr unsigned items = 16u / Lanes;
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    Value carry{0u, -133, false};
#pragma unroll 1
    for (unsigned base = 0u; base < count; base += 16u * Staging) {
        float products[Staging][items];
        int maxima[Staging];
#pragma unroll
        for (unsigned group = 0u; group < Staging; ++group) if (base + group * 16u < count) {
            using Packed = typename std::conditional<Lanes == 4u, uint64_t, uint16_t>::type;
            Packed a, b;
            __builtin_memcpy(&a, left + base + group * 16u + lane * items, sizeof(a));
            __builtin_memcpy(&b, right + base + group * 16u + lane * items, sizeof(b));
            int maximum = -133;
#pragma unroll
            for (unsigned i = 0u; i < items; ++i) {
                const uint16_t x = uint16_t(a >> (i * 16u)), y = uint16_t(b >> (i * 16u));
                products[group][i] = alignment::from_bits(uint32_t(x) << 16u) *
                    alignment::from_bits(uint32_t(y) << 16u);
                const int exponent = (x & 0x7fffu) && (y & 0x7fffu) ?
                    int((x >> 7u) & 255u) + int((y >> 7u) & 255u) - 254 : -133;
                maximum = exponent > maximum ? exponent : maximum;
            }
            maxima[group] = qrt_sm121_lane_reduce::maximum<Lanes>(maximum);
        }
#pragma unroll
        for (unsigned group = 0u; group < Staging; ++group) if (base + group * 16u < count) {
            const int maximum = maxima[group] > carry.exponent ? maxima[group] : carry.exponent;
            // A maximum below -100 proves all sixteen products are zero.
            // Preserve a tiny nonzero carry exactly rather than clamping it.
            if (maximum < -100) continue;
            const float scale = alignment::from_bits(uint32_t(127 + 25 - maximum) << 23u);
            uint32_t modulo = 0u;
#pragma unroll
            for (unsigned i = 0u; i < items; ++i)
                modulo += uint32_t(int32_t(products[group][i] * scale));
            modulo = qrt_sm121_lane_reduce::sum<Lanes>(modulo);
            const unsigned shift = unsigned(maximum - carry.exponent);
            const uint32_t aligned = shift >= 32u ? 0u : (carry.significand << 2u) >> shift;
            modulo += carry.negative ? 0u - aligned : aligned;
            uint32_t first_bits; __builtin_memcpy(&first_bits, &products[group][0], 4u);
            const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo, (first_bits >> 31u) != 0u);
            carry = range::normalize(sum.magnitude, sum.negative, maximum);
        }
    }
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_range_projection
#endif
