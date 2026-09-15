#ifndef QRT_SM121_MODULAR_INTEGER_CORE_H
#define QRT_SM121_MODULAR_INTEGER_CORE_H
#include "sm121_packed_core_remainder.h"
#include "sm121_float_alignment.h"
#if defined(__HIPCC__)
#define QRT_MODULAR_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MODULAR_INLINE inline
#endif

// Isolated component hypothesis, without a product dispatcher. An exact low16
// residue identifies the integer dot only IF the native FP16 WMMA error is
// strictly below32768. The recovery lemma does not establish that hardware
// bound. Native integer-oracle and captured full-score comparisons must test
// it; no existing PV envelope is imported as an integer precision guarantee.
namespace qrt_sm121_modular_core {
using Value = qrt_q1_moe_hawkeye::Value;
using AlignedSum = qrt_sm121_group16::AlignedSum;
struct Row {
    uint16_t original[18], half[16];
    uint32_t magnitudes[8];
    int unit, maximum;
    uint32_t exceptions, nonzero;
};
static_assert(sizeof(Row) == 116u); // Same capacity as the four-IU8 row.
constexpr int64_t maximum_dot = int64_t(16) * 32640 * 32640;

QRT_MODULAR_INLINE uint16_t half_bits(unsigned magnitude, bool negative) {
    // BF16-derived cores have at most eight significant bits and magnitude
    // <=32640. Both left shifting and toward-zero removal preserve this fact;
    // the FP16 encoding is exact. Arbitrary signed16 inputs are not supported.
    if (!magnitude) return 0u;
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned exponent = 31u - __clz(magnitude);
#else
    unsigned exponent = 0u;
    for (unsigned copy = magnitude; copy > 1u; copy >>= 1u) ++exponent;
#endif
    const unsigned significand = exponent <= 10u ? magnitude << (10u - exponent) : magnitude >> (exponent - 10u);
    return uint16_t((negative ? 0x8000u : 0u) | ((exponent + 15u) << 10u) | (significand & 1023u));
}
QRT_MODULAR_INLINE void prepare(Row& row) {
    qrt_sm121_integer_core::Row original{};
    for (unsigned i = 0u; i < 16u; ++i) original.original[i] = row.original[i];
    qrt_sm121_integer_core::prepare(original);
    row.unit = original.unit; row.maximum = original.maximum;
    row.exceptions = original.exceptions; row.nonzero = original.nonzero;
    row.original[16] = 0u; bool eligible = true;
    for (unsigned pair = 0u; pair < 8u; ++pair) {
        uint32_t packed = 0u;
        for (unsigned half = 0u; half < 2u; ++half) {
            const unsigned i = pair * 2u + half;
            const int core = qrt_sm121_integer_core::signed_core(row.original[i], row.unit);
            const unsigned magnitude = unsigned(core < 0 ? -core : core);
            const bool negative = (row.original[i] & 0x8000u) != 0u;
            row.half[i] = half_bits(magnitude, negative);
            packed |= magnitude << (half * 16u);
            row.original[16] |= uint16_t(unsigned(negative) << i);
            eligible = eligible && qrt_sm121_float_alignment::eligible(row.original[i]);
        }
        row.magnitudes[pair] = packed;
    }
    row.original[17] = uint16_t(eligible);
}

QRT_MODULAR_INLINE bool recover(float approximate, uint32_t residue, int64_t* result) {
    // Ordered comparisons reject NaNs/infinities before conversion. The broad
    // range also keeps every intermediate far inside int64. An exact midpoint
    // is ambiguous and falls back. Fractional values close to zero preserve
    // the direction of their truncation, including negative cancellation.
    if (!(approximate > -17179934720.0f && approximate < 17179934720.0f)) return false;
    const int64_t truncated = int64_t(approximate);
    const uint32_t offset = (residue - uint32_t(truncated)) & 65535u;
    if (offset == 32768u && approximate == float(truncated)) return false;
    int64_t value = truncated + offset;
    if (offset > 32768u || (offset == 32768u && approximate < float(truncated))) value -= 65536;
    if (value < -maximum_dot || value > maximum_dot) return false;
    *result = value;
    return true;
}
struct Remainders { int32_t residue, discarded; };
QRT_MODULAR_INLINE Remainders remainders(const Row& left, const Row& right, uint32_t mask) {
    Remainders result{};
    const unsigned signs = left.original[16] ^ right.original[16];
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (unsigned pair = 0u; pair < 8u; ++pair) {
        const uint32_t product = qrt_sm121_core_remainder::multiply_low16(left.magnitudes[pair], right.magnitudes[pair]);
        const int32_t low = int32_t(product & 65535u), high = int32_t(product >> 16u);
        result.residue += ((signs & (1u << (pair * 2u))) ? -low : low) +
            ((signs & (2u << (pair * 2u))) ? -high : high);
        const int32_t lo = int32_t(uint32_t(low) & mask), hi = int32_t(uint32_t(high) & mask);
        result.discarded += ((signs & (1u << (pair * 2u))) ? -lo : lo) +
            ((signs & (2u << (pair * 2u))) ? -hi : hi);
    }
    return result;
}
QRT_MODULAR_INLINE int paired_maximum(const Row& left, const Row& right, int maximum) {
    const uint32_t nonzero = left.nonzero & right.nonzero;
    for (unsigned pair = 0u; pair < 8u; ++pair) {
        const uint32_t a = uint32_t(left.original[2u * pair]) | (uint32_t(left.original[2u * pair + 1u]) << 16u);
        const uint32_t b = uint32_t(right.original[2u * pair]) | (uint32_t(right.original[2u * pair + 1u]) << 16u);
        const uint32_t sum = ((a >> 7u) & 0x00ff00ffu) + ((b >> 7u) & 0x00ff00ffu);
        const int first = (nonzero & (1u << (2u * pair))) ? int(sum & 65535u) - 254 : -133;
        const int second = (nonzero & (2u << (2u * pair))) ? int(sum >> 16u) - 254 : -133;
        maximum = first > maximum ? first : maximum; maximum = second > maximum ? second : maximum;
    }
    return maximum;
}
QRT_MODULAR_INLINE bool sum(Value carry, const Row& left, const Row& right, float approximate, AlignedSum* output) {
    if (left.unit < 0 || right.unit < 0) return false;
    int maximum = carry.exponent > -133 ? carry.exponent : -133;
    if (left.maximum + right.maximum - 254 > maximum) maximum = paired_maximum(left, right, maximum);
    const int shift = maximum - (left.unit + right.unit - 254) - 11;
    if (shift < -25 || shift > 16) return false;
    const auto low = remainders(left, right, shift > 0 ? (1u << shift) - 1u : 0u);
    int64_t mathematical;
    if (!recover(approximate, uint32_t(low.residue), &mathematical)) return false;
    const int64_t compensated = mathematical - low.discarded;
    const int64_t products = shift <= 0 ? compensated * (int64_t(1) << (-shift)) :
        compensated < 0 ? -int64_t(uint64_t(-compensated) >> shift) : int64_t(uint64_t(compensated) >> shift);
    int64_t corrections = 0;
    uint32_t pending = (left.exceptions | right.exceptions) & left.nonzero & right.nonzero;
    while (pending) {
        const unsigned i = qrt_sm121_integer_core::first_bit(pending); pending &= pending - 1u;
        const uint16_t a = left.original[i], b = right.original[i];
        const auto original = qrt_q1_moe_hawkeye::multiply_bf16(a, b, -133);
        const unsigned original_shift = unsigned(maximum - original.exponent);
        const uint32_t aligned = original_shift >= 32u ? 0u : (original.significand << 2u) >> original_shift;
        const unsigned half = (i & 1u) * 16u;
        const uint32_t ac = (left.magnitudes[i / 2u] >> half) & 65535u, bc = (right.magnitudes[i / 2u] >> half) & 65535u;
        const uint64_t magnitude = uint64_t(ac) * bc;
        const uint64_t core_aligned = shift <= 0 ? magnitude << (-shift) : magnitude >> shift;
        const int64_t difference = int64_t(aligned) - int64_t(core_aligned);
        corrections += ((a ^ b) & 0x8000u) ? -difference : difference;
    }
    const unsigned carry_shift = unsigned(maximum - carry.exponent);
    const uint32_t aligned = carry_shift >= 32u ? 0u : (carry.significand << 2u) >> carry_shift;
    const int64_t total = products + corrections + (carry.negative ? -int64_t(aligned) : int64_t(aligned));
    if (total < -int64_t(UINT32_MAX) || total > int64_t(UINT32_MAX)) return false;
    *output = {{uint32_t(total < 0 ? -total : total), total < 0}, maximum};
    return true;
}
QRT_MODULAR_INLINE AlignedSum fallback(Value carry, const Row& left, const Row& right) {
    AlignedSum result;
    if (left.original[17] && right.original[17] && (!carry.significand || carry.exponent >= -126)) {
        qrt_sm121_float_alignment::Group group;
        for (unsigned i = 0u; i < 16u; ++i) group.set(i, left.original[i], right.original[i]);
        if (qrt_sm121_float_alignment::sum(carry, group, &result)) return result;
    }
    uint32_t products[16];
    for (unsigned i = 0u; i < 16u; ++i) products[i] = qrt_sm121_group16::pack_product(
        qrt_q1_moe_hawkeye::multiply_bf16(left.original[i], right.original[i], -133));
    return qrt_sm121_group16::sum_packed(carry, products);
}
#if defined(__HIPCC__)
using F16x16 = _Float16 __attribute__((ext_vector_type(16)));
using F32x8 = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ F32x8 products(const Row& left, const Row& right) {
    F16x16 a{}, b{};
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        a[i] = __builtin_bit_cast(_Float16, left.half[i]); b[i] = __builtin_bit_cast(_Float16, right.half[i]);
    }
    const F32x8 zero{};
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, zero);
}
#endif
}
#undef QRT_MODULAR_INLINE
#endif
