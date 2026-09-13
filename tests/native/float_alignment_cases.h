#ifndef QRT_FLOAT_ALIGNMENT_CASES_H
#define QRT_FLOAT_ALIGNMENT_CASES_H
#include "../../native/providers/moe_accumulator/sm121_float_alignment.h"
#include "../../native/providers/moe_accumulator/sm121_canonical_normalize.h"
#if defined(__HIPCC__)
#define QRT_CASE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_CASE_INLINE inline
#endif
namespace qrt_float_alignment_cases {
namespace original = qrt_q1_moe_hawkeye;
namespace align = qrt_sm121_float_alignment;
struct Pair { uint16_t left, right; };
struct Result { uint32_t bits, accepted, fallback; };
QRT_CASE_INLINE uint32_t random(uint32_t x) {
    x ^= x << 13u; x ^= x >> 17u; x ^= x << 5u; return x;
}
QRT_CASE_INLINE Pair input(unsigned row, unsigned group, unsigned i) {
    const uint32_t a = random(0x3958192u ^ (row * 7919u + group * 997u + i * 17u));
    const uint32_t b = random(a ^ 0x8192395u);
    const unsigned mode = row % 8u;
    if (mode < 3u) {
        const uint16_t magnitude = uint16_t(((row / 8u % 127u + 64u) << 7u) | 127u);
        return {uint16_t(magnitude | (mode == 1u || (mode == 2u && i % 2u) ? 0x8000u : 0u)), magnitude};
    }
    if (mode == 3u) return {uint16_t((a & 0x807fu) | ((100u + a % 51u) << 7u)), uint16_t((b & 0x807fu) | ((100u + b % 51u) << 7u))};
    if (mode == 4u) return {uint16_t((a & 0x807fu) | ((64u + a % 127u) << 7u)), uint16_t((b & 0x807fu) | ((64u + b % 127u) << 7u))};
    if (mode == 5u) return {uint16_t((a & 0x807fu) | (64u << 7u)), uint16_t((b & 0x807fu) | (64u << 7u))};
    if (mode == 6u) return {uint16_t(a & 0x8000u), uint16_t(b)};
    return {uint16_t(row / 8u * 256u + group * 16u + i), uint16_t(b)};
}
QRT_CASE_INLINE uint32_t output_bits(original::Value carry) {
    const float value = original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
    uint32_t bits; __builtin_memcpy(&bits, &value, 4u); return bits;
}
QRT_CASE_INLINE Result candidate(unsigned row, uint32_t* trace = nullptr) {
    original::Value carry{0u, -133, false}; unsigned accepted = 0u, fallback = 0u;
    for (unsigned group = 0u; group < 16u; ++group) {
        align::Group products; bool eligible = true;
        for (unsigned i = 0u; i < 16u; ++i) {
            const auto p = input(row, group, i);
            eligible = eligible && align::eligible(p.left) && align::eligible(p.right);
        }
        qrt_sm121_group16::AlignedSum sum;
        if (eligible) for (unsigned i = 0u; i < 16u; ++i) {
            const auto p = input(row, group, i); products.set(i, p.left, p.right);
        }
        if (eligible && align::sum(carry, products, &sum)) ++accepted;
        else {
            ++fallback; uint32_t packed[16];
            for (unsigned i = 0u; i < 16u; ++i) {
                const auto p = input(row, group, i);
                packed[i] = qrt_sm121_group16::pack_product(original::multiply_bf16(p.left, p.right, -133));
            }
            sum = qrt_sm121_group16::sum_packed(carry, packed);
        }
        carry = qrt_sm121_canonical::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
        if (trace) trace[group] = output_bits(carry);
    }
    return {output_bits(carry), accepted, fallback};
}
inline uint32_t reference(unsigned row, uint32_t* trace = nullptr) {
    original::Value carry{0u, -133, false};
    for (unsigned group = 0u; group < 16u; ++group) {
        original::Value values[17]; values[0] = carry;
        for (unsigned i = 0u; i < 16u; ++i) {
            const auto p = input(row, group, i);
            values[i + 1u] = original::multiply_bf16(p.left, p.right, -133);
        }
        carry = original::group_sum<26, -133>(values, 17u);
        if (trace) trace[group] = output_bits(carry);
    }
    return output_bits(carry);
}
}  // namespace qrt_float_alignment_cases
#undef QRT_CASE_INLINE
#endif
