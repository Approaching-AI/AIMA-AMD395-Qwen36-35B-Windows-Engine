#ifndef QRT_F32_CARRY_CASES_H
#define QRT_F32_CARRY_CASES_H
#include "float_alignment_cases.h"
#include "../../native/providers/moe_accumulator/sm121_f32_carry.h"
#if defined(__HIPCC__)
#define QRT_F32_CASE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_F32_CASE_INLINE inline
#endif
namespace qrt_f32_carry_cases {
namespace cases = qrt_float_alignment_cases;
namespace fast = qrt_sm121_f32_carry;
namespace original = qrt_q1_moe_hawkeye;

template<unsigned Method>
QRT_F32_CASE_INLINE cases::Result candidate(unsigned row, uint32_t* trace) {
    float carry = 0.0f; original::Value fallback{0u, -133, false};
    bool active = true; unsigned accepted = 0u, original_groups = 0u;
    for (unsigned group = 0u; group < 16u; ++group) {
        fast::alignment::Group products; bool eligible = true;
        for (unsigned i = 0u; i < 16u; ++i) {
            const auto p = cases::input(row, group, i);
            eligible = eligible && fast::alignment::eligible(p.left) && fast::alignment::eligible(p.right);
        }
        if (active && eligible) for (unsigned i = 0u; i < 16u; ++i) {
            const auto p = cases::input(row, group, i); products.set(i, p.left, p.right);
        }
        float next;
        if (active && eligible && fast::accumulate<Method>(carry, products, &next)) {
            carry = next; ++accepted;
        } else {
            if (active) {
                fallback = original::value_from_float(carry, -133);
                active = false;
            }
            ++original_groups; uint32_t packed[16];
            for (unsigned i = 0u; i < 16u; ++i) {
                const auto p = cases::input(row, group, i);
                packed[i] = qrt_sm121_group16::pack_product(original::multiply_bf16(p.left, p.right, -133));
            }
            const auto sum = qrt_sm121_group16::sum_packed(fallback, packed);
            fallback = qrt_sm121_canonical::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
        }
        trace[group] = active ? fast::bits(carry) : cases::output_bits(fallback);
    }
    return {trace[15], accepted, original_groups};
}
} // namespace qrt_f32_carry_cases
#undef QRT_F32_CASE_INLINE
#endif
