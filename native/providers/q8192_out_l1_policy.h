#pragma once
#include "moe_accumulator/bf16_positive_sum_bound.h"

namespace qrt_out_l1_policy {
constexpr unsigned variants = 5u;
#if defined(__HIPCC__)
#define QRT_OUT_L1_HD __host__ __device__
#else
#define QRT_OUT_L1_HD
#endif
QRT_OUT_L1_HD inline unsigned multiplier(unsigned index) {
    return index == 0u ? 1u : 1u << (index + 1u);
}
// The L1 sum has already received the existing positive-sum inflation.
// An outward power-of-two scale and the original Cauchy cap define a proposed
// empirical error envelope. This is not a proof that a coefficient suffices
// for a matrix producer: compare every removed endpoint on a GB10-valid run.
// Infinity/NaN from an excluded or failed magnitude row keeps the old bound.
QRT_OUT_L1_HD inline float capped(float l1_upper, float cauchy, unsigned factor) {
    const float scaled = qrt_bf16_positive_sum_bound::next_up(l1_upper * float(factor));
    return scaled < cauchy ? scaled : cauchy;
}
#undef QRT_OUT_L1_HD
} // namespace qrt_out_l1_policy
