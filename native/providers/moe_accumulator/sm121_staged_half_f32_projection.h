#pragma once
#include "sm121_staged_half_projection.h"
#include "sm121_f32_carry.h"

// Component candidate shared by dense and routed replay. Retain the current
// lossless half operands, four-lane K16 ordering, unsigned modulo sum and
// normalization; hold each accepted zero/normal endpoint in one FP32 value.
// Any unsupported row, scale or endpoint restarts the complete original dot.
namespace qrt_sm121_staged_half_f32_projection {
namespace staged = qrt_sm121_staged_half_projection;
namespace half = qrt_sm121_scaled_half_products;
namespace f32 = qrt_sm121_f32_carry;
using Row = staged::Row;
struct Stats { unsigned accepted_groups = 0u; bool restarted = false; };

template<bool AllNonzero>
__device__ __forceinline__ bool accumulate(float carry,
    const staged::LaneOperands& operands, unsigned active, float* output) {
    const unsigned lane = threadIdx.x & 3u;
    float products[4]; uint32_t paired_maximum = 0u;
#pragma unroll
    for (unsigned i = 0u; i < 2u; ++i) {
        const uint32_t a = operands.left[i], b = operands.right[i];
        products[2u*i] = half::product<false>(a,b);
        products[2u*i+1u] = half::product<true>(a,b);
        uint32_t exponents = ((a >> 10u) & 0x001f001fu) + ((b >> 10u) & 0x001f001fu);
        if constexpr (!AllNonzero) {
            const unsigned live = (active >> (lane*4u+2u*i)) & 3u;
            exponents &= ((live & 1u) | ((live & 2u) << 15u)) * 65535u;
        }
        paired_maximum = qrt_sm121_prepared_integer_pairs::maximum_pair(paired_maximum,exponents);
    }
    int maximum = int((paired_maximum & 65535u) > (paired_maximum >> 16u)
        ? paired_maximum & 65535u : paired_maximum >> 16u);
    maximum = qrt_sm121_lane_reduce::maximum<4u>(maximum);
    const int unit = int(int16_t(operands.left_control)) + int(int16_t(operands.right_control));
    maximum = maximum - 30 + unit;
    const uint32_t absolute = f32::bits(carry) & 0x7fffffffu;
    const int carry_exponent = absolute ? int(absolute >> 23u) - 127 : -133;
    maximum = maximum > carry_exponent ? maximum : carry_exponent;
    maximum = maximum > -133 ? maximum : -133;
    const int power = 25 - maximum + unit;
    if (maximum < -101 || maximum > 127 || power > 127) return false;
    uint32_t modulo = 0u;
    if (power >= -126) {
        const float scale = f32::alignment::from_bits(uint32_t(127+power) << 23u);
#pragma unroll
        for (unsigned i = 0u; i < 4u; ++i) modulo += uint32_t(int32_t(products[i]*scale));
    }
    modulo = qrt_sm121_lane_reduce::sum<4u>(modulo);
    const float carry_scale = f32::alignment::from_bits(uint32_t(152-maximum) << 23u);
    modulo += uint32_t(int32_t(carry*carry_scale));
    const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo,
        ((operands.left[0] ^ operands.right[0]) & 0x8000u) != 0u);
    return f32::normalize<0u>(sum.magnitude,sum.negative,maximum,output);
}

template<unsigned StagingGroups, bool Audit = false>
__device__ __forceinline__ float dot(const Row* left, const Row* right,
    unsigned width, uint32_t* trace = nullptr, Stats* stats = nullptr) {
    static_assert(StagingGroups == 2u || StagingGroups == 4u);
    const unsigned lane = threadIdx.x & 3u, groups = width/16u;
    float carry = 0.0f; unsigned accepted = 0u;
    bool restart = false;
#pragma unroll 1
    for (unsigned base = 0u; base < groups; base += StagingGroups) {
        staged::LaneOperands operands[StagingGroups];
        const unsigned count = groups-base < StagingGroups ? groups-base : StagingGroups;
#pragma unroll
        for (unsigned i = 0u; i < StagingGroups; ++i)
            if (i < count) operands[i] = staged::load(left[base+i],right[base+i]);
#pragma unroll
        for (unsigned i = 0u; i < StagingGroups; ++i) if (i < count) {
            const auto& p = operands[i];
            if (int16_t(p.left_control) == -32768 || int16_t(p.right_control) == -32768) {
                restart = true; break;
            }
            const unsigned active = (p.left_control & p.right_control) >> 16u;
            float next = carry;
            const bool okay = !active || (active == 65535u
                ? accumulate<true>(carry,p,active,&next)
                : accumulate<false>(carry,p,active,&next));
            if (!okay) { restart = true; break; }
            carry = next; ++accepted;
            if constexpr (Audit) if (!lane && trace) trace[base+i] = f32::bits(carry);
        }
        if (restart) break;
    }
    if constexpr (Audit) if (!lane && stats) *stats = {accepted,restart};
    if (restart) {
        if constexpr (Audit) {
            staged::Value original{0u,-133,false};
            for (unsigned i = 0u; i < groups; ++i) {
                original = staged::accumulate(original,staged::load(left[i],right[i]));
                if (!lane && trace) trace[i] = f32::bits(qrt_q1_moe_hawkeye::value_to_float(original));
            }
            return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(
                qrt_sm121_group16::finish_accumulator(original));
        } else return staged::dot<StagingGroups>(left,right,width);
    }
    return lane ? 0.0f : carry;
}
} // namespace qrt_sm121_staged_half_f32_projection
