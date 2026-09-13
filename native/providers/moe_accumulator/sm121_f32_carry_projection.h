#ifndef QRT_SM121_F32_CARRY_PROJECTION_H
#define QRT_SM121_F32_CARRY_PROJECTION_H
#include "sm121_scalar_projection.h"
#include "sm121_f32_carry.h"

// Component experiment; existing dense and routed-MoE dispatch remain intact.
namespace qrt_sm121_f32_carry_projection {
namespace fast = qrt_sm121_f32_carry;
namespace original = qrt_sm121_scalar_projection;

template<unsigned Method, unsigned Lanes>
__device__ __forceinline__ bool accumulate(float carry,
    const qrt_sm121_float_subgroup::Product* products, float* output) {
    const uint32_t absolute_carry = fast::bits(carry) & 0x7fffffffu;
    int maximum = absolute_carry ? int(absolute_carry >> 23u) - 127 : -133;
#pragma unroll
    for (unsigned i = 0u; i < 16u / Lanes; ++i)
        maximum = products[i].exponent > maximum ? products[i].exponent : maximum;
    maximum = qrt_sm121_lane_reduce::maximum<Lanes>(maximum);
    if (maximum == -133 && !absolute_carry) { *output = 0.0f; return true; }
    if (maximum < -101 || maximum > 127) return false;
    const float scale = fast::alignment::from_bits(uint32_t(152 - maximum) << 23u);
    uint32_t modulo = 0u;
#pragma unroll
    for (unsigned i = 0u; i < 16u / Lanes; ++i)
        modulo += uint32_t(int32_t(products[i].value * scale));
    modulo = qrt_sm121_lane_reduce::sum<Lanes>(modulo);
    modulo += uint32_t(int32_t(carry * scale));
    const uint32_t pair = products[0].original;
    const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo,
        ((pair ^ (pair >> 16u)) & 0x8000u) != 0u);
    return fast::normalize<Method>(sum.magnitude, sum.negative, maximum, output);
}

template<unsigned Method, unsigned Lanes, unsigned Staging = 1u>
__device__ __forceinline__ float dot(const uint16_t* left,
    const uint16_t* right, unsigned count, bool eligible) {
    static_assert(Method < 2u && (Lanes == 4u || Lanes == 8u || Lanes == 16u));
    static_assert(Staging == 1u || Staging == 4u || Staging == 8u);
    if (!eligible) return qrt_sm121_subgroup::dot<Lanes, Staging>(left, right, count);
    constexpr unsigned items = 16u / Lanes;
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    float carry = 0.0f;
#pragma unroll 1
    for (unsigned base = 0u; base < count; base += 16u * Staging) {
        qrt_sm121_float_subgroup::Product products[Staging][items];
#pragma unroll
        for (unsigned group = 0u; group < Staging; ++group) if (base + group * 16u < count) {
            using Packed = typename std::conditional<Lanes == 4u, uint64_t,
                typename std::conditional<Lanes == 8u, uint32_t, uint16_t>::type>::type;
            Packed a, b;
            __builtin_memcpy(&a, left + base + group * 16u + lane * items, sizeof(a));
            __builtin_memcpy(&b, right + base + group * 16u + lane * items, sizeof(b));
#pragma unroll
            for (unsigned i = 0u; i < items; ++i) {
                const uint16_t x = uint16_t(a >> (i * 16u)), y = uint16_t(b >> (i * 16u));
                const bool zero = !(x & 0x7fffu) || !(y & 0x7fffu);
                products[group][i] = {
                    fast::alignment::from_bits(uint32_t(x) << 16u) * fast::alignment::from_bits(uint32_t(y) << 16u),
                    uint32_t(x) | (uint32_t(y) << 16u),
                    zero ? -133 : int((x >> 7u) & 255u) + int((y >> 7u) & 255u) - 254};
            }
        }
#pragma unroll
        for (unsigned group = 0u; group < Staging; ++group) if (base + group * 16u < count) {
            float next;
            if (!accumulate<Method, Lanes>(carry, products[group], &next))
                return original::validated_dot<Lanes, Staging>(left, right, count, true);
            carry = next;
        }
    }
    return lane ? 0.0f : carry;
}
} // namespace qrt_sm121_f32_carry_projection
#endif
