#pragma once
#include "sm121_silu_table.h"
#include "../triton_moe/routed_consumer_interval.h"

#if defined(__HIPCC__)
#define QRT_CONV_HD __host__ __device__
#else
#define QRT_CONV_HD
#endif
// Isolated certificate for the original four-tap BF16-product/F32-sum
// convolution and fully enumerated SM121 SiLU table. No runtime dispatch.
namespace qrt_conv_consumer {
namespace c = qrt_routed_consumer;
struct Interval { float low = 0.0f, high = 0.0f; bool valid = false; };
struct Certificate { Interval sum; uint16_t output = 0u; bool constant = false; };

QRT_CONV_HD inline float multiply(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __fmul_rn(a, b);
#else
    volatile float result = a * b; return result;
#endif
}
QRT_CONV_HD inline float add(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __fadd_rn(a, b);
#else
    volatile float result = a + b; return result;
#endif
}
QRT_CONV_HD inline bool ordinary_term(uint16_t value) {
    const unsigned magnitude = value & 0x7fffu;
    // Decline tiny products rather than rely on host/device denormal modes.
    return !magnitude || (magnitude >= 0x1000u && magnitude < 0x7f80u);
}
QRT_CONV_HD inline c::Range endpoint(float raw, float error, bool selected) {
    if (selected) {
        const auto word = c::bits(raw);
        if (!c::finite(raw) || !c::finite(error) || error < 0.0f || ((word >> 23u) & 255u) < 32u) return {};
        const float lower = c::outward(raw-error,false), upper = c::outward(raw+error,true);
        if (!c::finite(lower) || !c::finite(upper)) return {};
        uint16_t low = c::ordered(c::rounded(lower)), high = c::ordered(c::rounded(upper));
        const uint16_t a = c::ordered(uint16_t(word>>16u)), b = c::ordered(uint16_t((word>>16u)+1u));
        low = a < low ? a : low; low = b < low ? b : low;
        high = a > high ? a : high; high = b > high ? b : high;
        // Unlike the routed activation certificate, convolution encloses a
        // monotone product without enumerating interior BF16 encodings. Its
        // interval need not be limited to nine representable input values.
        if (low < 0x0080u || high > 0xff7fu || high < low) return {};
        return {low,high,true};
    }
    const uint16_t rounded = c::rounded(raw), key = c::ordered(rounded);
    return c::finite(raw) && c::finite(c::widen(rounded)) ? c::Range{key,key,true} : c::Range{};
}

QRT_CONV_HD inline Interval product(c::Range input, uint16_t weight) {
    if (!input.valid || input.low > input.high || !c::finite(c::widen(weight))) return {};
    const uint16_t low = c::unordered(input.low), high = c::unordered(input.high);
    if (!c::finite(c::widen(low)) || !c::finite(c::widen(high))) return {};
    // Reject possible FP32 underflow before doing the device multiply.
    const unsigned ew = (weight >> 7u) & 255u;
    for (unsigned i = 0u; i < 2u; ++i) {
        const uint16_t value = i ? high : low;
        const unsigned ex = (value >> 7u) & 255u;
        if ((value & 0x7fffu) && (weight & 0x7fffu) && (!ex || !ew || ex + ew <= 128u)) return {};
    }
    const uint16_t a = c::rounded(multiply(c::widen(low), c::widen(weight)));
    const uint16_t b = c::rounded(multiply(c::widen(high), c::widen(weight)));
    if (!ordinary_term(a) || !ordinary_term(b)) return {};
    const float av = c::widen(a), bv = c::widen(b);
    return weight & 0x8000u ? Interval{bv,av,true} : Interval{av,bv,true};
}

QRT_CONV_HD inline Interval sum(const c::Range (&inputs)[4], const uint16_t (&weights)[4], unsigned present) {
    if (present & ~15u) return {};
    Interval result{0.0f,0.0f,true};
    for (unsigned tap = 0u; tap < 4u; ++tap) if (present & (1u << tap)) {
        const auto term = product(inputs[tap],weights[tap]);
        if (!term.valid) return {};
        result.low = add(result.low,term.low); result.high = add(result.high,term.high);
        if (!c::finite(result.low) || !c::finite(result.high) || result.low > result.high) return {};
    }
    return result;
}

QRT_CONV_HD inline unsigned transition(const unsigned char* table, uint32_t raw) {
    const auto* directory = reinterpret_cast<const uint32_t*>(table + 64u);
    const auto* keys = reinterpret_cast<const uint32_t*>(table + qrt_sm121_silu::key_start);
    unsigned low = directory[raw >> 16u], high = directory[(raw >> 16u) + 1u] + 1u;
    while (low < high) {
        const unsigned middle = (low + high) / 2u;
        if (keys[middle] <= raw) low = middle + 1u; else high = middle;
    }
    return low - 1u;
}

QRT_CONV_HD inline bool table_constant(const unsigned char* table, Interval interval, uint16_t* output) {
    if (!table || !output || !interval.valid || !c::finite(interval.low) ||
        !c::finite(interval.high) || interval.low > interval.high) return false;
    uint32_t a = c::bits(interval.low), b = c::bits(interval.high);
    // Raw encodings are monotone within either sign. Crossing signed zero is
    // conservatively declined. A negative SiLU turning point is safe only if
    // the whole interval still belongs to one enumerated output segment.
    if ((a ^ b) & 0x80000000u) return false;
    if (a > b) { const uint32_t swap = a; a = b; b = swap; }
    const unsigned first = transition(table,a);
    const auto* keys = reinterpret_cast<const uint32_t*>(table + qrt_sm121_silu::key_start);
    if (first + 1u < qrt_sm121_silu::transition_count && b >= keys[first + 1u]) return false;
    const auto* values = reinterpret_cast<const uint16_t*>(table + qrt_sm121_silu::value_start);
    const uint16_t result = values[first];
    if (!c::finite(c::widen(result))) return false;
    *output = result; return true;
}

QRT_CONV_HD inline Certificate certify(const c::Range (&inputs)[4], const uint16_t (&weights)[4],
    unsigned present, const unsigned char* silu) {
    Certificate result; result.sum = sum(inputs,weights,present);
    result.constant = table_constant(silu,result.sum,&result.output); return result;
}

QRT_CONV_HD inline bool can_omit(unsigned token, unsigned tokens, const bool (&following_outputs)[4]) {
    // The last three projected rows seed the next request's convolution halo.
    // Preserving only this prefill's outputs is insufficient for continuation.
    if (!tokens || token >= tokens || tokens - token <= 3u) return false;
    return following_outputs[0] && following_outputs[1] && following_outputs[2] && following_outputs[3];
}
} // namespace qrt_conv_consumer
#undef QRT_CONV_HD
