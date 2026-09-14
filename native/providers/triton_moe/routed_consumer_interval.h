#pragma once
#include <cstdint>

#if defined(__HIPCC__)
#define QRT_CONSUMER_HD __host__ __device__
#else
#define QRT_CONSUMER_HD
#endif

namespace qrt_routed_consumer {
QRT_CONSUMER_HD inline uint32_t bits(float x) {
    uint32_t u; __builtin_memcpy(&u, &x, sizeof(u)); return u;
}
QRT_CONSUMER_HD inline float value(uint32_t u) {
    float x; __builtin_memcpy(&x, &u, sizeof(x)); return x;
}
QRT_CONSUMER_HD inline bool finite(float x) { return (bits(x) & 0x7fffffffu) < 0x7f800000u; }
QRT_CONSUMER_HD inline uint16_t rounded(float x) {
    const auto u = bits(x);
    return uint16_t((u + 0x7fffu + ((u >> 16u) & 1u)) >> 16u);
}
QRT_CONSUMER_HD inline float widen(uint16_t u) { return value(uint32_t(u) << 16u); }
QRT_CONSUMER_HD inline uint16_t ordered(uint16_t u) {
    return uint16_t(u & 0x8000u ? ~u : u ^ 0x8000u);
}
QRT_CONSUMER_HD inline uint16_t unordered(uint16_t u) {
    return uint16_t(u & 0x8000u ? u ^ 0x8000u : ~u);
}
QRT_CONSUMER_HD inline float outward(float x, bool up) {
    const uint32_t u = bits(x), magnitude = u & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) return x;
    if (!magnitude) return value(up ? 1u : 0x80000001u);
    return value(((u >> 31u) != unsigned(up)) ? u + 1u : u - 1u);
}
struct Range { uint16_t low = 0u, high = 0u; bool valid = false; };

// Retain the original two adjacent BF16 hypotheses as well as every endpoint
// in the outward absolute-error interval. Tiny, nonfinite, overflowing or wide
// intervals receive no certificate. The numerical error coefficient itself
// remains empirical; the audit must check corrected endpoints against it.
QRT_CONSUMER_HD inline Range range(float raw, float error) {
    const auto word = bits(raw);
    if (!finite(raw) || !finite(error) || error < 0.0f || ((word >> 23u) & 255u) < 32u)
        return {};
    const float lower = outward(raw - error, false), upper = outward(raw + error, true);
    if (!finite(lower) || !finite(upper)) return {};
    uint16_t low = ordered(rounded(lower)), high = ordered(rounded(upper));
    const uint16_t a = ordered(uint16_t(word >> 16u)), b = ordered(uint16_t((word >> 16u) + 1u));
    low = a < low ? a : low; low = b < low ? b : low;
    high = a > high ? a : high; high = b > high ? b : high;
    // At most nine table endpoints; never extrapolate SiLU monotonicity across
    // its negative minimum or silently clip a wider absolute-error interval.
    if (low < 0x0080u || high > 0xff7fu || high < low || unsigned(high - low) > 8u) return {};
    return {low, high, true};
}
QRT_CONSUMER_HD inline bool contains(Range r, float exact) {
    if (!r.valid || !finite(exact)) return false;
    const auto key = ordered(rounded(exact));
    return key >= r.low && key <= r.high;
}
QRT_CONSUMER_HD inline bool gate_constant(Range r, const uint16_t* table) {
    if (!r.valid || !table) return false;
    const uint16_t result = table[unordered(r.low)];
    if (!finite(widen(result))) return false;
    for (unsigned key = r.low + 1u; key <= r.high; ++key)
        if (table[unordered(uint16_t(key))] != result) return false;
    return true;
}
QRT_CONSUMER_HD inline uint16_t activated(uint16_t gate, uint16_t up, const uint16_t* table) {
    return rounded(widen(table[gate]) * widen(up));
}
QRT_CONSUMER_HD inline bool up_constant(Range r, uint16_t gate, const uint16_t* table) {
    if (!r.valid || !table || !finite(widen(table[gate]))) return false;
    const uint16_t result = activated(gate, unordered(r.low), table);
    if (!finite(widen(result))) return false;
    for (unsigned key = r.low + 1u; key <= r.high; ++key)
        if (activated(gate, unordered(uint16_t(key)), table) != result) return false;
    return true;
}
} // namespace qrt_routed_consumer
#undef QRT_CONSUMER_HD
