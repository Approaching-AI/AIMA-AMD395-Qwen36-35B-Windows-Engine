#pragma once

// Numerical experiment, disabled by default. The matrix prefix keeps the
// selective probability repair but has an approximate denominator. Completing
// an original-accumulator suffix is not a proof of model-level equivalence.
namespace qrt_selective_qk_tail {
inline bool parse(const char* text, unsigned& tail) {
    if (!text || !*text) { tail = 0u; return true; }
    if (text[0] == '0' && text[1]) return false;
    unsigned value = 0u;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        value = value * 10u + unsigned(*p - '0');
        if (value > 8192u) return false;
    }
    tail = value;
    return true;
}

inline unsigned prefix_queries(unsigned tail, unsigned start, unsigned count,
                               unsigned batch) {
    if (!tail || tail > 8192u || start || count <= tail || count > 8192u ||
        (batch != 32u && batch != 64u && batch != 128u)) return 0u;
    // A whole slab takes one route. Rounding the prefix down makes the exact
    // suffix at least as large as requested, including partial final slabs.
    return (count - tail) / batch * batch;
}
} // namespace qrt_selective_qk_tail
