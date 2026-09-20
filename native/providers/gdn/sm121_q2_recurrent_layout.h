#pragma once
#include "sm121_q2_recurrent_math.h"
#include <limits>

namespace qrt_sm121_q2 {
constexpr size_t staged_state_bytes = scheduled_rows * state_elements * sizeof(float);
constexpr size_t staged_core_bytes = scheduled_rows * core_elements * sizeof(uint16_t);

struct RecurrentViews {
    const uint16_t* convolution = nullptr; // [2,8192], original BF16 endpoint.
    const uint16_t* a = nullptr;           // [2,32]
    const uint16_t* b = nullptr;           // [2,32]
    const float* initial_state = nullptr; // [32,128,128], layer-declared layout.
    float* staged_states = nullptr;       // [2,32,128,128], same layout.
    uint16_t* staged_core = nullptr;      // [2,4096]
    bool key_major = false;
};

namespace recurrent_detail {
struct Span { const void* pointer; size_t bytes, alignment; };
inline bool valid_span(const Span& span) {
    const uintptr_t first = reinterpret_cast<uintptr_t>(span.pointer);
    return first && span.bytes && first % span.alignment == 0u &&
        span.bytes <= (std::numeric_limits<uintptr_t>::max)() - first;
}
inline bool disjoint(const Span& a, const Span& b) {
    if (!valid_span(a) || !valid_span(b)) return false;
    const auto x = reinterpret_cast<uintptr_t>(a.pointer), y = reinterpret_cast<uintptr_t>(b.pointer);
    return x + a.bytes <= y || y + b.bytes <= x;
}
} // namespace recurrent_detail

inline bool valid_recurrent_views(const RecurrentViews& view, const RecurrentTables& tables) {
    using recurrent_detail::Span;
    const Span writes[] = {{view.staged_states, staged_state_bytes, alignof(float)},
                           {view.staged_core, staged_core_bytes, alignof(uint16_t)}};
    const Span reads[] = {
        {view.convolution, 2u * 8192u * sizeof(uint16_t), alignof(uint16_t)},
        {view.a, 2u * 32u * sizeof(uint16_t), alignof(uint16_t)},
        {view.b, 2u * 32u * sizeof(uint16_t), alignof(uint16_t)},
        {view.initial_state, state_elements * sizeof(float), alignof(float)},
        {tables.g, 32u * 65536u * sizeof(float), alignof(float)},
        {tables.beta, 65536u * sizeof(float), alignof(float)},
        {tables.exp2, qrt_sm121_exp2::table_bytes, 1u},
        {tables.rsqrt, qrt_sm121_rsqrt::table_bytes, 1u}
    };
    if (!recurrent_detail::disjoint(writes[0], writes[1])) return false;
    for (const auto& write : writes)
        for (const auto& read : reads)
            if (!recurrent_detail::disjoint(write, read)) return false;
    return true;
}

// Selection is a borrowed completed result, not a cache publication. The
// enclosing target transaction owns GPU completion, rollback and accepted IDs.
inline const float* accepted_state(const RecurrentViews& view, unsigned accepted_rows) {
    return view.staged_states && accepted_rows >= 1u && accepted_rows <= scheduled_rows
        ? view.staged_states + size_t(accepted_rows - 1u) * state_elements : nullptr;
}
} // namespace qrt_sm121_q2
