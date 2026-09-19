#pragma once
#include "inplace_probability_storage.h"

namespace qrt_long_probability_storage {
inline bool select(const char* option, bool long_pipeline, unsigned& mode) {
    mode = 0u;
    unsigned requested = 0u;
    if (option && *option) {
        if (option[1] || option[0] < '0' || option[0] > '2') return false;
        requested = unsigned(option[0] - '0');
    }
    if (long_pipeline) mode = requested;
    return true;
}
constexpr qrt_long_attention_layout::Layout layout(unsigned mode, unsigned queries, unsigned stride) {
    return mode == 0u ? qrt_long_attention_layout::layout(queries, stride)
        : mode <= 2u ? qrt_inplace_probability_storage::layout(queries, stride)
        : qrt_long_attention_layout::Layout{};
}
} // namespace qrt_long_probability_storage
