#pragma once
#include <cstring>

namespace qrt_compact_decode_query {
// Single-query suffixes can borrow their caller's only Q row. The absolute
// position still controls causal history, score stride and ordered PV.
inline bool select(const char* option, unsigned first, unsigned count, bool& enabled) {
    enabled = false;
    if (option && *option && std::strcmp(option, "0") && std::strcmp(option, "1"))
        return false;
    enabled = first && count == 1u && option && std::strcmp(option, "1") == 0;
    return true;
}
} // namespace qrt_compact_decode_query
