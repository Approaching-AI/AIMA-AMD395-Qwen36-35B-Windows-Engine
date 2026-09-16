#pragma once
#include <cstring>

namespace qrt_register_pv_policy {
// Decode and calls extending beyond the proved final-envelope domain keep
// their existing dispatch. An eligible prefill must select the tested owner.
inline bool select(const char* option, unsigned start, unsigned count,
                   bool final_bound, bool direct_operands, bool transposed_value,
                   bool selective_qk, bool& active) {
    active = false;
    if (option && *option && std::strcmp(option, "0") && std::strcmp(option, "1"))
        return false;
    if (!option || std::strcmp(option, "1") || count <= 1u ||
        start >= 8192u || count > 8192u - start)
        return true;
    if (!final_bound || !direct_operands || !transposed_value || selective_qk)
        return false;
    active = true;
    return true;
}
} // namespace qrt_register_pv_policy
