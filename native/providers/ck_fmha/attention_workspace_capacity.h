#pragma once
#include "../sm121_attention_capacity.h"

namespace qrt_attention_workspace_capacity {
// A reservation is a minimum allocation capacity, never an input extent.
// Zero/absent retains demand growth. A later request can exceed the hint.
inline bool select(const char* option, unsigned needed, unsigned quantum,
                   unsigned& capacity, unsigned& requested) {
    constexpr unsigned maximum = qrt_sm121_attention_capacity::kTokens;
    if (!needed || needed > maximum || (quantum != 1u && quantum != 8192u))
        return false;
    unsigned reserve = 0u;
    if (option) {
        for (const char* p = option; *p; ++p) {
            if (*p < '0' || *p > '9') return false;
            const unsigned digit = static_cast<unsigned>(*p - '0');
            if (reserve > (maximum - digit) / 10u) return false;
            reserve = reserve * 10u + digit;
        }
    }
    const unsigned minimum = needed > reserve ? needed : reserve;
    const unsigned rounded = ((minimum - 1u) / quantum + 1u) * quantum;
    capacity = rounded > maximum ? maximum : rounded;
    requested = reserve;
    return true;
}
} // namespace qrt_attention_workspace_capacity
