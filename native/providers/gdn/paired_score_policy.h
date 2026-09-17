#pragma once
#include <cstdlib>
#include <cstring>

namespace qrt_fla_paired_policy {
inline int mode() {
    const char* value = std::getenv("QRT_FLA_GDN_PAIRED_SCORE_ARENAS");
    if (!value || !*value || !std::strcmp(value, "0")) return 0;
    return !std::strcmp(value, "1") ? 1 : -1;
}
inline bool selected(int mode, bool compatible, unsigned count) {
    return mode == 1 && compatible && count && count <= 1024u;
}
}
