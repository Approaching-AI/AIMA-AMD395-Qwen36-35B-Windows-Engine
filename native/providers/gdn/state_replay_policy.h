#pragma once
#include <cstdlib>
#include <cstring>

namespace qrt_fla_state_replay_policy {
// 0: retained state; 1: narrow fast + retained replay;
// 2: narrow fast + product/certified retry + retained replay.
inline int mode() {
    const char* value = std::getenv("QRT_FLA_GDN_STATE_REPLAY");
    if (!value || !*value || !std::strcmp(value, "0")) return 0;
    if (!std::strcmp(value, "1")) return 1;
    return !std::strcmp(value, "2") ? 2 : -1;
}
inline bool setting(const char* name, const char* expected) {
    const char* value = std::getenv(name);
    return value && !std::strcmp(value, expected);
}
inline bool disabled(const char* name) {
    const char* value = std::getenv(name);
    return !value || !*value || !std::strcmp(value, "0");
}
inline bool selected(int mode, unsigned checkpoints, unsigned count) {
    return (mode == 1 || mode == 2) && !checkpoints && count && count <= 1024u &&
        setting("QRT_FLA_GDN_STATE_BLACKWELL", "1") &&
        setting("QRT_FLA_GDN_BATCHED_EXACT", "1") &&
        setting("QRT_FLA_GDN_COOPERATIVE_EXACT", "1") &&
        setting("QRT_FLA_GDN_SCALAR_FLOAT_MATRICES", "1") &&
        setting("QRT_FLA_GDN_SCALAR_FLOAT_STATE", "8") &&
        setting("QRT_FLA_GDN_PAIRED_SCORE_ARENAS", "1") &&
        disabled("QRT_FLA_GDN_COARSE_INTERVAL") &&
        disabled("QRT_FLA_GDN_FUSED_STATE_OUTPUT");
}
constexpr unsigned receipt_count = 32u * 16u;
constexpr unsigned receipt_bytes = receipt_count * sizeof(unsigned);
}
