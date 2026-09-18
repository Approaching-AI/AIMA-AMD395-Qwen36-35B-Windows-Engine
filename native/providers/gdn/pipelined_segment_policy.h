#pragma once
#include <cstdlib>
#include <cstring>

namespace qrt_fla_pipeline_policy {
inline int mode() {
    const char* value=std::getenv("QRT_FLA_GDN_PIPELINED_SEGMENTS");
    if (!value || !*value || !std::strcmp(value,"0")) return 0;
    if (!std::strcmp(value,"1")) return 1;
    return !std::strcmp(value,"2") ? 2 : -1;
}
inline bool selected(int mode, bool compatible, unsigned tokens,
                     bool checkpoints, bool diagnostic, bool disjoint) {
    return (mode==1 || mode==2) && compatible && tokens>1024u && tokens<=65536u &&
        !checkpoints && !diagnostic && disjoint;
}
constexpr unsigned slots=3u, window_segments=8u;
constexpr unsigned slot(unsigned segment) { return segment%slots; }
constexpr int prior_consumer(unsigned segment) {
    return segment>=slots ? int(segment-slots) : -1;
}
} // namespace qrt_fla_pipeline_policy
