#pragma once
#include "../moe_accumulator/sm121_shared_gate.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MTP_HEAD_HD __host__ __device__
#else
#define QRT_MTP_HEAD_HD
#endif

namespace qrt_sm121_mtp {
constexpr unsigned head_vocabulary = 248320u;
struct HeadBest {
    uint32_t token = UINT32_MAX;
    float logit = -INFINITY;
};

// Greedy sampling consumes the rounded BF16 logits. Equal values retain the
// lower token ID, including signed zeros; nonfinite values invalidate the row.
QRT_MTP_HEAD_HD inline bool head_better(float logit, uint32_t token, const HeadBest& best) {
    return logit > best.logit || (logit == best.logit && token < best.token);
}
QRT_MTP_HEAD_HD inline bool head_candidate(uint16_t bits, uint32_t token, HeadBest* best) {
    if ((bits & 0x7f80u) == 0x7f80u || token >= head_vocabulary) return false;
    const float logit = qrt_sm121_shared_gate::value(bits);
    if (head_better(logit, token, *best)) *best = {token, logit};
    return true;
}
} // namespace qrt_sm121_mtp
#undef QRT_MTP_HEAD_HD
