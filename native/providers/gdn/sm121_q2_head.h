#pragma once
#include "sm121_q2_recurrent_layout.h"
#include "sm121_mtp_head.h"
#include "../moe_accumulator/sm121_q1_moe.h"

namespace qrt_sm121_q2 {
// Original two-row target logits use K16 MMA arithmetic, as verified against
// complete original target logit tensors. The MTP drafter keeps its own head.
inline hipError_t launch_target_head(const uint16_t* weights, const uint16_t* input,
    uint16_t* logits, uint32_t* tokens, float* values, uint32_t* invalid,
    unsigned maximum_blocks = 1024u, hipStream_t stream = nullptr) {
    using recurrent_detail::Span;
    constexpr unsigned vocabulary = qrt_sm121_mtp::head_vocabulary;
    if (!maximum_blocks || maximum_blocks > 4096u) return hipErrorInvalidValue;
    const Span reads[] = {{weights, size_t(vocabulary)*2048u*2u, 2u}, {input, 8192u, 2u}};
    const Span writes[] = {{logits, size_t(2u)*vocabulary*2u, 2u},
        {tokens, 8u, 4u}, {values, 8u, 4u}, {invalid, 4u, 4u}};
    for (unsigned i = 0; i < 4u; ++i) {
        for (unsigned j = 0; j < i; ++j)
            if (!recurrent_detail::disjoint(writes[i], writes[j])) return hipErrorInvalidValue;
        for (const auto& read : reads)
            if (!recurrent_detail::disjoint(writes[i], read)) return hipErrorInvalidValue;
    }
    auto status = hipMemsetAsync(invalid, 0, 4u, stream);
    if (status != hipSuccess) return status;
    const unsigned capacity = maximum_blocks*16u;
    for (unsigned row = 0; row < 2u; ++row) for (unsigned first = 0; first < vocabulary; first += capacity) {
        const unsigned remaining = vocabulary - first, count = remaining < capacity ? remaining : capacity;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_sm121_q1_moe::projection<2048u>),
            dim3((count + 15u)/16u), dim3(256u), 0u, stream,
            input + size_t(row)*2048u, weights + size_t(first)*2048u,
            logits + size_t(row)*vocabulary + first, count);
        status = hipGetLastError(); if (status != hipSuccess) return status;
    }
    hipLaunchKernelGGL(qrt_sm121_mtp::head_argmax, dim3(2u), dim3(256u), 0u, stream,
        logits, tokens, values, invalid);
    return hipGetLastError();
}
} // namespace qrt_sm121_q2
