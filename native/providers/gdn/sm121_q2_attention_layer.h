#pragma once
#include "sm121_q2_attention_block.h"
#include "sm121_mtp_residual.h"
#include "sm121_mtp_moe.h"

namespace qrt_sm121_q2 {
struct AttentionLayerViews {
    const uint16_t* hidden = nullptr;             // [2,2048]
    const uint16_t* residual = nullptr;           // [2,2048]
    const uint16_t* input_norm_weights = nullptr; // [2048]
    const uint16_t* post_norm_weights = nullptr;  // [2048]
    uint16_t* normalized_input = nullptr;         // [2,2048]
    uint16_t* input_residual = nullptr;           // [2,2048]
    uint16_t* moe_input = nullptr;                // [2,2048]
    uint16_t* output_residual = nullptr;          // [2,2048]
    AttentionBlockViews attention;
    qrt_sm121_mtp::MoeWeights moe_weights;
    void* moe_workspace = nullptr;
    size_t moe_workspace_bytes = 0;
};
struct AttentionLayerTables {
    AttentionBlockTables attention;
    qrt_sm121_mtp::MoeTables moe;
};

inline bool valid_attention_layer(const AttentionLayerViews& v, const AttentionLayerTables& t) {
    qrt_sm121_mtp::MoeBuffers moe;
    if (v.attention.normalized_input != v.normalized_input ||
        !valid_attention_block(v.attention, t.attention) ||
        !qrt_sm121_mtp::bind_moe_buffers(v.moe_workspace, v.moe_workspace_bytes, 2u, &moe))
        return false;
    using recurrent_detail::Span;
    const Span writes[] = {
        {v.normalized_input, 8192u, 2u}, {v.input_residual, 8192u, 2u},
        {v.moe_input, 8192u, 2u}, {v.output_residual, 8192u, 2u},
        {v.moe_workspace, qrt_sm121_mtp::moe_workspace_bytes(2u), 256u}
    };
    const Span reads[] = {
        {v.hidden, 8192u, 2u}, {v.residual, 8192u, 2u},
        {v.input_norm_weights, 4096u, 2u}, {v.post_norm_weights, 4096u, 2u},
        {v.moe_weights.router, 256u*2048u*2u, 2u},
        {v.moe_weights.shared_gate, 2048u*2u, 2u},
        {v.moe_weights.shared_gate_up, 1024u*2048u*2u, 2u},
        {v.moe_weights.shared_down, 2048u*512u*2u, 2u},
        {v.moe_weights.routed_gate_up, size_t(256u)*1024u*2048u*2u, 2u},
        {v.moe_weights.routed_down, size_t(256u)*2048u*512u*2u, 2u},
        {t.moe.silu, 65536u*2u, 2u}, {t.moe.sigmoid, 65536u*2u, 2u},
        {t.moe.router_exp_fraction, 8388608u*4u, 4u}
    };
    const auto block_writes = attention_block_detail::write_spans(v.attention);
    const auto block_reads = attention_block_detail::read_spans(v.attention, t.attention);
    for (size_t i = 0; i < sizeof(writes)/sizeof(writes[0]); ++i) {
        for (size_t j = 0; j < i; ++j)
            if (!recurrent_detail::disjoint(writes[i], writes[j])) return false;
        for (const auto& read : reads)
            if (!recurrent_detail::disjoint(writes[i], read)) return false;
        // Block read zero is this layer's normalized_input producer, already
        // checked for exact identity and disjointness from every other write.
        for (size_t j = 1; j < block_reads.size(); ++j)
            if (block_reads[j].bytes && !recurrent_detail::disjoint(writes[i], block_reads[j])) return false;
        for (const auto& write : block_writes)
            if (!recurrent_detail::disjoint(writes[i], write)) return false;
    }
    for (const auto& write : block_writes)
        for (const auto& read : reads)
            if (!recurrent_detail::disjoint(write, read)) return false;
    return true;
}

// All results remain private, including the two candidate KV rows. The owner
// must fence, check the MoE invalid flag and sample actual target outputs before
// accepting a tail. Submission failure requires draining or quarantine.
inline hipError_t launch_attention_layer(const AttentionLayerViews& v, const AttentionLayerTables& t,
    unsigned maximum_blocks = 1024u, hipStream_t stream = nullptr) {
    if (!maximum_blocks || maximum_blocks > 4096u || !valid_attention_layer(v, t))
        return hipErrorInvalidValue;
    auto status = qrt_sm121_mtp::launch_residual_normalize(v.hidden, v.residual,
        v.input_norm_weights, t.attention.rsqrt, 2u, v.normalized_input, v.input_residual, stream);
    if (status != hipSuccess) return status;
    status = launch_attention_block(v.attention, t.attention, stream);
    if (status != hipSuccess) return status;
    status = qrt_sm121_mtp::launch_residual_normalize(v.attention.output, v.input_residual,
        v.post_norm_weights, t.attention.rsqrt, 2u, v.moe_input, v.output_residual, stream);
    if (status != hipSuccess) return status;
    return qrt_sm121_mtp::launch_moe(v.moe_input, v.moe_weights, t.moe, v.moe_workspace,
        v.moe_workspace_bytes, 2u, maximum_blocks, stream);
}
} // namespace qrt_sm121_q2
