#pragma once
#include "sm121_q2_linear_block.h"
#include "sm121_mtp_residual.h"
#include "sm121_mtp_moe.h"

namespace qrt_sm121_q2 {
template<class Element> struct LinearLayerViews {
    const uint16_t* hidden = nullptr;          // [2,2048], previous MoE or embedding.
    const uint16_t* residual = nullptr;        // [2,2048], zero for the embedding layer.
    const uint16_t* input_norm_weights = nullptr; // [2048], original BF16 delta.
    const uint16_t* post_norm_weights = nullptr;  // [2048], original BF16 delta.
    uint16_t* normalized_input = nullptr;      // [2,2048], feeds linear.normalized_input.
    uint16_t* input_residual = nullptr;        // [2,2048], rounded first residual sum.
    uint16_t* moe_input = nullptr;             // [2,2048], normalized attention residual.
    uint16_t* output_residual = nullptr;       // [2,2048], paired with MoE output.
    LinearBlockViews<Element> linear;
    qrt_sm121_mtp::MoeWeights moe_weights;
    void* moe_workspace = nullptr;
    size_t moe_workspace_bytes = 0;
};
struct LinearLayerTables {
    LinearBlockTables linear;
    qrt_sm121_mtp::MoeTables moe;
};

// Validate the complete producer graph before its first enqueue. In particular,
// a downstream MoE workspace cannot overwrite an upstream cache, table or norm.
template<class Element> inline bool valid_linear_layer(const LinearLayerViews<Element>& v,
                                                       const LinearLayerTables& t) {
    qrt_sm121_mtp::MoeBuffers moe;
    if (v.linear.normalized_input != v.normalized_input ||
        !valid_linear_block(v.linear, t.linear) ||
        !qrt_sm121_mtp::bind_moe_buffers(v.moe_workspace, v.moe_workspace_bytes, 2u, &moe))
        return false;
    using recurrent_detail::Span;
    const auto& l = v.linear;
    const Span writes[] = {
        {v.normalized_input, 8192u, 2u}, {v.input_residual, 8192u, 2u},
        {v.moe_input, 8192u, 2u}, {v.output_residual, 8192u, 2u},
        {v.moe_workspace, qrt_sm121_mtp::moe_workspace_bytes(2u), 256u},
        {l.qkv, 2u*8192u*2u, 2u}, {l.z, 2u*4096u*2u, 2u},
        {l.a, 2u*32u*2u, 2u}, {l.b, 2u*32u*2u, 2u},
        {l.gated, 2u*4096u*2u, 2u}, {l.output, 8192u, 2u},
        {l.convolution.staged_rings, 2u*ring_elements*sizeof(Element), alignof(Element)},
        {l.convolution.staged_convolution, 2u*8192u*2u, 2u},
        {l.recurrent.staged_states, staged_state_bytes, alignof(float)},
        {l.recurrent.staged_core, staged_core_bytes, 2u}
    };
    const Span reads[] = {
        {v.hidden, 8192u, 2u}, {v.residual, 8192u, 2u},
        {v.input_norm_weights, 4096u, 2u}, {v.post_norm_weights, 4096u, 2u},
        {l.qkv_weights, 8192u*2048u*2u, 2u}, {l.z_weights, 4096u*2048u*2u, 2u},
        {l.a_weights, 32u*2048u*2u, 2u}, {l.b_weights, 32u*2048u*2u, 2u},
        {l.output_weights, 2048u*4096u*2u, 2u}, {l.norm_weights, 128u*2u, 2u},
        {l.convolution.initial_ring, ring_elements*sizeof(Element), alignof(Element)},
        {l.convolution.weights, 8192u*4u*2u, 2u},
        {l.convolution.silu, qrt_sm121_silu::table_bytes, 1u},
        {l.recurrent.initial_state, state_elements*sizeof(float), alignof(float)},
        {t.linear.recurrent.g, 32u*65536u*sizeof(float), alignof(float)},
        {t.linear.recurrent.beta, 65536u*sizeof(float), alignof(float)},
        {t.linear.recurrent.exp2, qrt_sm121_exp2::table_bytes, 1u},
        {t.linear.recurrent.rsqrt, qrt_sm121_rsqrt::table_bytes, 1u},
        {t.linear.gated_silu, 65536u*sizeof(float), alignof(float)},
        {v.moe_weights.router, 256u*2048u*2u, 2u},
        {v.moe_weights.shared_gate, 2048u*2u, 2u},
        {v.moe_weights.shared_gate_up, 1024u*2048u*2u, 2u},
        {v.moe_weights.shared_down, 2048u*512u*2u, 2u},
        {v.moe_weights.routed_gate_up, size_t(256u)*1024u*2048u*2u, 2u},
        {v.moe_weights.routed_down, size_t(256u)*2048u*512u*2u, 2u},
        {t.moe.silu, 65536u*2u, 2u}, {t.moe.sigmoid, 65536u*2u, 2u},
        {t.moe.router_exp_fraction, 8388608u*4u, alignof(uint32_t)}
    };
    for (size_t i = 0; i < sizeof(writes)/sizeof(writes[0]); ++i) {
        for (size_t j = 0; j < i; ++j)
            if (!recurrent_detail::disjoint(writes[i], writes[j])) return false;
        for (const auto& read : reads)
            if (!recurrent_detail::disjoint(writes[i], read)) return false;
    }
    return true;
}

// Complete private target layer, with original target arithmetic. Successful
// enqueue leaves every result private: the owner must fence, check the MoE
// invalid flag, sample actual target logits and commit only accepted outcomes.
// Partial failures require draining or quarantining all borrowed storage.
template<class Element> inline hipError_t launch_linear_layer(const LinearLayerViews<Element>& v,
    const LinearLayerTables& t, unsigned maximum_blocks = 1024u, hipStream_t stream = nullptr) {
    if (!maximum_blocks || maximum_blocks > 4096u || !valid_linear_layer(v, t))
        return hipErrorInvalidValue;
    auto status = qrt_sm121_mtp::launch_residual_normalize(v.hidden, v.residual,
        v.input_norm_weights, t.linear.recurrent.rsqrt, 2u, v.normalized_input,
        v.input_residual, stream);
    if (status != hipSuccess) return status;
    status = launch_linear_block(v.linear, t.linear, stream);
    if (status != hipSuccess) return status;
    status = qrt_sm121_mtp::launch_residual_normalize(v.linear.output, v.input_residual,
        v.post_norm_weights, t.linear.recurrent.rsqrt, 2u, v.moe_input, v.output_residual, stream);
    if (status != hipSuccess) return status;
    return qrt_sm121_mtp::launch_moe(v.moe_input, v.moe_weights, t.moe, v.moe_workspace,
        v.moe_workspace_bytes, 2u, maximum_blocks, stream);
}
} // namespace qrt_sm121_q2
