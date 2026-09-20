#pragma once
#include "sm121_q2_cache_view.h"
#include <array>

namespace qrt_sm121_q2 {
struct AttentionBlockViews {
    const uint16_t* normalized_input = nullptr; // [2,2048]
    const uint16_t* q_weights = nullptr;        // [8192,2048], interleaved Q/gate.
    const uint16_t* k_weights = nullptr;        // [512,2048]
    const uint16_t* v_weights = nullptr;        // [512,2048]
    const uint16_t* output_weights = nullptr;   // [2048,4096]
    const uint16_t* q_norm_weights = nullptr;   // [256], BF16 delta.
    const uint16_t* k_norm_weights = nullptr;   // [256], BF16 delta.
    CacheView cache;                           // Readonly prefix/decode planes plus private tail.
    unsigned first_position = 0;               // Exactly the committed history length.
    uint16_t* q_projected = nullptr;            // [2,8192]
    uint16_t* kv_projected = nullptr;           // [2,K512+V512]
    uint16_t* q_norm = nullptr;                 // [2,4096]
    uint16_t* k_norm = nullptr;                 // [2,512]
    uint16_t* queries = nullptr;                // [2,4096], original RoPE.
    uint16_t* gates = nullptr;                  // [2,4096]
    uint16_t* staged_kv = nullptr;              // [2,K512+V512], private rotated K/raw V.
    float* scores = nullptr;                   // [2,16,score_stride]
    unsigned score_stride = 0;
    float* float_context = nullptr;            // [2,4096]
    uint16_t* context = nullptr;                // [2,4096]
    uint16_t* gated = nullptr;                  // [2,4096]
    uint16_t* output = nullptr;                 // [2,2048]
};
struct AttentionBlockTables {
    const unsigned char* rsqrt = nullptr;
    const unsigned char* exp2 = nullptr;
    const unsigned char* reciprocal = nullptr;
    const uint16_t* rope = nullptr;             // [rope_rows,64]
    unsigned rope_rows = 0;
    const uint16_t* sigmoid = nullptr;          // [65536], original BF16 sigmoid.
};

namespace attention_block_detail {
using recurrent_detail::Span;
inline std::array<Span,12> write_spans(const AttentionBlockViews& v) {
    return {{
        {v.q_projected, 2u*8192u*2u, 2u}, {v.kv_projected, 2u*1024u*2u, 2u},
        {v.q_norm, 2u*4096u*2u, 2u}, {v.k_norm, 2u*512u*2u, 2u},
        {v.queries, 2u*4096u*2u, 2u}, {v.gates, 2u*4096u*2u, 2u},
        {v.staged_kv, 2u*1024u*2u, 2u}, {v.scores, size_t(2u)*16u*v.score_stride*4u, 4u},
        {v.float_context, 2u*4096u*4u, 4u}, {v.context, 2u*4096u*2u, 2u},
        {v.gated, 2u*4096u*2u, 2u}, {v.output, 2u*2048u*2u, 2u}
    }};
}
inline std::array<Span,16> read_spans(const AttentionBlockViews& v, const AttentionBlockTables& t) {
    const auto history = cache_view_detail::history_spans(v.cache);
    return {{
        {v.normalized_input, 8192u, 2u}, {v.q_weights, 8192u*2048u*2u, 2u},
        {v.k_weights, 512u*2048u*2u, 2u}, {v.v_weights, 512u*2048u*2u, 2u},
        {v.output_weights, 2048u*4096u*2u, 2u}, {v.q_norm_weights, 512u, 2u},
        {v.k_norm_weights, 512u, 2u}, history[0], history[1], history[2], history[3],
        {t.rsqrt, qrt_sm121_rsqrt::table_bytes, 1u}, {t.exp2, qrt_sm121_exp2::table_bytes, 1u},
        {t.reciprocal, qrt_sm121_attention_rcp::table_bytes, 1u},
        {t.rope, size_t(t.rope_rows)*64u*2u, 2u}, {t.sigmoid, 65536u*2u, 2u}
    }};
}
} // namespace attention_block_detail

inline bool valid_attention_block(const AttentionBlockViews& v, const AttentionBlockTables& t) {
    if (!valid_cache_view(v.cache) || v.cache.staged != v.staged_kv ||
        v.first_position != v.cache.committed_tokens() || v.first_position > target_context_limit - 2u ||
        t.rope_rows > target_context_limit || t.rope_rows < v.first_position + 2u ||
        v.score_stride < v.first_position + 2u || v.score_stride > target_context_limit ||
        v.score_stride % 32u) return false;
    const auto writes = attention_block_detail::write_spans(v);
    const auto reads = attention_block_detail::read_spans(v, t);
    for (size_t i = 0; i < writes.size(); ++i) {
        for (size_t j = 0; j < i; ++j)
            if (!recurrent_detail::disjoint(writes[i], writes[j])) return false;
        for (const auto& read : reads)
            if (read.bytes && !recurrent_detail::disjoint(writes[i], read)) return false;
    }
    return true;
}

struct AttentionSelection {
    const uint16_t* key_values = nullptr;
    unsigned first_position = 0;
    unsigned rows = 0;
};
// Borrowed private tail after caller-verified completion; no resident write.
inline AttentionSelection accepted_attention(const AttentionBlockViews& v, unsigned rows) {
    return v.staged_kv && rows >= 1u && rows <= 2u && v.first_position <= target_context_limit - 2u
        ? AttentionSelection{v.staged_kv, v.first_position, rows} : AttentionSelection{};
}
} // namespace qrt_sm121_q2
