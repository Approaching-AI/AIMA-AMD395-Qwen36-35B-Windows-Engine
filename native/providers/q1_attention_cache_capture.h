#pragma once
#include "prefix_linear_capture.h"

namespace qrt_q1_attention_cache_capture {
constexpr size_t max_tokens = 263680u;
constexpr size_t max_score_stride = max_tokens + 1u;
constexpr size_t byte_limit = 576u << 20u;
constexpr size_t kv_row_bytes = 512u * sizeof(uint16_t);
constexpr size_t rope_bytes = 9216u * sizeof(float);
constexpr size_t context_bytes = 4096u * sizeof(float);
constexpr size_t segment_output_bytes = 16u * 16u * 256u * sizeof(float);
constexpr size_t segment_scalar_bytes = 16u * 16u * sizeof(float);

struct Plan {
    const char *directory = nullptr;
    unsigned layer = 0u, position = 0u;
    bool matches(unsigned actual_layer, size_t actual_position) const {
        return directory && layer == actual_layer && position == actual_position;
    }
};

inline bool parse(const char *directory, const char *layer, const char *position,
                  Plan &plan, std::string &error) {
    plan = {};
    if (!directory || !*directory) return true;
    using qrt_prefix_linear_capture::number;
    if (!number(layer, 39u, plan.layer) || plan.layer % 4u != 3u ||
        !number(position, unsigned(max_tokens - 1u), plan.position) || !plan.position) {
        error = "q1 attention capture requires one full-attention layer and one bounded decode position";
        return false;
    }
    plan.directory = directory;
    return true;
}

struct View {
    const void *prefix_k = nullptr, *prefix_v = nullptr;
    const void *tail_k = nullptr, *tail_v = nullptr;
    const float *rope = nullptr, *scores = nullptr, *context = nullptr;
    const float *segment_output = nullptr, *segment_max = nullptr, *segment_sum = nullptr;
    size_t prefix_tokens = 0u, tail_tokens = 0u, score_stride = 0u;
    size_t prefix_k_capacity_bytes = 0u, prefix_v_capacity_bytes = 0u;
    size_t tail_k_capacity_bytes = 0u, tail_v_capacity_bytes = 0u;
    bool segmented = false;
};

inline size_t surface_bytes(const View &v) {
    return (v.prefix_tokens + v.tail_tokens) * kv_row_bytes * 2u +
        16u * v.score_stride * sizeof(float) + rope_bytes + context_bytes +
        segment_output_bytes + 2u * segment_scalar_bytes;
}

inline bool validate(const View &v, size_t position, std::string &error) {
    // Check bounds before adding or multiplying caller-provided extents.
    if (!v.segmented || !v.prefix_tokens || !v.tail_tokens ||
        v.prefix_tokens >= max_tokens || v.tail_tokens > max_tokens - v.prefix_tokens ||
        position >= max_tokens || v.prefix_tokens + v.tail_tokens != position + 1u ||
        v.score_stride < position + 1u || v.score_stride > max_score_stride ||
        v.prefix_k_capacity_bytes < v.prefix_tokens * kv_row_bytes ||
        v.prefix_v_capacity_bytes < v.prefix_tokens * kv_row_bytes ||
        v.tail_k_capacity_bytes < v.tail_tokens * kv_row_bytes ||
        v.tail_v_capacity_bytes < v.tail_tokens * kv_row_bytes ||
        !v.prefix_k || !v.prefix_v || !v.tail_k || !v.tail_v || !v.rope || !v.scores ||
        !v.context || !v.segment_output || !v.segment_max || !v.segment_sum ||
        surface_bytes(v) > byte_limit) {
        error = "q1 attention capture has inconsistent logical KV, score, or segmented-output extents";
        return false;
    }
    return true;
}

// Observe one completed original segmented-attention invocation before the
// gate overwrites the score scratch. Never upload or replace any inference
// operand. Copy logical KV rows only, using at most 1 MiB of host staging.
template<class Synchronize, class Copy>
bool run(const Plan &plan, unsigned layer, size_t position, const View &v,
         bool &captured, Synchronize synchronize, Copy copy, std::string &error) {
    if (!plan.matches(layer, position) || captured) return true;
    if (!validate(v, position, error)) return false;
    try {
        if (!std::filesystem::create_directory(plan.directory)) {
            error = "q1 attention capture requires a new directory"; return false;
        }
        const auto started = qrt_prefix_linear_capture::Clock::now();
        if (!synchronize()) {
            error = "q1 attention capture stream completion failed"; return false;
        }
        const std::filesystem::path root(plan.directory);
        using qrt_prefix_linear_capture::save;
        const size_t prefix_bytes = v.prefix_tokens * kv_row_bytes;
        const size_t tail_bytes = v.tail_tokens * kv_row_bytes;
        if (!save(root / "prefix-k-bf16.bin", v.prefix_k, prefix_bytes, copy, started, error) ||
            !save(root / "prefix-v-bf16.bin", v.prefix_v, prefix_bytes, copy, started, error) ||
            !save(root / "tail-k-bf16.bin", v.tail_k, tail_bytes, copy, started, error) ||
            !save(root / "tail-v-bf16.bin", v.tail_v, tail_bytes, copy, started, error) ||
            !save(root / "rope-f32.bin", v.rope, rope_bytes, copy, started, error) ||
            !save(root / "scores-padded-f32.bin", v.scores, 16u * v.score_stride * sizeof(float), copy, started, error) ||
            !save(root / "segment-output-f32.bin", v.segment_output, segment_output_bytes, copy, started, error) ||
            !save(root / "segment-max-f32.bin", v.segment_max, segment_scalar_bytes, copy, started, error) ||
            !save(root / "segment-sum-f32.bin", v.segment_sum, segment_scalar_bytes, copy, started, error) ||
            !save(root / "context-f32.bin", v.context, context_bytes, copy, started, error)) return false;
        std::ofstream record(root / "capture.json");
        record << "{\"kind\":\"native_q1_logical_attention_cache\",\"layer\":" << layer
               << ",\"position\":" << position << ",\"prefix_tokens\":" << v.prefix_tokens
               << ",\"tail_tokens_including_current\":" << v.tail_tokens
               << ",\"total_tokens\":" << position + 1u << ",\"score_stride\":" << v.score_stride
               << ",\"surface_bytes\":" << surface_bytes(v)
               << ",\"q_heads\":16,\"kv_heads\":2,\"head_dim\":256,\"segments\":16,"
                  "\"kv_layout\":\"token-head-channel\",\"score_layout\":\"head-stride\","
                  "\"rope_layout\":\"q4096-gate4096-k512-v512\",\"files\":10,"
                  "\"complete\":true,\"copy_chunk_bytes\":1048576,\"copy_deadline_seconds\":90,"
                  "\"diagnostic_only\":true,\"inference_acceptance\":false}\n";
        record.close();
        if (!record) { error = "q1 attention capture completion record failed"; return false; }
        captured = true;
        return true;
    } catch (...) {
        error = "q1 attention capture filesystem or allocation failure"; return false;
    }
}
}
