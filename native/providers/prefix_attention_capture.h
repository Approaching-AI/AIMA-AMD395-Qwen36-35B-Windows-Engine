#pragma once
#include "prefix_linear_capture.h"

namespace qrt_prefix_attention_capture {
struct Plan {
    const char *directory = nullptr;
    unsigned layer = 0, prefix = 0, tokens = 0;
    bool matches(unsigned actual_layer, unsigned actual_prefix, unsigned actual_tokens) const {
        return directory && layer == actual_layer && prefix == actual_prefix && tokens == actual_tokens;
    }
};

constexpr size_t byte_limit = 512u * 1024u * 1024u;
inline size_t surface_bytes(unsigned prefix, unsigned tokens) {
    return size_t(prefix) * 2048u + size_t(tokens) * (8192u + 2048u + 16384u);
}

inline bool parse(const char *directory, const char *layer, const char *prefix,
                  const char *tokens, Plan &plan, std::string &error) {
    plan = {};
    if (!directory || !*directory) return true;
    using qrt_prefix_linear_capture::number;
    if (!number(layer, 39u, plan.layer) || plan.layer % 4u != 3u ||
        !number(prefix, 131072u, plan.prefix) || plan.prefix < 8192u || plan.prefix % 8192u ||
        !number(tokens, 8192u, plan.tokens) || (plan.tokens != 1024u && plan.tokens != 8192u) ||
        surface_bytes(plan.prefix, plan.tokens) > byte_limit) {
        error = "prefix attention capture requires one full-attention layer and a bounded aligned original transaction";
        return false;
    }
    plan.directory = directory;
    return true;
}

inline bool environment(Plan &plan, std::string &error) {
    return parse(std::getenv("QRT_QWEN36_PREFIX_ATTENTION_CAPTURE_DIR"),
        std::getenv("QRT_QWEN36_PREFIX_ATTENTION_CAPTURE_LAYER"),
        std::getenv("QRT_QWEN36_PREFIX_ATTENTION_CAPTURE_POSITION"),
        std::getenv("QRT_QWEN36_PREFIX_ATTENTION_CAPTURE_TOKENS"), plan, error);
}

// Read the original BF16 inputs and FP32 context around exactly one unmodified
// provider invocation. Neither the resident KV nor the actual tail is replaced.
template<class Copy, class Execute>
bool run(const Plan &plan, unsigned layer, unsigned prefix, unsigned tokens,
         const uint16_t *q, const uint16_t *prefix_k, const uint16_t *prefix_v,
         const uint16_t *tail_k, const uint16_t *tail_v, const float *context,
         Copy copy, Execute execute, std::string &error) {
    if (!plan.matches(layer, prefix, tokens)) return execute();
    try {
        if (!q || !prefix_k || !prefix_v || !tail_k || !tail_v || !context ||
            !std::filesystem::create_directory(plan.directory)) {
            error = "prefix attention capture requires complete surfaces and a new directory"; return false;
        }
        const auto started = qrt_prefix_linear_capture::Clock::now();
        const std::filesystem::path root(plan.directory);
        using qrt_prefix_linear_capture::save;
        if (!save(root / "q-bf16.bin", q, size_t(tokens) * 4096u * 2u, copy, started, error) ||
            !save(root / "prefix-k-bf16.bin", prefix_k, size_t(prefix) * 512u * 2u, copy, started, error) ||
            !save(root / "prefix-v-bf16.bin", prefix_v, size_t(prefix) * 512u * 2u, copy, started, error) ||
            !save(root / "tail-k-bf16.bin", tail_k, size_t(tokens) * 512u * 2u, copy, started, error) ||
            !save(root / "tail-v-bf16.bin", tail_v, size_t(tokens) * 512u * 2u, copy, started, error)) return false;
        if (!execute()) { error = "original captured suffix attention launch failed"; return false; }
        if (!save(root / "context-f32.bin", context, size_t(tokens) * 4096u * 4u, copy, started, error)) return false;
        std::ofstream record(root / "capture.json");
        record << "{\"kind\":\"original_prefix_attention_window\",\"layer\":" << layer
               << ",\"first_position\":" << prefix << ",\"tokens\":" << tokens
               << ",\"q_heads\":16,\"kv_heads\":2,\"head_dim\":256,"
                  "\"input_dtype\":\"bf16\",\"context_dtype\":\"f32\","
                  "\"layout\":\"token-head-channel\",\"surface_bytes\":" << surface_bytes(prefix, tokens)
               << ",\"complete\":true,\"original_launch_count\":1,\"copy_chunk_bytes\":1048576,"
                  "\"diagnostic_only\":true,\"inference_acceptance\":false}\n";
        record.close();
        if (!record) { error = "prefix attention capture completion record failed"; return false; }
        return true;
    } catch (...) {
        error = "prefix attention capture filesystem or allocation failure"; return false;
    }
}
}
