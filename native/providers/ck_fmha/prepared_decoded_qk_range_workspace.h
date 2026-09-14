#pragma once
#include "prepared_decoded_qk_workspace.h"
#include "../sm121_attention_capacity.h"

namespace qrt_prepared_decoded_qk_range {
constexpr unsigned maximum_queries = qrt_prepared_decoded_qk::maximum_tokens;
constexpr unsigned maximum_keys = qrt_sm121_attention_capacity::kTokens;
constexpr size_t query_words = size_t(maximum_queries) * 16u * 256u;
constexpr size_t query_flag_words = size_t(maximum_queries) * 16u;
constexpr size_t key_words(unsigned capacity) { return size_t(capacity) * 2u * 256u; }
constexpr size_t key_flag_words(unsigned capacity) { return size_t(capacity) * 2u; }
constexpr size_t workspace_words(unsigned capacity) {
    return capacity && capacity <= maximum_keys
        ? query_words + key_words(capacity) + query_flag_words + key_flag_words(capacity) : 0u;
}
struct Workspace {
    uint32_t* words = nullptr;
    size_t word_count = 0u;
    unsigned key_capacity = 0u;
    unsigned query_start = 0u;
    unsigned query_count = 0u;
    unsigned key_tokens = 0u;
};
inline bool valid(const Workspace& workspace) {
    const size_t required = workspace_words(workspace.key_capacity);
    return workspace.words && required && workspace.word_count >= required &&
        workspace.query_count && workspace.query_count <= maximum_queries &&
        workspace.key_tokens && workspace.key_tokens <= workspace.key_capacity &&
        workspace.query_start < workspace.key_tokens &&
        workspace.query_count <= workspace.key_tokens - workspace.query_start;
}
} // namespace qrt_prepared_decoded_qk_range
