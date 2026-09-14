#pragma once
#include <cstddef>
#include <cstdint>

namespace qrt_prepared_decoded_qk {
constexpr unsigned maximum_tokens = 8192u;
constexpr size_t query_words = size_t(maximum_tokens) * 16u * 256u;
constexpr size_t key_words = size_t(maximum_tokens) * 2u * 256u;
constexpr size_t query_flag_words = size_t(maximum_tokens) * 16u;
constexpr size_t key_flag_words = size_t(maximum_tokens) * 2u;
constexpr size_t workspace_words = query_words + key_words + query_flag_words + key_flag_words;
struct Workspace { uint32_t* words = nullptr; unsigned tokens = 0u; };
inline bool valid(const Workspace& workspace) {
    return workspace.words && workspace.tokens && workspace.tokens <= maximum_tokens;
}
} // namespace qrt_prepared_decoded_qk
