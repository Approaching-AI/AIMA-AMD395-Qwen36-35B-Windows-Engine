#pragma once
#include <cstddef>
#include <cstdint>
#include "mtp_draft_schedule.h"

namespace qrt_mtp_prompt_inputs {
// The original proposer shifts tokens within each target prefill batch.
// A discarded partial-prefill sample uses the request's last known prompt
// token at the batch tail. A final prefill uses its actual sampled token.
inline bool shift(const uint32_t* prompt, size_t prompt_tokens,
    size_t first_position, size_t rows, bool discarded_prefill,
    uint32_t sampled_token, uint32_t* output) {
    constexpr uint32_t vocabulary = 248320u;
    if (!prompt || !output || !prompt_tokens ||
        prompt_tokens > qrt_mtp_draft_schedule::reference_drafter_limit ||
        !rows || rows > 8192u || first_position >= prompt_tokens ||
        rows > prompt_tokens - first_position ||
        discarded_prefill != (first_position + rows < prompt_tokens)) return false;
    const uint32_t final_token = discarded_prefill ? prompt[prompt_tokens - 1u] : sampled_token;
    if (final_token >= vocabulary) return false;
    for (size_t row = 0; row + 1u < rows; ++row)
        if (prompt[first_position + row + 1u] >= vocabulary) return false;
    for (size_t row = 0; row + 1u < rows; ++row)
        output[row] = prompt[first_position + row + 1u];
    output[rows - 1u] = final_token;
    return true;
}
} // namespace qrt_mtp_prompt_inputs
