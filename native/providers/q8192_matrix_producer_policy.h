#pragma once

#include <cstring>

namespace qrt_q8192_matrix_producer {
inline bool eligible(unsigned rows, unsigned k, unsigned tokens, bool output_f32) {
    return output_f32 && tokens == 8192u &&
        ((k == 2048u && (rows == 8192u || rows == 4096u || rows == 9216u)) ||
         (k == 4096u && rows == 2048u));
}

// The measured native algorithm is an opt-in for four complete product shapes.
// Explicit-index calls, BF16 outputs and other token counts keep their policy.
inline bool resolve(const char* setting, unsigned rows, unsigned k,
                    unsigned tokens, bool output_f32, unsigned* choice) {
    if (!choice) return false;
    *choice = 0u;
    if (setting && *setting && std::strcmp(setting, "0") && std::strcmp(setting, "4"))
        return false;
    if (setting && std::strcmp(setting, "4") == 0 && eligible(rows, k, tokens, output_f32))
        *choice = 4u;
    return true;
}
} // namespace qrt_q8192_matrix_producer
