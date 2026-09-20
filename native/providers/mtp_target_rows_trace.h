#pragma once
#include <filesystem>
#include <fstream>
#include <string>
#include "mtp_target_rows.h"

namespace qrt_mtp_target_rows {
// Optional numerical evidence for the actual provider handoff. The caller
// chooses a fresh path prefix in an existing directory. Metadata is written
// last; partial output from a failed write remains unaccepted and is never
// overwritten by a retry. This observer does not run or enable an MTP draft.
inline bool write_trace(const PrefillRows &batch, const std::string &prefix,
                        std::string *failure) {
    const auto fail = [&](const char *message) {
        if (failure) *failure = message;
        return false;
    };
    if (!batch.published() || prefix.empty() || !failure)
        return fail("MTP target-row trace requires a published batch and a fresh path prefix");
    const std::string hidden_path = prefix + ".hidden.bf16.bin";
    const std::string tokens_path = prefix + ".shifted.u32.bin";
    const std::string metadata_path = prefix + ".json";
    for (const auto *path : {&hidden_path, &tokens_path, &metadata_path}) {
        std::error_code error;
        const bool exists = std::filesystem::exists(*path, error);
        if (error || exists) return fail("MTP target-row trace path exists or cannot be checked");
    }
    const auto write = [&](const std::string &path, const void *data, size_t bytes) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) return false;
        file.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
        file.close();
        return bool(file);
    };
    if (!write(hidden_path, batch.hidden().data(), batch.hidden().size() * sizeof(uint16_t)) ||
        !write(tokens_path, batch.shifted_tokens().data(), batch.shifted_tokens().size() * sizeof(uint32_t)))
        return fail("MTP target-row trace write or close failed");
    const std::string metadata = "{\n  \"schema\": \"qrt-mtp-target-rows-v1\",\n"
        "  \"source\": \"actual_completed_target_final_norm_and_sample\",\n"
        "  \"first_position\": " + std::to_string(batch.first_position()) + ",\n"
        "  \"prompt_tokens\": " + std::to_string(batch.prompt_tokens()) + ",\n"
        "  \"rows\": " + std::to_string(batch.rows()) + ",\n"
        "  \"hidden_width\": " + std::to_string(hidden_width) + ",\n"
        "  \"sampled_token\": " + std::to_string(batch.sampled_token()) + ",\n"
        "  \"discarded_prefill\": " + (batch.discarded_prefill() ? "true" : "false") + ",\n"
        "  \"hidden_bytes\": " + std::to_string(batch.hidden().size() * sizeof(uint16_t)) + ",\n"
        "  \"shifted_token_bytes\": " + std::to_string(batch.shifted_tokens().size() * sizeof(uint32_t)) + ",\n"
        "  \"mtp_inference_enabled\": false,\n  \"numerical_acceptance_claimed\": false\n}\n";
    if (!write(metadata_path, metadata.data(), metadata.size()))
        return fail("MTP target-row trace metadata write or close failed");
    return true;
}
} // namespace qrt_mtp_target_rows
