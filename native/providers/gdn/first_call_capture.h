#ifndef QRT_FLA_FIRST_CALL_CAPTURE_H
#define QRT_FLA_FIRST_CALL_CAPTURE_H
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace qrt_fla_capture {
// Keep selection bounded and explicit; malformed text must not capture a
// different layer under a valid-looking completion record.
inline bool parse_call_index(const char* setting, unsigned& value) {
    value = 0;
    if (!setting) return true;
    if (!*setting) return false;
    for (const char* p = setting; *p; ++p) {
        if (*p < '0' || *p > '9' || value > 6u) return false;
        value = value * 10u + static_cast<unsigned>(*p - '0');
        if (value > 63u) return false;
    }
    return true;
}
// Diagnostic reads only. The output directory must be new, and a completion
// record is written only after the operator and every bounded read succeeds.
class FirstCall {
    enum class Phase { idle, complete, failed };
    Phase phase_ = Phase::idle;
    unsigned calls_ = 0;
public:
    static constexpr size_t chunk_bytes = 1u << 20u;
    const char* error = nullptr;
    template<class Copy, class Execute>
    bool run(const char* directory, unsigned tokens, const float* raw,
             const float* gates, const float* output, const float* state,
             Copy copy, Execute execute, unsigned selected_call = 0) {
        if (phase_ == Phase::failed) { error = "first-call capture previously failed"; return false; }
        if (!directory || !*directory || phase_ == Phase::complete) return execute();
        if (selected_call > 63u) {
            phase_ = Phase::failed; error = "capture call index exceeds 63"; return false;
        }
        const unsigned call_index = calls_++;
        if (call_index != selected_call) {
            if (execute()) return true;
            phase_ = Phase::failed; error = "operator failed before selected capture"; return false;
        }
        phase_ = Phase::failed;
        if (!tokens || tokens > 8192u || !raw || !gates || !output || !state) {
            error = "first-call capture requires 1..8192 tokens and complete surfaces"; return false;
        }
        try {
            const std::filesystem::path root(directory);
            if (!std::filesystem::create_directory(root)) {
                error = "first-call capture refuses an existing directory"; return false;
            }
            std::vector<unsigned char> host(chunk_bytes);
            auto save = [&](const char* name, const void* source, size_t bytes) {
                std::ofstream file(root / name, std::ios::binary);
                if (!file) { error = "first-call capture file open failed"; return false; }
                for (size_t offset = 0; offset < bytes; offset += chunk_bytes) {
                    const size_t count = std::min(chunk_bytes, bytes - offset);
                    if (!copy(host.data(), static_cast<const unsigned char*>(source) + offset, count)) {
                        error = "first-call capture device read failed"; return false;
                    }
                    file.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(count));
                    if (!file) { error = "first-call capture write failed"; return false; }
                }
                file.close();
                if (!file) { error = "first-call capture close failed"; return false; }
                return true;
            };
            if (!save("raw-f32.bin", raw, size_t(tokens) * 8192u * 4u) ||
                !save("gates-f32.bin", gates, size_t(tokens) * 64u * 4u)) return false;
            if (!execute()) { error = "first-call captured operator failed"; return false; }
            if (!save("output-f32.bin", output, size_t(tokens) * 4096u * 4u) ||
                !save("state-f32.bin", state, 32u * 128u * 128u * 4u)) return false;
            std::ofstream record(root / "capture.json");
            record << "{\"kind\":\"" << (call_index == 0 ? "first_native_gdn_call" : "selected_native_gdn_call")
                   << "\",\"call_index\":" << call_index << ",\"tokens\":" << tokens
                   << ",\"raw_rows\":8192,\"gate_rows\":64,\"output_rows\":4096,"
                      "\"state_elements\":524288,\"dtype\":\"float32\",\"layout\":\"token_rows\","
                      "\"copy_chunk_bytes\":1048576,\"complete\":true,\"inference_acceptance\":false}\n";
            record.close();
            if (!record) { error = "first-call completion record failed"; return false; }
            phase_ = Phase::complete;
            return true;
        } catch (...) {
            error = "first-call capture filesystem or host allocation failed"; return false;
        }
    }
};
}
#endif
