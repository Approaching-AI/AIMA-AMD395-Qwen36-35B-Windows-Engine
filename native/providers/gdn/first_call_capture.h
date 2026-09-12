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
struct Window {
    unsigned first_position = 0, tokens = 0;
};
inline bool parse_window(const char* first, const char* count, Window& window) {
    window = {};
    if (!first && !count) return true;
    auto number = [](const char* text, unsigned maximum, unsigned& value) {
        value = 0;
        if (!text || !*text) return false;
        for (const char* p = text; *p; ++p) {
            if (*p < '0' || *p > '9' || value > maximum / 10u) return false;
            value = value * 10u + unsigned(*p - '0');
            if (value > maximum) return false;
        }
        return true;
    };
    return number(first, 65536u, window.first_position) &&
        number(count, 8192u, window.tokens) && window.first_position && window.tokens &&
        window.first_position % 1024u == 0 && window.tokens % 1024u == 0 &&
        window.first_position + window.tokens <= 65536u;
}
// Diagnostic reads only. The output directory must be new, and a completion
// record is written only after the operator and every bounded read succeeds.
class FirstCall {
    enum class Phase { idle, complete, failed };
    Phase phase_ = Phase::idle;
    unsigned calls_ = 0;
    bool active_ = false, initial_saved_ = false, final_saved_ = false;
    Window window_{};
    std::filesystem::path root_;
    template<class Copy>
    bool save(const char* name, const void* source, size_t bytes, Copy copy) {
        std::vector<unsigned char> host(chunk_bytes);
        std::ofstream file(root_ / name, std::ios::binary);
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
    }
public:
    static constexpr size_t chunk_bytes = 1u << 20u;
    const char* error = nullptr;
    // Called between the existing segments, before any reset or kernel for
    // that segment. This only reads the original unrounded recurrent state.
    template<class Copy>
    bool before_segment(unsigned position, const float* state, Copy copy) {
        if (!active_ || !window_.tokens) return true;
        if (position == window_.first_position && !initial_saved_) {
            if (!save("initial-state-f32.bin", state, 32u * 128u * 128u * 4u, copy)) return false;
            initial_saved_ = true;
        }
        if (position == window_.first_position + window_.tokens && !final_saved_) {
            if (!save("state-f32.bin", state, 32u * 128u * 128u * 4u, copy)) return false;
            final_saved_ = true;
        }
        return true;
    }
    template<class Copy, class Execute>
    bool run(const char* directory, unsigned tokens, const float* raw,
             const float* gates, const float* output, const float* state,
             Copy copy, Execute execute, unsigned selected_call = 0, Window window = {}) {
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
        const unsigned observed = window.tokens ? window.tokens : tokens;
        if (!tokens || tokens > 65536u || !observed || observed > 8192u ||
            (window.tokens && (!window.first_position || window.first_position % 1024u ||
                window.tokens % 1024u || window.first_position > tokens ||
                window.tokens > tokens - window.first_position)) ||
            (!window.tokens && window.first_position) || !raw || !gates || !output || !state) {
            error = "capture requires complete surfaces and at most 8192 inputs at original segment boundaries"; return false;
        }
        try {
            root_ = std::filesystem::path(directory);
            if (!std::filesystem::create_directory(root_)) {
                error = "first-call capture refuses an existing directory"; return false;
            }
            window_ = window;
            if (!save("raw-f32.bin", raw + size_t(window.first_position) * 8192u,
                      size_t(observed) * 8192u * 4u, copy) ||
                !save("gates-f32.bin", gates + size_t(window.first_position) * 64u,
                      size_t(observed) * 64u * 4u, copy)) return false;
            active_ = true;
            struct Reset { bool& active; ~Reset() { active = false; } } reset{active_};
            if (!execute()) { error = "first-call captured operator failed"; return false; }
            if (window.tokens && !initial_saved_) {
                error = "original initial state boundary was not observed"; return false;
            }
            if (!window.tokens || window.first_position + observed == tokens) {
                if (!save("state-f32.bin", state, 32u * 128u * 128u * 4u, copy)) return false;
                final_saved_ = true;
            }
            if (!final_saved_) { error = "original final state boundary was not observed"; return false; }
            if (!save("output-f32.bin", output + size_t(window.first_position) * 4096u,
                      size_t(observed) * 4096u * 4u, copy)) return false;
            std::ofstream record(root_ / "capture.json");
            record << "{\"kind\":\"" << (call_index == 0 ? "first_native_gdn_call" : "selected_native_gdn_call")
                   << "\",\"call_index\":" << call_index << ",\"tokens\":" << tokens
                   << ",\"first_position\":" << window.first_position << ",\"captured_tokens\":" << observed
                   << ",\"state_tokens\":" << window.first_position + observed
                   << ",\"initial_state_captured\":" << (initial_saved_ ? "true" : "false")
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
