#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace qrt_prefix_linear_capture {
struct Plan {
    const char *directory = nullptr;
    unsigned layer = 0, prefix = 0, tokens = 0;
    bool matches(unsigned actual_layer, unsigned actual_prefix, unsigned actual_tokens) const {
        return directory && layer == actual_layer && prefix == actual_prefix && tokens == actual_tokens;
    }
};

inline bool number(const char *text, unsigned maximum, unsigned &out) {
    out = 0;
    if (!text || !*text) return false;
    for (const char *p = text; *p; ++p) {
        if (*p < '0' || *p > '9' || out > maximum / 10u) return false;
        out = out * 10u + unsigned(*p - '0');
        if (out > maximum) return false;
    }
    return true;
}

inline bool parse(const char *directory, const char *layer, const char *prefix,
                  const char *tokens, Plan &plan, std::string &error) {
    plan = {};
    if (!directory || !*directory) return true;
    if (!number(layer, 38u, plan.layer) || plan.layer % 4u == 3u ||
        !number(prefix, 262144u, plan.prefix) || plan.prefix < 8192u || plan.prefix % 8192u ||
        !number(tokens, 8192u, plan.tokens) || (plan.tokens != 1024u && plan.tokens != 8192u) ||
        plan.prefix > 263168u - plan.tokens) {
        error = "prefix linear capture requires one linear layer and an actual aligned 1024/8192-input transaction";
        return false;
    }
    plan.directory = directory;
    return true;
}

inline bool environment(Plan &plan, std::string &error) {
    return parse(std::getenv("QRT_QWEN36_PREFIX_LINEAR_CAPTURE_DIR"),
        std::getenv("QRT_QWEN36_PREFIX_LINEAR_CAPTURE_LAYER"),
        std::getenv("QRT_QWEN36_PREFIX_LINEAR_CAPTURE_POSITION"),
        std::getenv("QRT_QWEN36_PREFIX_LINEAR_CAPTURE_TOKENS"), plan, error);
}

constexpr size_t copy_chunk_bytes = 1u << 20u;
using Clock = std::chrono::steady_clock;

template<class Copy>
bool save(const std::filesystem::path &path, const void *source, size_t bytes,
          Copy copy, Clock::time_point started, std::string &error) {
    if (!source || !bytes || std::filesystem::exists(path)) {
        error = "prefix linear capture refuses a missing surface or existing file"; return false;
    }
    std::vector<unsigned char> host((std::min)(copy_chunk_bytes, bytes));
    std::ofstream file(path, std::ios::binary | std::ios::out);
    if (!file) { error = "prefix linear capture file open failed"; return false; }
    for (size_t offset = 0; offset < bytes; offset += host.size()) {
        const size_t count = (std::min)(host.size(), bytes - offset);
        if (std::chrono::duration<double>(Clock::now() - started).count() > 90.0) {
            error = "prefix linear capture exceeded 90 seconds"; return false;
        }
        if (!copy(host.data(), static_cast<const unsigned char *>(source) + offset, count)) {
            error = "prefix linear capture device read failed"; return false;
        }
        file.write(reinterpret_cast<const char *>(host.data()), static_cast<std::streamsize>(count));
        if (!file) { error = "prefix linear capture write failed"; return false; }
    }
    file.close();
    if (!file) { error = "prefix linear capture close failed"; return false; }
    return true;
}

// The original seeded launch executes once. Inputs and both state boundaries
// are read without casts; key-major state interpretation is left to analysis.
template<class Copy, class Execute>
bool run(const Plan &plan, unsigned layer, unsigned prefix, unsigned tokens,
         const float *raw, const float *gates, const float *output, const float *state,
         bool key_major, Copy copy, Execute execute, std::string &error) {
    if (!plan.matches(layer, prefix, tokens)) return execute();
    try {
        if (!raw || !gates || !output || !state || !std::filesystem::create_directory(plan.directory)) {
            error = "prefix linear capture requires complete surfaces and a new directory"; return false;
        }
        const auto started = Clock::now();
        const std::filesystem::path root(plan.directory);
        if (!save(root / "raw-f32.bin", raw, size_t(tokens) * 8192u * 4u, copy, started, error) ||
            !save(root / "gates-f32.bin", gates, size_t(tokens) * 64u * 4u, copy, started, error) ||
            !save(root / "initial-state-f32.bin", state, 524288u * 4u, copy, started, error)) return false;
        if (!execute()) { error = "original captured seeded GDN launch failed"; return false; }
        if (!save(root / "output-f32.bin", output, size_t(tokens) * 4096u * 4u, copy, started, error) ||
            !save(root / "final-state-f32.bin", state, 524288u * 4u, copy, started, error)) return false;
        std::ofstream record(root / "capture.json");
        record << "{\"kind\":\"original_prefix_linear_window\",\"layer\":" << layer
               << ",\"first_position\":" << prefix << ",\"tokens\":" << tokens
               << ",\"state_layout\":\"" << (key_major ? "key-major" : "value-major")
               << "\",\"dtype\":\"f32\",\"complete\":true,\"original_launch_count\":1,"
                  "\"copy_chunk_bytes\":1048576,\"diagnostic_only\":true,\"inference_acceptance\":false}\n";
        record.close();
        if (!record) { error = "prefix linear capture completion record failed"; return false; }
        return true;
    } catch (...) {
        error = "prefix linear capture filesystem or allocation failure"; return false;
    }
}

template<class Copy>
bool row(const Plan &plan, unsigned layer, unsigned prefix, unsigned tokens,
         const char *surface, const void *source, size_t width, size_t element_bytes,
         Copy copy, std::string &error) {
    if (!plan.matches(layer, prefix, tokens)) return true;
    // Device preparation intentionally has no normalized legacy postconv
    // surface. Capture only values produced by the existing computation.
    if (surface && std::strcmp(surface, "postconv") == 0 && !source) return true;
    if (!surface || !*surface || !source || !width || width > 8192u ||
        (element_bytes != 2u && element_bytes != 4u)) {
        error = "prefix linear row capture has an invalid original surface"; return false;
    }
    for (const char *p = surface; *p; ++p) {
        if ((*p < 'a' || *p > 'z') && *p != '_') {
            error = "prefix linear row capture has an invalid surface name"; return false;
        }
    }
    try {
        const std::filesystem::path root(plan.directory);
        if (!std::filesystem::is_regular_file(root / "capture.json")) {
            error = "prefix linear row capture requires its completed GDN window"; return false;
        }
        const auto *selected = static_cast<const unsigned char *>(source) + size_t(tokens - 1u) * width * element_bytes;
        return save(root / (std::string("terminal-") + surface + (element_bytes == 2u ? "-bf16.bin" : "-f32.bin")),
                    selected, width * element_bytes, copy, Clock::now(), error);
    } catch (...) {
        error = "prefix linear row capture filesystem or allocation failure"; return false;
    }
}
}
