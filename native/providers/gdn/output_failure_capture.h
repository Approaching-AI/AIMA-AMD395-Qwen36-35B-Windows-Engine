#pragma once
#include "completion_guard.h"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <vector>

// Diagnostic copies of a completed, rejected output stage. No kernel is
// retried and no captured value is supplied to the model. Failure to capture
// must leave the original operation failure intact.
namespace qrt_fla_output_failure {
constexpr size_t copy_bytes = 1u << 20u;
constexpr size_t maximum_bytes = 64u << 20u;

template<class Copy>
bool capture(const char* directory, unsigned count,
    const uint16_t* q, const uint16_t* k, const uint16_t* v,
    const uint16_t* h, const float* g, const uint16_t* scores,
    const float* output, const qrt_fla_completion::Observation& clock, Copy copy) {
    if (!directory || !*directory) return true;
    if (!count || count > 1024u || !q || !k || !v || !h || !g || !scores || !output ||
        !clock.completed || !std::isfinite(clock.gpu_ms) || !std::isfinite(clock.host_ms) ||
        clock.gpu_ms < 0.0 || clock.host_ms < 0.0 ||
        qrt_fla_completion::evaluate(clock.gpu_ms, clock.host_ms).accepted()) return false;
    const size_t chunks = (count + 63u) / 64u;
    struct Surface { const char* name; const void* data; size_t bytes; };
    const Surface surfaces[] = {
        {"q-bf16.bin", q, size_t(count) * 2048u * 2u},
        {"k-bf16.bin", k, size_t(count) * 2048u * 2u},
        {"v-new-bf16.bin", v, size_t(count) * 4096u * 2u},
        {"chunk-state-bf16.bin", h, chunks * 524288u * 2u},
        {"g-cumsum-f32.bin", g, size_t(count) * 32u * 4u},
        {"scores-bf16.bin", scores, size_t(count) * 2048u * 2u},
        {"output-f32.bin", output, size_t(count) * 4096u * 4u}
    };
    size_t total = 0;
    for (const auto& surface : surfaces) total += surface.bytes;
    if (total > maximum_bytes) return false;
    try {
        const std::filesystem::path root(directory);
        if (!std::filesystem::create_directory(root)) return false;
        std::vector<unsigned char> host(copy_bytes);
        for (const auto& surface : surfaces) {
            std::ofstream file(root / surface.name, std::ios::binary);
            if (!file) return false;
            for (size_t offset = 0; offset < surface.bytes; offset += copy_bytes) {
                const size_t bytes = std::min(copy_bytes, surface.bytes - offset);
                if (!copy(host.data(), static_cast<const unsigned char*>(surface.data) + offset, bytes)) return false;
                file.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(bytes));
                if (!file) return false;
            }
            file.close();
            if (!file) return false;
        }
        std::ofstream record(root / "capture.json");
        record << std::setprecision(17)
            << "{\"kind\":\"completed_fla_output_guard_failure\",\"tokens\":" << count
            << ",\"chunks\":" << chunks << ",\"gpu_ms\":" << clock.gpu_ms
            << ",\"host_ms\":" << clock.host_ms << ",\"guard_ms\":100,\"bytes\":" << total
            << ",\"copy_chunk_bytes\":1048576,\"maximum_bytes\":67108864,"
               "\"stage_completed\":true,\"guard_accepted\":false,\"complete\":true,"
               "\"kernel_retried\":false,\"native_tensor_inputs\":false,"
               "\"numerical_acceptance\":false}\n";
        record.close();
        return bool(record);
    } catch (...) { return false; }
}
} // namespace qrt_fla_output_failure
