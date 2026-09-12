#include "../../native/providers/ck_fmha/blackwell_attention.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer) (void)hipFree(pointer); }
    template<class T> T* as() { return static_cast<T*>(pointer); }
};
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("probability completion deadline");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(hipEventDestroy(event));
}
template<class T> void upload(Device& d, const std::vector<T>& v) {
    check(hipMemcpy(d.pointer, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
}
template<class T> std::vector<T> download(Device& d, size_t count) {
    std::vector<T> v(count); check(hipMemcpy(v.data(), d.pointer, count * sizeof(T), hipMemcpyDeviceToHost)); return v;
}
float score_value(unsigned row, unsigned key, unsigned mode) {
    uint32_t random = (row + 3u) * 747796405u + key * 2891336453u;
    random = (random ^ (random >> 16u)) * 2246822519u;
    if (mode == 0u) return float(int(random % 10241u) - 5120) / 256.0f;
    if (mode == 1u) return key % 7u == 0u ? 8.0f : -float(random % 4096u) / 128.0f;
    if (mode == 2u) return float(key / 32u) * 256.0f - float(key % 32u);
    if (mode == 3u) return key % 2u ? -0.0f : 0.0f;
    if (mode == 4u) return std::ldexp(float(int(random % 33u) - 16), -129);
    return (key / 32u) % 3u == 0u ? -1000.0f : float(int(random % 65u) - 32) / 8.0f;
}
void run(unsigned start, unsigned queries, unsigned mode, bool vllm, const unsigned char* table) {
    constexpr size_t guard = 64u;
    const unsigned stride = start + queries, rows = queries * kQueryHeads;
    const unsigned tile_stride = (stride + 31u) / 32u;
    const size_t cells = size_t(rows) * stride, scale_cells = size_t(rows) * (tile_stride + 1u);
    std::vector<float> scores(cells + 2u * guard, 12345.0f), scales(scale_cells + 2u * guard, 12345.0f);
    std::vector<uint16_t> probability(cells + 2u * guard, 0xa5a5u);
    for (unsigned row = 0u; row < rows; ++row) {
        const unsigned tokens = start + row / kQueryHeads + 1u;
        for (unsigned key = 0u; key < tokens; ++key)
            scores[guard + size_t(row) * stride + key] = score_value(row, key, mode);
    }
    Device ds(scores.size() * sizeof(float)), dp(probability.size() * sizeof(uint16_t)), da(scales.size() * sizeof(float));
    upload(ds, scores); upload(dp, probability); upload(da, scales);
    hipLaunchKernelGGL(blackwell_online_probability_kernel, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
        ds.as<float>() + guard, dp.as<uint16_t>() + guard, da.as<float>() + guard, start, stride, table, vllm);
    check(hipGetLastError()); finish();
    const auto control_p = download<uint16_t>(dp, probability.size());
    const auto control_s = download<float>(da, scales.size());
    upload(dp, probability); upload(da, scales);
    hipLaunchKernelGGL(blackwell_parallel_probability_kernel, dim3(kQueryHeads, queries), dim3(kThreads), 0u, nullptr,
        ds.as<float>() + guard, dp.as<uint16_t>() + guard, da.as<float>() + guard, start, stride, table, vllm);
    check(hipGetLastError()); finish();
    const auto actual_p = download<uint16_t>(dp, probability.size());
    const auto actual_s = download<float>(da, scales.size());
    const auto actual_scores = download<float>(ds, scores.size());
    if (std::memcmp(actual_scores.data(), scores.data(), scores.size() * sizeof(float))) throw std::runtime_error("scores changed");
    size_t bad_p = 0u, bad_s = 0u, denominators = 0u, zero_alphas = 0u;
    for (size_t i = 0u; i < actual_p.size(); ++i) {
        bad_p += actual_p[i] != control_p[i];
        bool live = i >= guard && i < guard + cells;
        if (live) {
            const unsigned row = unsigned((i - guard) / stride), key = unsigned((i - guard) % stride);
            const unsigned tokens = start + row / kQueryHeads + 1u;
            live = key < ((tokens + 31u) / 32u) * 32u;
            if (live && mode == 3u && actual_p[i] != (key < tokens ? 0x3f80u : 0u))
                throw std::runtime_error("constant probability CPU check");
        }
        if (!live && actual_p[i] != probability[i]) throw std::runtime_error("probability tail or redzone changed");
    }
    for (size_t i = 0u; i < actual_s.size(); ++i) {
        const bool mismatch = std::memcmp(&actual_s[i], &control_s[i], sizeof(float)) != 0;
        bad_s += mismatch;
        bool live = i >= guard && i < guard + scale_cells;
        if (live) {
            const unsigned row = unsigned((i - guard) / (tile_stride + 1u)), tile = unsigned((i - guard) % (tile_stride + 1u));
            const unsigned tokens = start + row / kQueryHeads + 1u;
            live = tile < (tokens + 31u) / 32u || tile == tile_stride;
            if (live && !std::isfinite(actual_s[i])) throw std::runtime_error("nonfinite scale");
            if (tile == tile_stride) denominators += mismatch;
            else if (live && actual_s[i] == 0.0f) ++zero_alphas;
            if (live && mode == 3u && actual_s[i] != (tile == tile_stride ? float(tokens) : tile == 0u ? 0.0f : 1.0f))
                throw std::runtime_error("constant denominator CPU check");
        }
        if (!live && actual_s[i] != scales[i]) throw std::runtime_error("scale tail or redzone changed");
    }
    std::printf("{\"kind\":\"parallel_probability\",\"query_start\":%u,\"queries\":%u,\"mode\":%u,\"vllm_sum\":%s,\"sm121_exp2_table\":%s,\"probability_cells\":%llu,\"scale_cells\":%llu,\"probability_bit_mismatches\":%llu,\"scale_bit_mismatches\":%llu,\"denominator_bit_mismatches\":%llu,\"zero_alphas\":%llu,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        start, queries, mode, vllm ? "true" : "false", table ? "true" : "false", (unsigned long long)cells,
        (unsigned long long)scale_cells, (unsigned long long)bad_p, (unsigned long long)bad_s, (unsigned long long)denominators, (unsigned long long)zero_alphas);
    if (bad_p || bad_s) throw std::runtime_error("parallel probability differs from original one-wave recurrence");
}
}
int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("requires SHA-verified SM121 exp2 table path");
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(qrt_sm121_exp2::table_bytes)) throw std::runtime_error("table size mismatch");
    std::vector<unsigned char> table(qrt_sm121_exp2::table_bytes);
    file.seekg(0); file.read(reinterpret_cast<char*>(table.data()), table.size());
    if (!file || !qrt_sm121_exp2::valid_layout(table.data(), table.size())) throw std::runtime_error("table layout mismatch");
    Device dt(table.size()); upload(dt, table);
    for (auto shape : {std::pair<unsigned,unsigned>{0,1},{0,32},{31,2},{17,32},{255,2},{7167,2},{8191,1},{32767,1},{65535,1}})
        for (unsigned mode = 0u; mode < 6u; ++mode)
            for (bool vllm : {false, true})
                for (bool use_table : {false, true}) run(shape.first, shape.second, mode, vllm, use_table ? dt.as<unsigned char>() : nullptr);
    const auto after = download<unsigned char>(dt, table.size());
    if (after != table) throw std::runtime_error("exp2 table changed");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "parallel_probability_selftest_error=%s\n", e.what()); return 2; }
