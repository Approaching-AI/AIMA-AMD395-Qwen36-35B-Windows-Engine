// Isolated original-input replay. Never linked into the inference runtime.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <hip/hip_runtime.h>
#include "fla_checkpoint.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t kStateElements = 32u * 128u * 128u;
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
float as_float(float x) { return x; }
float as_float(uint16_t x) {
    uint32_t bits = uint32_t(x) << 16; float value;
    std::memcpy(&value, &bits, 4); return value;
}
uint16_t bf16(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
// Original FLA captures use [head][value][key]. The optional key-major ABI
// needs a bitwise transpose at the fixture boundary, in both directions.
std::vector<float> transpose_state(const std::vector<float>& source) {
    if (source.size() != kStateElements) throw std::runtime_error("invalid state size");
    std::vector<float> result(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        const size_t j = i / 16384u * 16384u + (i % 128u) * 128u + (i % 16384u) / 128u;
        std::memcpy(&result[i], &source[j], sizeof(float));
    }
    return result;
}
template<class T> std::vector<T> read(const std::filesystem::path& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(T)))
        throw std::runtime_error("capture size mismatch: " + path.string());
    std::vector<T> values(count);
    file.seekg(0); file.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    if (!file) throw std::runtime_error("capture read failed: " + path.string());
    for (T x : values) if (!std::isfinite(as_float(x))) throw std::runtime_error("nonfinite capture input");
    return values;
}
template<class T> void write(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    file.close();
    if (!file) throw std::runtime_error("capture write failed: " + path.string());
}
struct Buffer {
    float* value = nullptr;
    explicit Buffer(size_t elements) { check(hipMalloc(reinterpret_cast<void**>(&value), elements * sizeof(float))); }
    ~Buffer() { if (value) (void)hipFree(value); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<float>& source) {
        check(hipMemcpy(value, source.data(), source.size() * sizeof(float), hipMemcpyHostToDevice));
    }
    std::vector<float> download(size_t count) const {
        std::vector<float> result(count);
        check(hipMemcpy(result.data(), value, count * sizeof(float), hipMemcpyDeviceToHost));
        for (float x : result) if (!std::isfinite(x)) throw std::runtime_error("nonfinite native output");
        return result;
    }
};
struct Provider {
    HMODULE module = nullptr;
    int (*prepare)(const char*) = nullptr;
    qrt_fla_checkpoint::SeededLaunch launch = nullptr;
    const char* (*error)() = nullptr;
    void (*release)() = nullptr;
    ~Provider() { if (release) release(); if (module) FreeLibrary(module); }
    void load(const char* dll, const char* kernels, bool key_major) {
        module = LoadLibraryA(dll);
        if (!module) throw std::runtime_error("cannot load selected FLA provider");
        prepare = reinterpret_cast<decltype(prepare)>(GetProcAddress(module, "qrt_aiter_fused_gdn_q8192_prepare"));
        launch = reinterpret_cast<decltype(launch)>(GetProcAddress(module, key_major
            ? "qrt_fla_gdn_launch_async_seeded_key_major_f32_v1"
            : "qrt_fla_gdn_launch_async_seeded_f32_v1"));
        error = reinterpret_cast<decltype(error)>(GetProcAddress(module, "qrt_aiter_fused_gdn_q8192_last_error"));
        release = reinterpret_cast<decltype(release)>(GetProcAddress(module, "qrt_aiter_fused_gdn_q8192_release"));
        if (!prepare || !launch || !error || !release) throw std::runtime_error("incomplete seeded FLA ABI");
        if (!prepare(kernels)) throw std::runtime_error(error());
    }
};
struct Event {
    hipEvent_t value = nullptr;
    Event() { check(hipEventCreate(&value)); }
    ~Event() { if (value) (void)hipEventDestroy(value); }
};
struct Stats {
    size_t elements = 0, bit_mismatches = 0;
    double maximum = 0, error2 = 0, reference2 = 0;
    template<class T> void compare(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) throw std::runtime_error("comparison length changed");
        elements = a.size();
        for (size_t i = 0; i < a.size(); ++i) {
            bit_mismatches += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
            const double expected = as_float(b[i]), delta = as_float(a[i]) - expected;
            maximum = std::max(maximum, std::abs(delta)); error2 += delta * delta;
            reference2 += expected * expected;
        }
    }
    double relative_l2() const { return std::sqrt(error2 / std::max(reference2, 1e-300)); }
    void json(std::ostream& out) const {
        out << "{\"elements\":" << elements << ",\"bit_mismatches\":" << bit_mismatches
            << ",\"maximum_absolute_error\":" << maximum << ",\"relative_l2\":" << relative_l2() << '}';
    }
};
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error("usage: fla_seeded_capture_replay kernels provider.dll inputs outputs tokens value-major|key-major");
        const std::string layout(argv[6]);
        if (layout != "value-major" && layout != "key-major") throw std::runtime_error("explicit state layout required");
        const bool key_major = layout == "key-major";
        const std::string text(argv[5]);
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos || text.size() > 4)
            throw std::runtime_error("invalid token count");
        const unsigned tokens = std::stoul(text);
        if (!tokens || tokens > 1024) throw std::runtime_error("replay requires 1..1024 original inputs");
        char hostname[256]{}; DWORD length = sizeof(hostname);
        if (!GetComputerNameA(hostname, &length) || _stricmp(hostname, "baiying") != 0)
            throw std::runtime_error("requires baiying");
        const std::filesystem::path input(argv[3]), output(argv[4]);
        if (!std::filesystem::create_directory(output)) throw std::runtime_error("output directory must be new");
        const auto raw = read<float>(input / "raw-f32.bin", size_t(tokens) * 8192u);
        const auto gates = read<float>(input / "gates-f32.bin", size_t(tokens) * 64u);
        const auto original_initial = read<float>(input / "initial-value-major-f32.bin", kStateElements);
        const auto initial = key_major ? transpose_state(original_initial) : original_initial;
        const auto expected_output = read<uint16_t>(input / "expected-output-bf16.bin", size_t(tokens) * 4096u);
        const auto expected_state = read<float>(input / "expected-value-major-f32.bin", kStateElements);
        check(hipSetDevice(0)); hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
        if (std::string(device.gcnArchName).find("gfx1151") != 0) throw std::runtime_error("requires gfx1151");
        Provider provider; provider.load(argv[2], argv[1], key_major);
        Buffer d_raw(raw.size()), d_gates(gates.size()), d_output(expected_output.size()), d_state(kStateElements);
        d_raw.upload(raw); d_gates.upload(gates);
        Event begin, end;
        std::vector<uint16_t> first_output;
        std::vector<float> first_state;
        float timings[3]{}; bool repeated = false;
        Stats output_stats, state_stats, zero_control;
        for (unsigned run = 0; run < 3; ++run) {
            if (run == 2) check(hipMemset(d_state.value, 0, kStateElements * sizeof(float)));
            else d_state.upload(initial);
            check(hipEventRecord(begin.value, nullptr));
            if (!provider.launch(d_raw.value, d_gates.value, d_output.value, d_state.value, 0, nullptr, tokens)) {
                (void)hipDeviceSynchronize(); throw std::runtime_error(provider.error());
            }
            check(hipEventRecord(end.value, nullptr)); check(hipEventSynchronize(end.value));
            check(hipEventElapsedTime(&timings[run], begin.value, end.value));
            auto state = d_state.download(kStateElements);
            if (key_major) state = transpose_state(state);
            const auto values = d_output.download(expected_output.size());
            std::vector<uint16_t> rounded(values.size());
            std::transform(values.begin(), values.end(), rounded.begin(), bf16);
            if (run == 0) {
                output_stats.compare(rounded, expected_output); state_stats.compare(state, expected_state);
                first_output = std::move(rounded); first_state = std::move(state);
            } else if (run == 1) {
                repeated = rounded == first_output && std::memcmp(state.data(), first_state.data(), state.size() * sizeof(float)) == 0;
            } else zero_control.compare(rounded, first_output);
        }
        const auto after_raw = d_raw.download(raw.size()), after_gates = d_gates.download(gates.size());
        const bool inputs_unchanged = std::memcmp(raw.data(), after_raw.data(), raw.size() * sizeof(float)) == 0 &&
            std::memcmp(gates.data(), after_gates.data(), gates.size() * sizeof(float)) == 0;
        constexpr double state_tolerance = 1e-5;
        const bool pass = output_stats.bit_mismatches == 0 && state_stats.maximum <= state_tolerance &&
            state_stats.relative_l2() <= state_tolerance && repeated && inputs_unchanged && zero_control.bit_mismatches != 0;
        write(output / "native-output-bf16.bin", first_output);
        write(output / "native-value-major-f32.bin", first_state);
        std::ofstream record(output / "result.json"); record << std::setprecision(17);
        record << "{\"kind\":\"original_seeded_fla_capture_replay\",\"host\":\"baiying\",\"tokens\":" << tokens
            << ",\"capture_state_layout\":\"value_head_value_key_fp32\",\"interface_state_layout\":\"" << layout << '"'
            << ",\"fixture_state_transpose\":" << (key_major ? "true" : "false")
            << ",\"diagnostic_only\":true,\"product_acceptance\":false,\"component_pass\":" << (pass ? "true" : "false")
            << ",\"output_bf16\":"; output_stats.json(record);
        record << ",\"state_f32\":"; state_stats.json(record);
        record << ",\"state_absolute_and_relative_tolerance\":" << state_tolerance
            << ",\"seeded_repeat_bitwise\":" << (repeated ? "true" : "false")
            << ",\"inputs_unchanged\":" << (inputs_unchanged ? "true" : "false")
            << ",\"zero_initial_state_vs_seeded\":"; zero_control.json(record);
        record << ",\"operator_gpu_ms\":[" << timings[0] << ',' << timings[1] << ',' << timings[2] << "]}\n";
        record.close(); if (!record) throw std::runtime_error("result write failed");
        std::cout << "seeded_fla_component_pass=" << (pass ? 1 : 0) << " product_acceptance=0\n";
        return pass ? 0 : 6;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 2;
    }
}
