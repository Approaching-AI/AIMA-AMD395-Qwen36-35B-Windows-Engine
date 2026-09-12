// Isolated original-input suffix replay; never linked into the model runtime.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <hip/hip_runtime.h>
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
void check(hipError_t code) {
    if (code != hipSuccess) throw std::runtime_error(hipGetErrorString(code));
}
float fp32(uint16_t x) {
    uint32_t bits = uint32_t(x) << 16; float value;
    std::memcpy(&value, &bits, 4); return value;
}
uint16_t bf16(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
std::vector<uint16_t> read(const std::filesystem::path& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(count * 2u))
        throw std::runtime_error("invalid capture length: " + path.string());
    std::vector<uint16_t> result(count);
    file.seekg(0); file.read(reinterpret_cast<char*>(result.data()), count * 2u);
    if (!file) throw std::runtime_error("capture read failed");
    for (auto x : result) if (!std::isfinite(fp32(x))) throw std::runtime_error("nonfinite capture");
    return result;
}
template<class T> void write(const std::filesystem::path& path, const std::vector<T>& data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
    file.close(); if (!file) throw std::runtime_error("output write failed");
}
struct Buffer {
    uint16_t* value = nullptr;
    explicit Buffer(size_t bytes) { check(hipMalloc(reinterpret_cast<void**>(&value), bytes)); }
    ~Buffer() { if (value) (void)hipFree(value); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<uint16_t>& data) {
        check(hipMemcpy(value, data.data(), data.size() * 2u, hipMemcpyHostToDevice));
    }
    bool matches(const std::vector<uint16_t>& data) {
        std::vector<uint16_t> current(data.size());
        check(hipMemcpy(current.data(), value, current.size() * 2u, hipMemcpyDeviceToHost));
        return current == data;
    }
};
struct Provider {
    using Launch = int (*)(const uint16_t*, const uint16_t*, const uint16_t*,
        const uint16_t*, const uint16_t*, float*, void*, unsigned, unsigned);
    HMODULE module = nullptr; Launch launch = nullptr;
    int (*release)() = nullptr;
    ~Provider() { if (release) (void)release(); if (module) FreeLibrary(module); }
    void load(const char* path) {
        module = LoadLibraryA(path);
        if (!module) throw std::runtime_error("cannot load CK provider");
        launch = reinterpret_cast<Launch>(GetProcAddress(module, "qrt_ck_fmha_sm121_suffix_bf16_v1"));
        release = reinterpret_cast<decltype(release)>(GetProcAddress(module, "qrt_ck_fmha_q8192_release"));
        auto prepare = reinterpret_cast<int (*)()>(GetProcAddress(module, "qrt_ck_fmha_q8192_prepare"));
        if (!launch || !release || !prepare) throw std::runtime_error("incomplete exact suffix ABI");
        check(hipError_t(prepare()));
    }
};
struct Event {
    hipEvent_t value = nullptr;
    Event() { check(hipEventCreate(&value)); }
    ~Event() { if (value) (void)hipEventDestroy(value); }
};
unsigned parse(const char* value) {
    const std::string text(value);
    if (text.empty() || text.size() > 5 || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid token count");
    return std::stoul(text);
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: suffix_attention_capture_replay provider.dll inputs outputs prefix_tokens suffix_tokens");
        const unsigned prefix = parse(argv[4]), suffix = parse(argv[5]);
        if (!prefix || prefix >= 32768u || !suffix || suffix > 1024u || suffix > 32768u - prefix)
            throw std::runtime_error("unsupported bounded original suffix");
        char hostname[256]{}; DWORD length = sizeof(hostname);
        if (!GetComputerNameA(hostname, &length) || _stricmp(hostname, "baiying") != 0)
            throw std::runtime_error("requires baiying");
        const std::filesystem::path input(argv[2]), output(argv[3]);
        if (!std::filesystem::create_directory(output)) throw std::runtime_error("output directory must be new");
        const size_t query_cells = size_t(suffix) * 4096u, kv_cells = size_t(prefix + suffix) * 512u;
        const auto q = read(input/"query-bf16.bin", query_cells);
        const auto k = read(input/"key-bf16.bin", kv_cells);
        const auto v = read(input/"value-bf16.bin", kv_cells);
        const auto expected = read(input/"expected-output-bf16.bin", query_cells);
        check(hipSetDevice(0)); hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
        if (std::string(device.gcnArchName).find("gfx1151") != 0) throw std::runtime_error("requires gfx1151");
        Provider provider; provider.load(argv[1]);
        Buffer dq(q.size() * 2u), dk(k.size() * 2u), dv(v.size() * 2u), result((query_cells + 128u) * 4u);
        dq.upload(q); dk.upload(k); dv.upload(v);
        auto* cells = reinterpret_cast<float*>(result.value);
        Event begin, end; float times[4]{};
        std::vector<float> first; std::vector<uint16_t> rounded;
        size_t mismatches = 0u, zero_differences = 0u;
        double maximum = 0.0, error2 = 0.0, norm2 = 0.0;
        bool repeat = true, refreshed = false;
        for (unsigned run = 0u; run < 4u; ++run) {
            if (run == 2u) {
                check(hipMemset(dk.value, 0, size_t(prefix) * 1024u));
                check(hipMemset(dv.value, 0, size_t(prefix) * 1024u));
            } else if (run == 3u) { dk.upload(k); dv.upload(v); }
            check(hipMemset(result.value, 0xa5, (query_cells + 128u) * 4u));
            check(hipEventRecord(begin.value, nullptr));
            check(hipError_t(provider.launch(dq.value, dk.value, dv.value,
                dk.value + size_t(prefix) * 512u, dv.value + size_t(prefix) * 512u,
                cells + 64u, nullptr, prefix, suffix)));
            check(hipEventRecord(end.value, nullptr)); check(hipEventSynchronize(end.value));
            check(hipEventElapsedTime(&times[run], begin.value, end.value));
            std::vector<uint32_t> bits(query_cells + 128u);
            check(hipMemcpy(bits.data(), result.value, bits.size() * 4u, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < 64u; ++i)
                if (bits[i] != 0xa5a5a5a5u || bits[query_cells + 64u + i] != 0xa5a5a5a5u)
                    throw std::runtime_error("suffix output redzone changed");
            std::vector<float> values(query_cells);
            std::memcpy(values.data(), bits.data() + 64u, query_cells * 4u);
            for (float x : values) if (!std::isfinite(x)) throw std::runtime_error("nonfinite native attention");
            if (run == 0u) {
                first = values; rounded.resize(query_cells);
                std::transform(values.begin(), values.end(), rounded.begin(), bf16);
                for (size_t i = 0; i < query_cells; ++i) {
                    mismatches += rounded[i] != expected[i];
                    const double gold = fp32(expected[i]), delta = fp32(rounded[i]) - gold;
                    maximum = std::max(maximum, std::abs(delta)); error2 += delta * delta; norm2 += gold * gold;
                }
            } else if (run == 1u) repeat = std::memcmp(first.data(), values.data(), query_cells * 4u) == 0;
            else if (run == 2u) {
                for (size_t i = 0; i < query_cells; ++i) zero_differences += bf16(values[i]) != rounded[i];
            } else refreshed = std::memcmp(first.data(), values.data(), query_cells * 4u) == 0;
        }
        const bool unchanged = dq.matches(q) && dk.matches(k) && dv.matches(v);
        const bool pass = !mismatches && repeat && refreshed && unchanged && zero_differences;
        write(output/"native-output-bf16.bin", rounded); write(output/"native-output-f32.bin", first);
        std::ofstream record(output/"result.json"); record << std::setprecision(17);
        record << "{\"kind\":\"original_suffix_attention_capture_replay\",\"host\":\"baiying\",\"prefix_tokens\":" << prefix
            << ",\"suffix_tokens\":" << suffix << ",\"elements\":" << query_cells << ",\"bf16_mismatches\":" << mismatches
            << ",\"maximum_absolute_error\":" << maximum << ",\"relative_l2\":" << std::sqrt(error2 / std::max(norm2, 1e-300))
            << ",\"repeat_fp32_bitwise\":" << (repeat ? "true" : "false") << ",\"same_address_refresh_fp32_bitwise\":" << (refreshed ? "true" : "false")
            << ",\"input_cells_unchanged\":" << (unchanged ? "true" : "false") << ",\"zero_prefix_kv_bf16_differences\":" << zero_differences
            << ",\"output_redzones_pass\":true,\"component_pass\":" << (pass ? "true" : "false")
            << ",\"operator_gpu_ms\":[" << times[0] << ',' << times[1] << ',' << times[2] << ',' << times[3]
            << "],\"diagnostic_only\":true,\"model_inference_acceptance\":false,\"performance_acceptance\":false}\n";
        record.close(); if (!record) throw std::runtime_error("result write failed");
        std::cout << "suffix_attention_component_pass=" << pass << " model_inference_acceptance=0\n";
        return pass ? 0 : 6;
    } catch (const std::exception& error) {
        (void)hipDeviceSynchronize(); std::cerr << error.what() << '\n'; return 2;
    }
}
