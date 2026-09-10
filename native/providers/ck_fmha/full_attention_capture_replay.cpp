// Read-only captured-input numerical replay. No model, oracle output, or
// reference tensor is consumed by device computation.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "blackwell_attention.h"
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
void check(hipError_t result) {
    if (result != hipSuccess) throw std::runtime_error(hipGetErrorString(result));
}
template<class T> std::vector<T> read(const std::string& path, size_t elements) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(elements * sizeof(T)))
        throw std::runtime_error("capture size mismatch: " + path);
    std::vector<T> result(elements);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(result.data()), elements * sizeof(T)))
        throw std::runtime_error("short capture read: " + path);
    return result;
}
template<class T> void write(const std::string& path, const std::vector<T>& data) {
    // CREATE_NEW is atomic: a prior capture can never be overwritten.
    HANDLE file = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr,
                             CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("output exists: " + path);
    DWORD written = 0;
    const DWORD bytes = DWORD(data.size() * sizeof(T));
    const bool ok = WriteFile(file, data.data(), bytes, &written, nullptr) && written == bytes;
    CloseHandle(file);
    if (!ok) throw std::runtime_error("short capture write: " + path);
}
uint16_t bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4);
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return uint16_t((bits >> 16) | ((bits & 0x7fffffu) ? 0x40u : 0u));
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
float fp32(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16; float result;
    std::memcpy(&result, &bits, 4); return result;
}
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer) (void)hipFree(pointer); }
    template<class T> T* as() { return static_cast<T*>(pointer); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Event {
    hipEvent_t value = nullptr;
    Event() { check(hipEventCreate(&value)); }
    ~Event() { if (value) (void)hipEventDestroy(value); }
};
float finish(Event& begin, Event& end) {
    check(hipEventRecord(end.value)); check(hipEventSynchronize(end.value));
    float result; check(hipEventElapsedTime(&result, begin.value, end.value));
    if (!std::isfinite(result) || result > 3000.0f)
        throw std::runtime_error("component dispatch exceeds 3 seconds");
    return result;
}
void report(const char* route, const std::vector<float>& output,
            const std::vector<uint16_t>& reference, unsigned start,
            const std::string& prefix, float total_ms, float max_ms) {
    size_t mismatches = 0, nonfinite = 0, first = size_t(-1), affected = 0;
    double error2 = 0, norm2 = 0; float maximum_error = 0;
    std::vector<uint16_t> rounded(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        const auto expected = reference[size_t(start) * 4096u + i];
        rounded[i] = bf16(output[i]);
        if (!std::isfinite(output[i])) ++nonfinite;
        if (rounded[i] != expected) { ++mismatches; if (first == size_t(-1)) first = i; }
        const double difference = double(fp32(rounded[i])) - fp32(expected);
        error2 += difference * difference; norm2 += double(fp32(expected)) * fp32(expected);
        maximum_error = std::max(maximum_error, float(std::abs(difference)));
    }
    for (size_t token = 0; token < output.size() / 4096u; ++token) {
        const auto* r = reference.data() + (size_t(start) + token) * 4096u;
        if (std::memcmp(rounded.data() + token * 4096u, r, 8192u)) ++affected;
    }
    write(prefix + "-" + route + "-f32.bin", output);
    write(prefix + "-" + route + "-bf16.bin", rounded);
    std::cout << "{\"kind\":\"full_attention_capture_replay\",\"route\":\"" << route
              << "\",\"query_start\":" << start << ",\"elements\":" << output.size()
              << ",\"bf16_mismatches\":" << mismatches << ",\"affected_tokens\":" << affected
              << ",\"nonfinite\":" << nonfinite << ",\"first_mismatch\":" << (first == size_t(-1) ? -1LL : (long long)first)
              << ",\"maximum_absolute_error\":" << maximum_error
              << ",\"relative_l2\":" << std::sqrt(error2 / std::max(norm2, 1e-300))
              << ",\"kernel_total_ms\":" << total_ms << ",\"maximum_dispatch_ms\":" << max_ms
              << ",\"reference_is_compute_input\":false,\"inference_acceptance\":false}" << std::endl;
    if (nonfinite) throw std::runtime_error("nonfinite attention output");
}
unsigned parse(const char* text, unsigned maximum) {
    char* end = nullptr; const auto n = std::strtoul(text, &end, 10);
    if (text == end || *end || n > maximum) throw std::runtime_error("invalid bounded integer");
    return unsigned(n);
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 13) throw std::runtime_error(
            "usage: replay Q K V reference CK_DLL output_prefix tokens query_start count batch exp2_table_or_dash baseline_0_or_1");
        const unsigned tokens = parse(argv[7], 8192), start = parse(argv[8], 8191);
        const unsigned count = parse(argv[9], 8192), batch = parse(argv[10], 32);
        const bool baseline = parse(argv[12], 1) != 0;
        if (!tokens || !count || !batch || start >= tokens || count > tokens - start)
            throw std::runtime_error("invalid query span");
        hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
        if (std::string(properties.gcnArchName).find("gfx1151") != 0)
            throw std::runtime_error("requires gfx1151");
        const auto begun = Clock::now();
        const auto q = read<uint16_t>(argv[1], size_t(tokens) * 4096u);
        const auto k = read<uint16_t>(argv[2], size_t(tokens) * 512u);
        const auto v = read<uint16_t>(argv[3], size_t(tokens) * 512u);
        const auto reference = read<uint16_t>(argv[4], size_t(tokens) * 4096u);
        Device dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2);
        check(hipMemcpy(dq.pointer, q.data(), q.size() * 2, hipMemcpyHostToDevice));
        check(hipMemcpy(dk.pointer, k.data(), k.size() * 2, hipMemcpyHostToDevice));
        check(hipMemcpy(dv.pointer, v.data(), v.size() * 2, hipMemcpyHostToDevice));
        Event begin, end;
        if (baseline) {
            HMODULE dll = LoadLibraryA(argv[5]);
            if (!dll) throw std::runtime_error("cannot load CK provider");
            using Launch = int (*)(const uint16_t*, const uint16_t*, const uint16_t*, float*, void*, unsigned);
            auto launch = reinterpret_cast<Launch>(GetProcAddress(dll, "qrt_ck_fmha_dynamic_bf16_launch"));
            if (!launch) { FreeLibrary(dll); throw std::runtime_error("missing dynamic CK export"); }
            Device output(size_t(tokens) * 4096u * 4u);
            check(hipEventRecord(begin.value));
            check(hipError_t(launch(dq.as<uint16_t>(), dk.as<uint16_t>(), dv.as<uint16_t>(), output.as<float>(), nullptr, tokens)));
            const float ms = finish(begin, end);
            std::vector<float> host(size_t(tokens) * 4096u);
            check(hipMemcpy(host.data(), output.pointer, host.size() * 4, hipMemcpyDeviceToHost));
            report("ck", host, reference, 0, argv[6], ms, ms);
            FreeLibrary(dll);
        }
        const bool use_table = std::string(argv[11]) != "-";
        std::vector<unsigned char> table;
        if (use_table) {
            table = read<unsigned char>(argv[11], qrt_sm121_exp2::table_bytes);
            if (!qrt_sm121_exp2::valid_layout(table.data(), table.size()))
                throw std::runtime_error("invalid exponent table layout");
            BCRYPT_ALG_HANDLE algorithm = nullptr; unsigned char digest[32]{};
            if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
                throw std::runtime_error("SHA256 provider unavailable");
            const auto status = BCryptHash(algorithm, nullptr, 0, table.data(), ULONG(table.size()), digest, sizeof(digest));
            BCryptCloseAlgorithmProvider(algorithm, 0);
            if (status < 0 || std::memcmp(digest, qrt_sm121_exp2::sha256, 32))
                throw std::runtime_error("exponent table fingerprint mismatch");
        }
        Device dt(use_table ? table.size() : 4u), output(size_t(count) * 4096u * 4u);
        Device accumulator(size_t(count) * 4096u * 4u), denominator(size_t(count) * 16u * 4u);
        const char* rcp_path = std::getenv("QRT_CK_FMHA_SM121_RCP_TABLE");
        const bool use_rcp = rcp_path && *rcp_path;
        std::vector<unsigned char> rcp_table;
        if (use_rcp) {
            rcp_table = read<unsigned char>(rcp_path, qrt_sm121_attention_rcp::table_bytes);
            if (!qrt_sm121_attention_rcp::valid_layout(rcp_table.data(), rcp_table.size()))
                throw std::runtime_error("invalid reciprocal table layout");
            BCRYPT_ALG_HANDLE algorithm = nullptr; unsigned char digest[32]{};
            if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
                throw std::runtime_error("SHA256 provider unavailable");
            const auto status = BCryptHash(algorithm, nullptr, 0, rcp_table.data(), ULONG(rcp_table.size()), digest, sizeof(digest));
            BCryptCloseAlgorithmProvider(algorithm, 0);
            if (status < 0 || std::memcmp(digest, qrt_sm121_attention_rcp::sha256, 32))
                throw std::runtime_error("reciprocal table fingerprint mismatch");
        }
        Device dr(use_rcp ? rcp_table.size() : 4u);
        if (use_rcp) check(hipMemcpy(dr.pointer, rcp_table.data(), rcp_table.size(), hipMemcpyHostToDevice));
        if (use_table) check(hipMemcpy(dt.pointer, table.data(), table.size(), hipMemcpyHostToDevice));
        float total = 0, maximum = 0;
        for (unsigned offset = 0; offset < count; offset += batch) {
            if (std::chrono::duration<double>(Clock::now() - begun).count() > 150.0)
                throw std::runtime_error("replay aggregate deadline exceeded");
            check(hipEventRecord(begin.value));
            check(hipError_t(qrt_blackwell_attention::launch_queries(dq.as<uint16_t>(), dk.as<uint16_t>(),
                dv.as<uint16_t>(), output.as<float>(), nullptr, start + offset,
                std::min(batch, count - offset), offset, use_table ? dt.as<unsigned char>() : nullptr,
                accumulator.as<float>(), denominator.as<float>(), true,
                use_rcp ? dr.as<unsigned char>() : nullptr)));
            const float ms = finish(begin, end); total += ms; maximum = std::max(maximum, ms);
        }
        std::vector<float> host(size_t(count) * 4096u);
        check(hipMemcpy(host.data(), output.pointer, host.size() * 4, hipMemcpyDeviceToHost));
        const char* route = use_table
            ? (use_rcp ? "blackwell-sm121-exp-rcp" : "blackwell-sm121-exp")
            : (use_rcp ? "blackwell-amd-exp-rcp" : "blackwell-amd-exp");
        report(route, host, reference, start, argv[6], total, maximum);
        check(hipMemcpy(host.data(), accumulator.pointer, host.size() * 4, hipMemcpyDeviceToHost));
        write(std::string(argv[6]) + "-accumulator-f32.bin", host);
        host.resize(size_t(count) * 16u);
        check(hipMemcpy(host.data(), denominator.pointer, host.size() * 4, hipMemcpyDeviceToHost));
        write(std::string(argv[6]) + "-denominator-f32.bin", host);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "attention_capture_replay_error=" << error.what() << std::endl;
        return 1;
    }
}
