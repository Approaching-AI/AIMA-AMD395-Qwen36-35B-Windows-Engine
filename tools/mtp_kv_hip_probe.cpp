#include <hip/hip_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "native/providers/gdn/sm121_mtp_kv.h"

void check(hipError_t code) {
    if (code != hipSuccess) throw std::runtime_error(hipGetErrorString(code));
}
template<class T> std::vector<T> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() % sizeof(T))
        throw std::runtime_error("invalid input file");
    const size_t bytes = static_cast<size_t>(file.tellg());
    if (!bytes || bytes > 128u * 1024u * 1024u)
        throw std::runtime_error("empty file or size exceeds bound");
    std::vector<T> result(bytes / sizeof(T));
    file.seekg(0); file.read(reinterpret_cast<char*>(result.data()), bytes);
    if (!file) throw std::runtime_error("short input file");
    return result;
}
struct Scratch {
    std::vector<void*> pointers;
    ~Scratch() { for (auto pointer : pointers) (void)hipFree(pointer); }
    template<class T> T* upload(const std::vector<T>& values) {
        T* pointer = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&pointer), values.size() * sizeof(T)));
        pointers.push_back(pointer);
        check(hipMemcpy(pointer, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
        return pointer;
    }
};

int main(int argc, char** argv) try {
    if (argc != 8)
        throw std::runtime_error("kv key_weights expected_norm expected_kv positions rsqrt_table rope_table");
    const auto input = read_file<uint16_t>(argv[1]);
    const auto weights = read_file<uint16_t>(argv[2]);
    const auto expected_norm = read_file<uint16_t>(argv[3]);
    const auto expected_kv = read_file<uint16_t>(argv[4]);
    const auto positions = read_file<uint32_t>(argv[5]);
    const auto table = read_file<unsigned char>(argv[6]);
    const auto rope = read_file<uint16_t>(argv[7]);
    const size_t rows = input.size() / 1024u;
    if (!rows || rows > 8192u || input.size() != rows * 1024u || weights.size() != 256u ||
        expected_norm.size() != rows * 512u || expected_kv.size() != input.size() ||
        positions.size() != rows || rope.size() % 64u ||
        !qrt_sm121_rsqrt::valid_layout(table.data(), table.size()))
        throw std::runtime_error("inconsistent KV shapes or table");
    for (size_t i = 0; i < rows; ++i) {
        if (positions[i] >= 262144u || positions[i] >= rope.size() / 64u ||
            (i && positions[i] <= positions[i - 1u]))
            throw std::runtime_error("positions must increase within the original MTP limit");
    }
    constexpr unsigned int capacity = 262144u;
    constexpr size_t guard = 64u;
    constexpr uint16_t sentinel = 0xa5a5u;
    // Verify every untouched cache cell, including large gaps between the
    // selected original rows and both ends of the complete 256k allocation.
    std::vector<uint16_t> expected_cache(size_t(capacity) * 1024u + guard * 2u, sentinel);
    std::vector<uint16_t> actual_cache(expected_cache.size(), sentinel);
    std::vector<uint16_t> actual_norm(expected_norm.size() + guard * 2u, sentinel);
    for (size_t row = 0; row < rows; ++row)
        std::copy_n(expected_kv.data() + row * 1024u, 1024u,
                    expected_cache.data() + guard + size_t(positions[row]) * 1024u);
    Scratch scratch;
    const auto device_input = scratch.upload(input), device_weights = scratch.upload(weights);
    const auto device_table = scratch.upload(table);
    const auto device_rope = scratch.upload(rope);
    auto device_cache = scratch.upload(actual_cache), device_norm = scratch.upload(actual_norm);
    unsigned int launches = 0;
    for (size_t first = 0; first < rows;) {
        size_t last = first + 1u;
        while (last < rows && positions[last] == positions[last - 1u] + 1u) ++last;
        check(qrt_sm121_mtp::launch_key_values(device_input + first * 1024u, device_weights,
            device_table, device_rope, static_cast<unsigned int>(rope.size() / 64u),
            positions[first], static_cast<unsigned int>(last - first), capacity,
            device_cache + guard, device_norm + guard + first * 512u));
        ++launches;
        first = last;
    }
    check(hipDeviceSynchronize());
    check(hipMemcpy(actual_cache.data(), device_cache, actual_cache.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    check(hipMemcpy(actual_norm.data(), device_norm, actual_norm.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    size_t norm_bad = 0, key_bad = 0, value_bad = 0, guard_bad = 0;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t i = 0; i < 512u; ++i) {
            norm_bad += actual_norm[guard + row * 512u + i] != expected_norm[row * 512u + i];
            key_bad += actual_cache[guard + size_t(positions[row]) * 1024u + i] != expected_kv[row * 1024u + i];
            value_bad += actual_cache[guard + size_t(positions[row]) * 1024u + 512u + i] != expected_kv[row * 1024u + 512u + i];
        }
    }
    size_t all_cache_bad = 0;
    for (size_t i = 0; i < actual_cache.size(); ++i) all_cache_bad += actual_cache[i] != expected_cache[i];
    guard_bad = all_cache_bad - key_bad - value_bad;
    for (size_t i = 0; i < guard; ++i) {
        guard_bad += actual_norm[i] != sentinel;
        guard_bad += actual_norm[actual_norm.size() - 1u - i] != sentinel;
    }
    const bool passed = !norm_bad && !key_bad && !value_bad && !guard_bad;
    std::cout << "{\"kind\":\"original_mtp_kv_native_replay\",\"rows\":" << rows
              << ",\"launches\":" << launches << ",\"cache_capacity\":" << capacity
              << ",\"norm_elements\":" << rows * 512u << ",\"key_elements\":" << rows * 512u
              << ",\"value_elements\":" << rows * 512u << ",\"norm_bf16_mismatches\":" << norm_bad
              << ",\"key_bf16_mismatches\":" << key_bad << ",\"value_bf16_mismatches\":" << value_bad
              << ",\"guard_mismatches\":" << guard_bad
              << ",\"guard_elements\":" << actual_cache.size() - rows * 1024u + 2u * guard
              << ",\"passed\":" << (passed ? "true" : "false") << ",\"inference_acceptance\":false}\n";
    return passed ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
}
