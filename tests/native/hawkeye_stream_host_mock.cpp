// This host-only test inserts the production launcher at the marker below.
// Mock HIP operations execute synchronously and record index transport, bounds,
// admission, count-only behavior, and cleanup. Native tests own GPU arithmetic.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "hawkeye_dispatch_policy.h"
#include "moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"

enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorInvalidConfiguration, hipErrorUnknown };
using hipStream_t = void *;
constexpr int hipMemcpyDeviceToHost = 0;
constexpr unsigned int kSelectedHawkeyeCorrectionThreads = 256u;
constexpr unsigned int kSelectedHawkeyeCorrectionMaximumBlocksPerLaunchLimit = 8u;
constexpr unsigned int kThreads = 256u;
struct dim3 { unsigned int x; explicit dim3(unsigned int value) : x(value) {} };
static unsigned int allocations = 0, frees = 0, collections = 0, corrections = 0, rounds = 0;
static unsigned int reject_collection = 0, fail_sync = 0, syncs = 0;
static size_t scratch_bytes = 0;
static bool invalid_grid = false, invalid_range = false;
static std::vector<size_t> corrected;
static size_t total_elements = 0;

hipError_t hipMalloc(void **p, size_t bytes) {
    ++allocations; scratch_bytes = bytes; *p = std::malloc(bytes);
    return *p ? hipSuccess : hipErrorUnknown;
}
hipError_t hipFree(void *p) { ++frees; std::free(p); return hipSuccess; }
hipError_t hipMemsetAsync(void *p, int value, size_t n, hipStream_t) {
    std::memset(p, value, n); return hipSuccess;
}
hipError_t hipMemcpy(void *to, const void *from, size_t n, int) {
    std::memcpy(to, from, n); return hipSuccess;
}
hipError_t hipGetLastError() { return hipSuccess; }
hipError_t hipStreamSynchronize(hipStream_t) {
    return ++syncs == fail_sync ? hipErrorUnknown : hipSuccess;
}
void grid(const char *name, dim3 blocks, dim3 threads) {
    const unsigned int limit = std::strstr(name, "midpoint_correction") ? 8u : 256u;
    if (blocks.x == 0 || blocks.x > limit || threads.x != 256u) invalid_grid = true;
}
#define hipLaunchKernelGGL(kernel, blocks, threads, shared, stream, ...) \
    do { grid(#kernel, blocks, threads); kernel(__VA_ARGS__); } while (0)

void selected_bf16_projection_hawkeye_compact_kernel(
    const float *, const float *, const float *, const float *output,
    unsigned int rows, unsigned int, unsigned int prefix, unsigned int,
    unsigned int *counts, unsigned int *indices, size_t offset, unsigned int count
) {
    ++collections;
    if (offset + count > total_elements) { invalid_range = true; return; }
    unsigned int block_count = 0;
    for (unsigned int j = 0; j < count; ++j) {
        const size_t i = offset + j;
        if (output[i] == 1.00390625f || i < static_cast<size_t>(prefix) * rows) {
            indices[counts[0]++] = static_cast<unsigned int>(i);
            ++block_count;
        }
        if ((j + 1u) % 256u == 0u || j + 1u == count) {
            counts[1] = (std::max)(counts[1], block_count); block_count = 0u;
        }
    }
    if (collections == reject_collection) counts[1] = 65u;
}
void round_f32_outputs_to_bf16_kernel(float *output, unsigned int count) {
    ++rounds;
    for (unsigned int i = 0; i < count; ++i) output[i] = 1.0f;
}
void selected_bf16_projection_hawkeye_midpoint_correction_kernel(
    const uint16_t *, const uint16_t *, float *output, unsigned int rows,
    unsigned int, const unsigned int *indices, unsigned int offset, unsigned int count
) {
    ++corrections;
    // Test calls use the hard cap of eight blocks => 128 candidate subgroups.
    const unsigned int end = (std::min)(count, offset + 128u);
    for (unsigned int j = offset; j < end; ++j) {
        const size_t index = indices[j];
        if (index >= total_elements) { invalid_range = true; continue; }
        corrected.push_back(index);
        output[index] = static_cast<float>((index / rows) * 2u + index % rows);
    }
}

// QRT_ACTUAL_LAUNCHER

void count_only(bool enabled) {
#ifdef _WIN32
    _putenv_s("QRT_QWEN36_HAWKEYE_CORRECTION_COUNT_ONLY", enabled ? "1" : "");
#else
    if (enabled) setenv("QRT_QWEN36_HAWKEYE_CORRECTION_COUNT_ONLY", "1", 1);
    else unsetenv("QRT_QWEN36_HAWKEYE_CORRECTION_COUNT_ONLY");
#endif
}
void reset() {
    allocations = frees = collections = corrections = rounds = 0;
    reject_collection = fail_sync = syncs = 0;
    invalid_grid = invalid_range = false; corrected.clear();
}
int main() {
    // Independently validate the native synthetic test's closed-form dot,
    // including zero signs, against the production scalar accumulator.
    for (unsigned int k : {16u, 2048u}) {
        std::vector<uint16_t> left(k, 0u), right(k, 0u);
        for (int r = -6; r <= 6; ++r) for (int t = -8; t <= 8; ++t) {
            const float a = r / 8.0f, b = t / 16.0f;
            uint32_t ab = 0u, bb = 0u;
            std::memcpy(&ab, &a, 4u); std::memcpy(&bb, &b, 4u);
            left.back() = static_cast<uint16_t>(ab >> 16u);
            right.back() = static_cast<uint16_t>(bb >> 16u);
            const float actual = qrt_q1_moe_hawkeye::dot_bf16_impl<26, 16, -133>(
                left.data(), right.data(), k);
            float expected = a * b;
            if (expected == 0.0f) expected = 0.0f;
            if (std::memcmp(&actual, &expected, 4u) != 0) return 10;
        }
    }
    const unsigned int rows = 129u, tokens = 1031u;
    total_elements = static_cast<size_t>(rows) * tokens;
    std::vector<float> initial(total_elements, 1.001f);
    for (size_t i = 0; i < total_elements; i += 64u) initial[i] = 1.00390625f;
    initial.back() = 1.00390625f;
    uint16_t value = 0u;
    auto invoke = [&](std::vector<float> &output) {
        return launch_selected_bf16_projection_hawkeye_midpoint_correction(
            &value, &value, nullptr, nullptr, nullptr, output.data(), rows, tokens,
            2048u, 512u, 0u, 0u, 999u, nullptr);
    };
    auto output = initial;
    if (invoke(output) != hipSuccess || collections != 3u || rounds != 3u) return 1;
    for (size_t i = 0; i < total_elements; ++i) {
        const float expected = initial[i] == 1.00390625f
            ? static_cast<float>((i / rows) * 2u + i % rows) : 1.0f;
        if (output[i] != expected) return 2;
    }
    std::sort(corrected.begin(), corrected.end());
    if (std::adjacent_find(corrected.begin(), corrected.end()) != corrected.end()) return 3;
    if (invalid_grid || invalid_range || allocations != 1u || frees != 1u ||
        scratch_bytes != (131072u + 2u) * sizeof(unsigned int)) return 4;
    reset(); count_only(true); output = initial;
    if (invoke(output) != hipErrorInvalidConfiguration || collections != 3u ||
        rounds || corrections || output != initial || allocations != frees) return 5;
    count_only(false); reset(); reject_collection = 2u; output = initial;
    if (invoke(output) != hipErrorInvalidConfiguration || collections != 2u ||
        rounds != 1u || allocations != frees) return 6;
    for (size_t i = 65536u; i < total_elements; ++i) if (output[i] != initial[i]) return 7;
    reset(); fail_sync = 2u; output = initial;
    if (invoke(output) != hipErrorUnknown || collections != 1u || rounds ||
        corrections || allocations != frees) return 8;
    reset(); fail_sync = 1u; output = initial;
    if (invoke(output) != hipErrorUnknown || collections || rounds ||
        corrections || allocations || frees || output != initial) return 11;
    reset();
    if (launch_selected_bf16_projection_hawkeye_midpoint_correction(
        &value, &value, nullptr, nullptr, nullptr, output.data(), UINT32_MAX,
        2u, 2048u, 512u, 0u, 0u, 8u, nullptr) != hipErrorInvalidValue || allocations || syncs) return 9;
    return 0;
}
