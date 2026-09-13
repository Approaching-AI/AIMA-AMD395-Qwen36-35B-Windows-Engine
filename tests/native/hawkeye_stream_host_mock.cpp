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
#include "moe_accumulator/sm121_prefill_projection.h"

enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorInvalidConfiguration, hipErrorUnknown };
using hipStream_t = void *;
constexpr int hipMemcpyDeviceToHost = 0;
constexpr unsigned int kSelectedHawkeyeCorrectionThreads = 256u;
#ifndef QRT_PREFILL_HAWKEYE_REPLAY_LANES
#define QRT_PREFILL_HAWKEYE_REPLAY_LANES 16
#endif
constexpr unsigned int kSelectedHawkeyeReplayLanes = QRT_PREFILL_HAWKEYE_REPLAY_LANES;
constexpr unsigned int kSelectedHawkeyeCorrectionMaximumBlocksPerLaunchLimit =
    qrt_hawkeye_dispatch::maximum_exact_blocks;
constexpr unsigned int kThreads = 256u;
struct dim3 { unsigned int x, y; explicit dim3(unsigned int value, unsigned int second = 1u) : x(value), y(second) {} };
static unsigned int allocations = 0, frees = 0, collections = 0, corrections = 0, rounds = 0;
static unsigned int reject_collection = 0, fail_sync = 0, syncs = 0;
static unsigned int exact_blocks = 0;
static unsigned int requested_blocks = 8u;
static unsigned int count_reads = 0u;
static size_t scratch_bytes = 0;
static bool invalid_grid = false, invalid_range = false;
static std::vector<size_t> corrected;
static size_t total_elements = 0;
static float *tracked_output = nullptr;
static const float *tracked_sums = nullptr, *tracked_input_bounds = nullptr, *tracked_weight_bounds = nullptr;
static const uint16_t *tracked_inputs = nullptr;
static unsigned tracked_prefix = 0, tracked_k = 0;
static size_t shape_aware_outputs = 0;
static unsigned preparations = 0, fail_preparation = 0, fail_allocation = 0;
static bool preparation_fault = false;
static uint16_t* prepared_buffers[2]{};
static unsigned* prepared_flags[2]{};
static unsigned prepared_rows[2]{};

hipError_t hipMalloc(void **p, size_t bytes) {
    ++allocations; scratch_bytes = bytes;
    if (allocations == fail_allocation) { *p = nullptr; return hipErrorUnknown; }
    *p = std::malloc(bytes);
    return *p ? hipSuccess : hipErrorUnknown;
}
hipError_t hipFree(void *p) { ++frees; std::free(p); return hipSuccess; }
hipError_t hipMemsetAsync(void *p, int value, size_t n, hipStream_t) {
    std::memset(p, value, n); return hipSuccess;
}
hipError_t hipMemcpy(void *to, const void *from, size_t n, int) {
    ++count_reads;
    std::memcpy(to, from, n); return hipSuccess;
}
hipError_t hipGetLastError() {
    const bool failed = preparation_fault; preparation_fault = false;
    return failed ? hipErrorUnknown : hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
    return ++syncs == fail_sync ? hipErrorUnknown : hipSuccess;
}
void grid(const char *name, dim3 blocks, dim3 threads) {
    if (std::strstr(name, "transpose_weights") != nullptr) {
        if (!blocks.x || !blocks.y || threads.x != 32u || threads.y != 8u) invalid_grid = true;
        return;
    }
    const bool exact = std::strstr(name, "midpoint_correction") != nullptr ||
        std::strstr(name, "prepared_correction") != nullptr ||
        std::strstr(name, "packed_correction") != nullptr;
    const unsigned int limit = std::strstr(name, "device_correction") != nullptr
        ? qrt_hawkeye_dispatch::maximum_device_replay_blocks : exact
        ? (std::min)(requested_blocks, qrt_hawkeye_dispatch::maximum_exact_blocks)
        : qrt_hawkeye_dispatch::maximum_window_elements / 256u;
    if (exact) exact_blocks = blocks.x;
    if (blocks.x == 0 || blocks.x > limit || threads.x != 256u) invalid_grid = true;
}
#define hipLaunchKernelGGL(kernel, blocks, threads, shared, stream, ...) \
    do { grid(#kernel, blocks, threads); kernel(__VA_ARGS__); } while (0)

template<bool RoundOutputs>
void selected_bf16_projection_hawkeye_compact_kernel(
    const float *sums, const float *input_bounds, const float *weight_bounds, float *output,
    unsigned int rows, unsigned int, unsigned int prefix, unsigned int,
    unsigned int *counts, unsigned int *indices, size_t offset, unsigned int count
) {
    ++collections;
    if (tracked_output) {
        const size_t base = static_cast<size_t>(output - tracked_output);
        if (base != 0u && base != static_cast<size_t>(tracked_prefix) * rows) invalid_range = true;
        if (sums != tracked_sums + base || input_bounds != tracked_input_bounds + base / rows ||
            weight_bounds != tracked_weight_bounds || base + offset + count > total_elements) invalid_range = true;
    }
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
    if (collections == reject_collection) counts[0] = qrt_hawkeye_dispatch::maximum_candidates + 1u;
    if constexpr (RoundOutputs) {
        ++rounds;
        for (unsigned j = 0; j < count; ++j) output[offset + j] = 1.0f;
    }
}
void round_f32_outputs_to_bf16_kernel(float *output, unsigned int count) {
    ++rounds;
    for (unsigned int i = 0; i < count; ++i) output[i] = 1.0f;
}
template<bool ShapeAware>
void selected_bf16_projection_hawkeye_midpoint_correction_kernel(
    const uint16_t *, const uint16_t *inputs, float *output, unsigned int rows,
    unsigned int, const unsigned int *indices, unsigned int offset, unsigned int count,
    qrt_sm121_prefill_projection::Plan plan
) {
    if (ShapeAware != qrt_sm121_prefill_projection::changes_dot(plan)) invalid_range = true;
    ++corrections;
    const unsigned int end = (std::min)(count, offset + exact_blocks * (256u / kSelectedHawkeyeReplayLanes));
    if (tracked_output) {
        const size_t base = static_cast<size_t>(output - tracked_output);
        if (inputs != tracked_inputs + base / rows * tracked_k ||
            ShapeAware != (base != 0u)) invalid_range = true;
        if (ShapeAware) shape_aware_outputs += end - offset;
    }
    for (unsigned int j = offset; j < end; ++j) {
        const size_t index = indices[j];
        if (index >= total_elements) { invalid_range = true; continue; }
        corrected.push_back(index);
        output[index] = static_cast<float>((index / rows) * 2u + index % rows);
    }
}

void selected_hawkeye_transpose_weights_kernel(const uint16_t*, uint16_t*, unsigned int, unsigned int) {}
template<bool ShapeAware>
void selected_bf16_projection_hawkeye_device_correction_kernel(
    const uint16_t *, const uint16_t *inputs, float *output, unsigned int rows,
    unsigned int, const unsigned int *counts, const unsigned int *indices,
    unsigned int window, qrt_sm121_prefill_projection::Plan plan
) {
    ++corrections;
    if (ShapeAware != qrt_sm121_prefill_projection::changes_dot(plan) || counts[0] > window)
        invalid_range = true;
    if (tracked_output) {
        const size_t base = static_cast<size_t>(output - tracked_output);
        if (inputs != tracked_inputs + base / rows * tracked_k ||
            ShapeAware != (base != 0u)) invalid_range = true;
        if (ShapeAware) shape_aware_outputs += counts[0];
    }
    for (unsigned j = 0; j < counts[0]; ++j) {
        const size_t index = indices[j];
        if (index >= total_elements) { invalid_range = true; continue; }
        corrected.push_back(index);
        output[index] = static_cast<float>((index / rows) * 2u + index % rows);
    }
}
void selected_bf16_projection_hawkeye_packed_correction_kernel(
    const uint16_t*, const uint16_t*, float* output, unsigned int rows,
    unsigned int, const unsigned int* indices, unsigned int offset, unsigned int count
) {
    ++corrections;
    const unsigned int end = (std::min)(count, offset + exact_blocks * 256u);
    for (unsigned int j = offset; j < end; ++j) {
        const size_t index = indices[j];
        if (index >= total_elements) { invalid_range = true; continue; }
        corrected.push_back(index);
        output[index] = static_cast<float>((index / rows) * 2u + index % rows);
    }
}

namespace qrt_sm121_prepared_projection {
void prepare_rows_kernel(const uint16_t*,uint16_t* encoded,unsigned* eligible,unsigned rows,unsigned width) {
    if (preparations >= 2u) { invalid_range = true; return; }
    prepared_buffers[preparations] = encoded; prepared_flags[preparations] = eligible;
    prepared_rows[preparations] = rows;
    if (++preparations == 2u && (encoded != prepared_buffers[0] + size_t(prepared_rows[0])*width ||
        eligible != prepared_flags[0] + prepared_rows[0] ||
        prepared_flags[0] != reinterpret_cast<unsigned*>(prepared_buffers[0] + size_t(prepared_rows[0]+rows)*width)))
        invalid_range = true;
    preparation_fault = preparations == fail_preparation;
}
}
void selected_bf16_projection_hawkeye_prepared_correction_kernel(
    const uint16_t* weights,const uint16_t* inputs,const uint16_t* pw,const uint16_t* px,
    const unsigned* wf,const unsigned* xf,float* output,unsigned rows,unsigned k,
    const unsigned* indices,unsigned offset,unsigned count) {
    if (preparations != 2u || pw != prepared_buffers[0] || px != prepared_buffers[1] ||
        wf != prepared_flags[0] || xf != prepared_flags[1]) invalid_range = true;
    selected_bf16_projection_hawkeye_midpoint_correction_kernel<false>(
        weights,inputs,output,rows,k,indices,offset,count,{});
}

// QRT_ACTUAL_LAUNCHER

void prepared_mode(const char* value) {
#ifdef _WIN32
    _putenv_s("QRT_QWEN36_HAWKEYE_PREPARED_OPERANDS",value);
#else
    setenv("QRT_QWEN36_HAWKEYE_PREPARED_OPERANDS",value,1);
#endif
}
void count_only(bool enabled) {
#ifdef _WIN32
    _putenv_s("QRT_QWEN36_HAWKEYE_CORRECTION_COUNT_ONLY", enabled ? "1" : "");
#else
    if (enabled) setenv("QRT_QWEN36_HAWKEYE_CORRECTION_COUNT_ONLY", "1", 1);
    else unsetenv("QRT_QWEN36_HAWKEYE_CORRECTION_COUNT_ONLY");
#endif
}
void packed_mode(bool enabled) {
#ifdef _WIN32
    _putenv_s("QRT_QWEN36_HAWKEYE_PACKED_CANDIDATES", enabled ? "1" : "");
#else
    if (enabled) setenv("QRT_QWEN36_HAWKEYE_PACKED_CANDIDATES", "1", 1);
    else unsetenv("QRT_QWEN36_HAWKEYE_PACKED_CANDIDATES");
#endif
}
void device_mode(bool enabled) {
#ifdef _WIN32
    _putenv_s("QRT_QWEN36_HAWKEYE_DEVICE_REPLAY", enabled ? "1" : "");
#else
    if (enabled) setenv("QRT_QWEN36_HAWKEYE_DEVICE_REPLAY", "1", 1);
    else unsetenv("QRT_QWEN36_HAWKEYE_DEVICE_REPLAY");
#endif
}
void reset() {
    allocations = frees = collections = corrections = rounds = 0;
    reject_collection = fail_sync = syncs = 0;
    invalid_grid = invalid_range = false; corrected.clear();
    shape_aware_outputs = 0;
    count_reads = 0;
    preparations = fail_preparation = fail_allocation = 0;
    preparation_fault = false;
}
int main() {
    device_mode(false);
    packed_mode(false);
    prepared_mode("0");
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
            2048u, 512u, 0u, 0u, requested_blocks, nullptr, 65536u);
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
        scratch_bytes != (65536u + 2u) * sizeof(unsigned int)) return 4;
    // Fully dense source blocks remain bounded after compaction. The actual
    // kernel has at most 64 candidate subgroups per CTA, within the cap.
    for (unsigned int cap : {8u, 64u, 999u, 4096u, UINT32_MAX}) {
        requested_blocks = cap;
        reset(); output.assign(total_elements, 1.00390625f);
        if (invoke(output) != hipSuccess || collections != 3u || rounds != 3u ||
            corrected.size() != total_elements || invalid_grid || allocations != frees) return 14;
        for (size_t i = 0; i < total_elements; ++i) {
            if (output[i] != static_cast<float>((i / rows) * 2u + i % rows)) return 15;
        }
        std::sort(corrected.begin(), corrected.end());
        if (std::adjacent_find(corrected.begin(), corrected.end()) != corrected.end()) return 16;
        if (cap >= 4096u && corrections != collections) return 17;
    }
    // A production-sized collection window gathers this whole projection at
    // once while exact dispatches still obey their independent work quantum.
    reset(); output = initial;
    if (launch_selected_bf16_projection_hawkeye_midpoint_correction(
        &value, &value, nullptr, nullptr, nullptr, output.data(), rows, tokens,
        2048u, 512u, 0u, 0u, 4096u, nullptr) != hipSuccess || collections != 1u ||
        rounds != 1u || scratch_bytes != (total_elements + 2u) * sizeof(unsigned int) ||
        invalid_grid || invalid_range || allocations != frees) return 18;
    for (size_t i = 0; i < total_elements; ++i) {
        const float expected = initial[i] == 1.00390625f
            ? static_cast<float>((i / rows) * 2u + i % rows) : 1.0f;
        if (output[i] != expected) return 19;
    }
    requested_blocks = 8u;
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
    for (unsigned int capacity : {0u, qrt_hawkeye_dispatch::maximum_window_elements + 1u}) {
        reset();
        if (launch_selected_bf16_projection_hawkeye_midpoint_correction(
            &value, &value, nullptr, nullptr, nullptr, output.data(), rows, tokens,
            2048u, 512u, 0u, 0u, 8u, nullptr, capacity) != hipErrorInvalidValue ||
            allocations || syncs) return 20;
    }
    packed_mode(true);
    for (unsigned int cap : {1u, 8u, 17u, 4096u}) {
        requested_blocks = cap;
        reset(); output.assign(total_elements, 1.00390625f);
        if (invoke(output) != hipSuccess || collections != 3u || rounds != 3u ||
            corrected.size() != total_elements || invalid_grid || invalid_range ||
            allocations != 2u || frees != 2u) return 21;
        for (size_t i = 0; i < total_elements; ++i)
            if (output[i] != static_cast<float>((i / rows) * 2u + i % rows)) return 22;
        std::sort(corrected.begin(), corrected.end());
        if (std::adjacent_find(corrected.begin(), corrected.end()) != corrected.end()) return 23;
    }
    reset(); fail_sync = 2u; output = initial;
    if (invoke(output) != hipErrorUnknown || collections || rounds || corrections ||
        allocations != 2u || frees != 2u || output != initial) return 24;
    reset(); count_only(true); output = initial;
    if (invoke(output) != hipErrorInvalidConfiguration || collections != 3u ||
        rounds || corrections || output != initial || allocations != 1u || frees != 1u) return 25;
    count_only(false); packed_mode(false);
    // A changed dense plan must replay outputs outside the old midpoint band,
    // retain bounded index dispatch, and bypass the unsplit packed kernel.
    for (unsigned int dense_rows : {32u, 64u, 2048u}) {
        const unsigned int dense_tokens = 19u;
        const unsigned int k = dense_rows == 2048u ? 4096u : 2048u;
        total_elements = static_cast<size_t>(dense_rows) * dense_tokens;
        for (bool packed : {false, true}) {
            packed_mode(packed); reset(); requested_blocks = 8u;
            output.assign(total_elements, 1.001f);
            if (launch_selected_bf16_projection_hawkeye_midpoint_correction(
                &value, &value, nullptr, nullptr, nullptr, output.data(), dense_rows,
                dense_tokens, k, 512u, 0u, 0u, requested_blocks, nullptr, 65536u) != hipSuccess ||
                corrected.size() != total_elements || invalid_grid || invalid_range ||
                allocations != 1u || frees != 1u) return 26;
            std::sort(corrected.begin(), corrected.end());
            for (size_t i = 0; i < total_elements; ++i)
                if (corrected[i] != i) return 27;
        }
    }
    packed_mode(false);
    // Exercise the actual long-prompt launcher with differently sized input,
    // output, and bound arrays. Only the final reference batch changes plan;
    // failure in a complete prefix must never enqueue or round its tail.
    for (const auto shape : {std::array<unsigned, 4>{32u, 2048u, 16384u, 1024u},
                            std::array<unsigned, 4>{64u, 2048u, 8192u, 19u},
                            std::array<unsigned, 4>{2048u, 4096u, 8192u, 19u}}) {
        const unsigned r = shape[0], k = shape[1], prefix = shape[2], tail = shape[3];
        total_elements = static_cast<size_t>(r) * (prefix + tail);
        std::vector<uint16_t> inputs(static_cast<size_t>(prefix + tail) * k);
        std::vector<float> sums(total_elements), bounds(prefix + tail), weight_bounds(r);
        output.assign(total_elements, 1.001f);
        tracked_output = output.data(); tracked_sums = sums.data(); tracked_inputs = inputs.data();
        tracked_input_bounds = bounds.data(); tracked_weight_bounds = weight_bounds.data();
        tracked_prefix = prefix; tracked_k = k; requested_blocks = 4096u;
        auto run = [&] {
            return launch_selected_bf16_projection_hawkeye_midpoint_correction(
                &value, inputs.data(), sums.data(), bounds.data(), weight_bounds.data(), output.data(),
                r, prefix + tail, k, 512u, 0u, 0u, requested_blocks, nullptr);
        };
        for (bool device : {false, true}) {
        device_mode(device); reset(); std::fill(output.begin(), output.end(), 1.001f);
        if (run() != hipSuccess || shape_aware_outputs != static_cast<size_t>(r) * tail ||
            corrected.size() != static_cast<size_t>(r) * tail || invalid_grid || invalid_range ||
            allocations != 2u || frees != 2u) return 28;
        for (size_t i = 0; i < static_cast<size_t>(r) * prefix; ++i)
            if (output[i] != 1.0f) return 29;
        for (size_t i = 0; i < static_cast<size_t>(r) * tail; ++i)
            if (output[static_cast<size_t>(prefix) * r + i] != static_cast<float>((i / r) * 2u + i % r)) return 30;
        if (device && count_reads) return 33;
        }
        device_mode(false);
        reset(); fail_sync = 2u; std::fill(output.begin(), output.end(), 1.001f);
        if (run() != hipErrorUnknown || corrections || rounds || allocations != frees || shape_aware_outputs)
            return 31;
        if (!std::all_of(output.begin(), output.end(), [](float x) { return x == 1.001f; })) return 32;
        tracked_output = nullptr;
    }
    // The device route selects before fused rounding, processes every index
    // once across multiple windows, performs no D2H count reads, and stops
    // later windows on a completion fault. Count-only remains nonmutating.
    device_mode(true); requested_blocks = 4096u;
    total_elements = static_cast<size_t>(rows) * tokens;
    for (unsigned mode = 0; mode < 3; ++mode) {
        reset(); output = initial;
        if (mode != 1u) std::fill(output.begin(), output.end(), mode ? 1.00390625f : 1.001f);
        const auto before = output;
        if (invoke(output) != hipSuccess || collections != 3 || rounds != 3 || corrections != 3 ||
            syncs != 4 || count_reads || invalid_grid || invalid_range || allocations != frees) return 34;
        for (size_t i = 0; i < total_elements; ++i) {
            const float expected = before[i] == 1.00390625f
                ? static_cast<float>((i / rows) * 2u + i % rows) : 1.0f;
            if (output[i] != expected) return 35;
        }
        std::sort(corrected.begin(), corrected.end());
        if (std::adjacent_find(corrected.begin(), corrected.end()) != corrected.end()) return 36;
    }
    reset(); output = initial; fail_sync = 2;
    if (invoke(output) != hipErrorUnknown || collections != 1 || corrections != 1 ||
        count_reads || allocations != frees) return 37;
    for (size_t i = 65536; i < output.size(); ++i) if (output[i] != initial[i]) return 38;
    reset(); output = initial; count_only(true);
    if (invoke(output) != hipErrorInvalidConfiguration || output != initial || rounds || corrections ||
        count_reads != 3 || allocations != frees) return 39;
    count_only(false); device_mode(false);
    prepared_mode("2"); reset(); output=initial;
    if (invoke(output)!=hipErrorInvalidValue || allocations || syncs || output!=initial) return 40;
    prepared_mode("1"); reset(); output=initial;
    if (invoke(output)!=hipSuccess || preparations || allocations!=frees || invalid_grid || invalid_range) return 41;
    total_elements=1024u*1024u;requested_blocks=4096u;
    std::vector<float> prepared_initial(total_elements,1.001f);
    for(size_t i=0;i<total_elements;i+=37u)prepared_initial[i]=1.00390625f;
    auto run_prepared=[&] {
        return launch_selected_bf16_projection_hawkeye_midpoint_correction(&value,&value,
            nullptr,nullptr,nullptr,output.data(),1024u,1024u,16u,512u,0u,0u,requested_blocks,nullptr,65536u);
    };
    reset();output=prepared_initial;
    if(run_prepared()!=hipSuccess || preparations!=2u || allocations!=2u || frees!=2u || invalid_range || invalid_grid)
        return 42;
    for(size_t i=0;i<total_elements;++i) {
        const float expected=i%37u ? 1.0f : float((i/1024u)*2u+i%1024u);
        if(output[i]!=expected)return 43;
    }
    for(unsigned failure:{1u,2u}) {
        reset();fail_preparation=failure;output=prepared_initial;
        if(run_prepared()!=hipErrorUnknown || preparations!=failure || allocations!=frees ||
            collections || rounds || corrections || syncs!=2u || output!=prepared_initial)return 44;
    }
    reset();fail_sync=2u;output=prepared_initial;
    if(run_prepared()!=hipErrorUnknown || preparations!=2u || allocations!=frees || collections ||
        rounds || corrections || syncs!=3u || output!=prepared_initial)return 45;
    reset();fail_allocation=2u;output=prepared_initial;
    if(run_prepared()!=hipErrorUnknown || preparations || allocations!=2u || frees!=1u ||
        collections || output!=prepared_initial)return 46;
    for(const auto shape:{std::array<unsigned,3>{2048u,19u,4096u},{1024u,1u,16u},{1024u,8193u,16u}}) {
        reset();total_elements=size_t(shape[0])*shape[1];output.assign(total_elements,1.001f);
        if(launch_selected_bf16_projection_hawkeye_midpoint_correction(&value,&value,nullptr,nullptr,nullptr,
            output.data(),shape[0],shape[1],shape[2],512u,0u,0u,4096u,nullptr)!=hipSuccess ||
            preparations || allocations!=frees || invalid_grid || invalid_range)return 47;
    }
    prepared_mode("0");
    return 0;
}
