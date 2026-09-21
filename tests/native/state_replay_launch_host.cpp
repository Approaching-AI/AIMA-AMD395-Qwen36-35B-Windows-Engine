#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <vector>
#include "native/providers/gdn/state_replay_policy.h"
#include "native/providers/gdn/fla_checkpoint.h"
#include "native/providers/gdn/sm121_exp2_table.h"

enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t = void*;
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a = 1u, unsigned b = 1u, unsigned c = 1u): x(a), y(b), z(c) {}
};
namespace qrt_fla_separate_state {
template<unsigned> void fast_kernel() {}
template<unsigned> void replay_kernel() {}
}
namespace qrt_fla_hybrid_state { template<unsigned> void retry_kernel() {} }

// Addresses represent disjoint device allocations; host code must never
// dereference them. Each gap exceeds all of the checked device spans.
const void* buffers[8];
unsigned* receipts;
hipStream_t expected_stream = reinterpret_cast<void*>(uintptr_t(0x395u));
unsigned expected_count, fail_at, launches, queries, checked;
std::vector<void(*)()> submitted;
void reset(unsigned count, unsigned failure = 0u) {
    for (unsigned i = 0u; i < 8u; ++i)
        buffers[i] = reinterpret_cast<void*>(uintptr_t(i + 1u) * 0x20000000u);
    receipts = reinterpret_cast<unsigned*>(uintptr_t(9u) * 0x20000000u);
    expected_count = count; fail_at = failure; launches = queries = 0u; submitted.clear();
}
void record(void(*kernel)(), dim3 grid, dim3 block, unsigned shared, hipStream_t stream,
            const uint16_t* k, const uint16_t* u, const uint16_t* w, const float* g,
            uint16_t* h, uint16_t* v_new, float* state, unsigned count,
            const unsigned char* table, unsigned* flags) {
    ++launches; submitted.push_back(kernel);
    assert(grid.x == 16u && grid.y == 32u && grid.z == 1u);
    assert(block.x == 256u && block.y == 1u && block.z == 1u && !shared);
    assert(stream == expected_stream && count == expected_count && flags == receipts);
    const void* actual[] = {k, u, w, g, h, v_new, state, table};
    for (unsigned i = 0u; i < 8u; ++i) assert(actual[i] == buffers[i]);
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel, ...) record(kernel, __VA_ARGS__)
hipError_t hipGetLastError() { ++queries; return queries == fail_at ? hipErrorUnknown : hipSuccess; }
namespace qrt_fla_blackwell_cooperative {
constexpr unsigned threads = 256u;
#include "state_replay_wrapper_under_test.h"
bool enabled() { return true; }
unsigned checkpoint_calls = 0u;
hipError_t state_checkpoints(const uint16_t*, const uint16_t*, const uint16_t*, const float*,
    uint16_t*, uint16_t*, float* state, unsigned count, const unsigned char*, hipStream_t stream,
    qrt_fla_checkpoint::Segment checkpoint) {
    assert(checkpoint.count && state == buffers[6] && count == expected_count && stream == expected_stream);
    ++checkpoint_calls; return fail_at ? hipErrorUnknown : hipSuccess;
}
}

unsigned ensure_calls = 0u, original_calls = 0u, math_calls = 0u, copies = 0u;
bool scratch_ok = true, caller_error = false;
struct { float* blackwell_temporary_state; uint16_t* blackwell_residual; } g_state;
struct BlackwellSegmentGuard { static void* active; };
void* BlackwellSegmentGuard::active = reinterpret_cast<void*>(uintptr_t(1u));
constexpr int32_t kSegmentTokens = 1024;
constexpr uint32_t kChunk = 64u, kValueFeatures = 4096u, kStateElements = 524288u;
constexpr uint32_t kValueHeads = 32u, kQkHeads = 16u, kKeyDim = 128u;
void set_error_text(const char*) { caller_error = true; }
void set_error(const char*, hipError_t) { caller_error = true; }
bool ensure_blackwell_state_scratch() { ++ensure_calls; return scratch_ok; }
bool blackwell_batched_enabled() {
    return qrt_fla_state_replay_policy::setting("QRT_FLA_GDN_BATCHED_EXACT", "1");
}
template<class Operation>
bool launch_blackwell_math(const char*, hipStream_t stream, Operation operation, float*) {
    ++math_calls; assert(stream == expected_stream); return operation() == hipSuccess;
}
namespace qrt_fla_blackwell_state {
const unsigned char* exp2_table_device() { return static_cast<const unsigned char*>(buffers[7]); }
hipError_t segment(const uint16_t*, const uint16_t*, const uint16_t*, const float*,
    uint16_t*, uint16_t*, float* state, unsigned count, hipStream_t stream) {
    assert(state == buffers[6] && count == expected_count && stream == expected_stream);
    ++original_calls; return fail_at ? hipErrorUnknown : hipSuccess;
}
hipError_t project(const uint16_t*, const uint16_t*, const float*, const float*,
    uint16_t*, uint16_t*, uint16_t*, unsigned, hipStream_t) { ++original_calls; return hipSuccess; }
hipError_t update(const uint16_t*, const uint16_t*, const float*, const float*, float*,
    unsigned, hipStream_t) { ++original_calls; return hipSuccess; }
}
constexpr int hipMemcpyDeviceToDevice = 1;
hipError_t hipMemcpyAsync(void*, const void*, size_t, int, hipStream_t) { ++copies; return hipSuccess; }
hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
#include "state_replay_caller_under_test.h"

#define STATE_ARGS static_cast<const uint16_t*>(buffers[0]), static_cast<const uint16_t*>(buffers[1]), static_cast<const uint16_t*>(buffers[2]), static_cast<const float*>(buffers[3]), static_cast<uint16_t*>(const_cast<void*>(buffers[4])), static_cast<uint16_t*>(const_cast<void*>(buffers[5])), static_cast<float*>(const_cast<void*>(buffers[6]))
hipError_t invoke(int mode) {
    ++checked;
    return qrt_fla_blackwell_cooperative::state_replay(STATE_ARGS, expected_count,
        static_cast<const unsigned char*>(buffers[7]), expected_stream, receipts, mode);
}
const char* names[] = {"QRT_FLA_GDN_STATE_BLACKWELL", "QRT_FLA_GDN_BATCHED_EXACT",
    "QRT_FLA_GDN_COOPERATIVE_EXACT", "QRT_FLA_GDN_SCALAR_FLOAT_MATRICES",
    "QRT_FLA_GDN_SCALAR_FLOAT_STATE", "QRT_FLA_GDN_PAIRED_SCORE_ARENAS",
    "QRT_FLA_GDN_COARSE_INTERVAL", "QRT_FLA_GDN_FUSED_STATE_OUTPUT"};
const char* settings[] = {"1", "1", "1", "1", "8", "1", "0", "0"};
void compatible() { for (unsigned i = 0u; i < 8u; ++i) setenv(names[i], settings[i], 1); }
void order(int mode, unsigned failure) {
    const unsigned count = unsigned(mode + 1);
    const bool failed = failure && failure <= count;
    assert(launches == (failed ? failure : count) && queries == launches);
    assert(submitted[0] == qrt_fla_separate_state::fast_kernel<8u>);
    if (launches > 1u) assert(submitted[1] == (mode == 2 ?
        qrt_fla_hybrid_state::retry_kernel<8u> : qrt_fla_separate_state::replay_kernel<8u>));
    if (launches > 2u) assert(submitted[2] == qrt_fla_separate_state::replay_kernel<8u>);
}
int main() {
    using namespace qrt_fla_state_replay_policy;
    unsetenv("QRT_FLA_GDN_STATE_REPLAY"); assert(mode() == 0);
    for (const char* value : {"", "0", "1", "2", "3", "01", "2 ", "-1", "true"}) {
        setenv("QRT_FLA_GDN_STATE_REPLAY", value, 1);
        assert(mode() == (!*value || !std::strcmp(value, "0") ? 0 :
            !std::strcmp(value, "1") ? 1 : !std::strcmp(value, "2") ? 2 : -1));
    }
    compatible();
    for (int selection : {-1, 0, 1, 2, 3}) for (unsigned checkpoints : {0u, 1u, 3u})
    for (unsigned count : {0u, 1u, 63u, 64u, 65u, 1023u, 1024u, 1025u, UINT32_MAX})
        assert(selected(selection, checkpoints, count) ==
            ((selection == 1 || selection == 2) && !checkpoints && count && count <= 1024u));
    for (unsigned option = 0u; option < 8u; ++option) {
        for (const char* value : {"", "invalid", "1", "0", "8"}) {
            compatible(); setenv(names[option], value, 1);
            const bool matches = !std::strcmp(value, settings[option]) || (option >= 6u && !*value);
            assert(selected(2, 0u, 1024u) == matches);
        }
        compatible(); unsetenv(names[option]); assert(selected(1, 0u, 64u) == (option >= 6u));
    }
    for (int selection : {1, 2}) for (unsigned count : {1u, 63u, 64u, 65u, 1023u, 1024u})
    for (unsigned failure = 0u; failure <= 3u; ++failure) {
        reset(count, failure); const auto result = invoke(selection);
        assert(result == (failure && failure <= unsigned(selection + 1) ? hipErrorUnknown : hipSuccess));
        order(selection, failure);
    }
    for (int selection : {-1, 0, 3}) { reset(64u); assert(invoke(selection) == hipErrorInvalidValue && !launches && !queries); }
    for (unsigned count : {0u, 1025u, UINT32_MAX}) { reset(count); assert(invoke(2) == hipErrorInvalidValue && !launches); }
    for (unsigned field = 0u; field < 8u; ++field) {
        reset(1024u); buffers[field] = nullptr; assert(invoke(2) == hipErrorInvalidValue && !launches);
        reset(1024u); buffers[field] = reinterpret_cast<void*>(UINTPTR_MAX - 3u);
        assert(invoke(2) == hipErrorInvalidValue && !launches);
        reset(1024u); receipts = reinterpret_cast<unsigned*>(reinterpret_cast<uintptr_t>(buffers[field]) + 4u);
        assert(invoke(2) == hipErrorInvalidValue && !launches);
        reset(1024u); receipts = reinterpret_cast<unsigned*>(reinterpret_cast<uintptr_t>(buffers[field]) - 4u);
        assert(invoke(2) == hipErrorInvalidValue && !launches);
    }
    for (uintptr_t pointer : {uintptr_t(0u), uintptr_t(1u), UINTPTR_MAX - 3u}) {
        reset(1024u); receipts = reinterpret_cast<unsigned*>(pointer); assert(invoke(2) == hipErrorInvalidValue && !launches);
    }
    for (unsigned destination : {4u, 5u, 6u}) for (unsigned input : {0u, 1u, 2u, 3u, 6u}) {
        if (destination == input) continue;
        reset(65u); buffers[destination] = buffers[input]; assert(invoke(2) == hipErrorInvalidValue && !launches);
    }
    reset(65u); buffers[4] = buffers[5]; assert(invoke(2) == hipErrorInvalidValue && !launches);
    unsigned caller_cases = 0u;
    for (const char* setting : {"0", "1", "2", "invalid"})
    for (unsigned count : {1u, 63u, 64u, 65u, 1024u}) for (unsigned checkpoints : {0u, 1u})
    for (bool storage : {false, true}) for (unsigned failure = 0u; failure <= 3u; ++failure) {
        reset(count, failure); compatible(); setenv("QRT_FLA_GDN_STATE_REPLAY", setting, 1);
        g_state.blackwell_temporary_state = reinterpret_cast<float*>(receipts);
        scratch_ok = storage; ensure_calls = original_calls = math_calls = copies = 0u; caller_error = false;
        qrt_fla_blackwell_cooperative::checkpoint_calls = 0u;
        qrt_fla_checkpoint::Segment checkpoint{}; checkpoint.count = checkpoints;
        const int selected_mode = mode(); const int padded = int((count + 63u) / 64u * 64u);
        const bool ok = launch_blackwell_state(STATE_ARGS, padded, expected_stream, int(count), checkpoint);
        ++caller_cases;
        assert(!copies);
        if (selected_mode < 0) { assert(!ok && caller_error && !ensure_calls && !math_calls && !launches); continue; }
        assert(ensure_calls == 1u);
        if (!storage) { assert(!ok && !math_calls && !launches && !original_calls); continue; }
        assert(math_calls == 1u);
        if (checkpoints) {
            assert(ok == !failure && qrt_fla_blackwell_cooperative::checkpoint_calls == 1u && !launches && !original_calls);
        } else if (!selected_mode) {
            assert(ok == !failure && original_calls == 1u && !launches);
        } else {
            assert(ok == !(failure && failure <= unsigned(selected_mode + 1)) && !original_calls);
            order(selected_mode, failure);
        }
    }
    std::printf("state_replay_launch_host=pass wrapper_cases=%u caller_cases=%u receipt_storage_reused=1 checkpoint_fallback=1 launch_failure_stops_chain=1\n", checked, caller_cases);
}
