"""Run the real SM121 router submission and event code with a HIP recorder."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Q1RouterProfileTests(unittest.TestCase):
    def test_sm121_router_events_follow_the_actual_stream_and_stop_on_fault(self):
        provider = (ROOT / "native/providers/whole_provider.cpp").read_text()
        enum = "enum class Q1LayerProfileBoundary" + provider.split(
            "enum class Q1LayerProfileBoundary", 1
        )[1].split("enum class Q1CoarseDecodeGpuSpanEvent", 1)[0]
        helper = "hipError_t record_qwen36_resident_decode_q1_layer_profile_boundary(" + provider.split(
            "hipError_t record_qwen36_resident_decode_q1_layer_profile_boundary(", 1
        )[1].split("bool qwen36_resident_decode_coarse_gpu_spans_active(", 1)[0]
        callback = "auto record_q1_moe_router_profile_boundary" + provider.split(
            "auto record_q1_moe_router_profile_boundary", 1
        )[1].split("bool paired_moe_projection_prepared =", 1)[0]
        router = provider.split("const uint64_t router_start_ns = qrt_now_ns();", 1)[1]
        router = router.split("} else if (paired_tail_only || paired_moe_router_prepared)", 1)[0] + "}"
        source = r'''
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
enum hipError_t { hipSuccess, hipErrorInvalidValue };
using hipStream_t = void *;
struct Event { unsigned int boundary; bool recorded = false; };
using hipEvent_t = Event *;
constexpr size_t QRT_QWEN36_LAYER_COUNT = 40;
constexpr size_t QRT_QWEN36_EXPERT_COUNT = 256;
constexpr size_t kQ1LayerProfileBoundaryCount = 19;
constexpr size_t kQ1MoePriorityEventSlotCount = 38;
struct Qwen36ResidentDecodeActivationWorkspace {
    std::array<std::array<hipEvent_t, 38>, 19> q1_layer_profile_events{};
};
bool cached = false, profiling = false;
unsigned int fail_launch = 0, fail_event = 0, launches = 0;
std::vector<unsigned int> trace;
hipStream_t expected_stream = reinterpret_cast<void *>(1234);
bool q1_decode_control_plane_cache_enabled() { return cached; }
bool env_flag_enabled(const char *) { return profiling; }
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    assert(stream == expected_stream);
    trace.push_back(event->boundary);
    if (fail_event == event->boundary) return hipErrorInvalidValue;
    event->recorded = true;
    return hipSuccess;
}
bool check_hip(hipError_t status, const char *, std::string *, std::string *) {
    return status == hipSuccess;
}
struct Lease { bool invalid = false; void invalidate() { invalid = true; } };
struct dim3 { dim3(unsigned int) {} };
constexpr unsigned int f32_to_bf16_kernel = 101;
namespace qrt_sm121_q1_moe {
template<unsigned int> constexpr unsigned int projection = 102;
constexpr unsigned int router = 103;
}
template<class... Args>
void launch(unsigned int kernel, dim3, dim3, int, hipStream_t stream, Args...) {
    assert(stream == expected_stream);
    ++launches;
    trace.push_back(kernel);
}
#define hipLaunchKernelGGL(...) launch(__VA_ARGS__)
bool check_launch(const char *) { return launches != fail_launch; }
''' + enum + helper + r'''
bool run(Qwen36ResidentDecodeActivationWorkspace *workspace, unsigned int layer, Lease *lease) {
    struct { unsigned int layer_index; } descriptor{layer};
    std::string stage, error;
    std::string *failure_stage = &stage, *failure = &error;
    hipStream_t q1_moe_router_stream = expected_stream;
    bool q1_sm121_moe_requested = true;
    float *device_post_attention = nullptr;
    uint16_t *device_input_bf16 = nullptr, *router_weights_row_major = nullptr;
    uint16_t *device_router_logits_bf16 = nullptr;
    unsigned int *device_topk_ids = nullptr;
    float *device_topk_weights = nullptr;
    constexpr size_t hidden_elements = 2048;
    struct { float *router = nullptr; } q1_sm121_moe_tables;
''' + callback + router + r'''
    return true;
}
int main() {
    for (bool cache : {false, true}) for (bool profile : {false, true})
    for (unsigned int layer : {0u, 1u, 2u, 39u})
    for (unsigned int fault : {0u, 1u, 2u, 17u, 18u}) {
        cached = cache; profiling = profile;
        fail_launch = fault <= 2 ? fault : 0;
        fail_event = fault > 2 ? fault : 0;
        launches = 0; trace.clear();
        Event input{17}, projection{18};
        Qwen36ResidentDecodeActivationWorkspace workspace;
        bool events_active = profile && layer >= 2;
        if (events_active) {
            workspace.q1_layer_profile_events[17][layer - 2] = &input;
            workspace.q1_layer_profile_events[18][layer - 2] = &projection;
        }
        Lease lease;
        bool result = run(&workspace, layer, &lease);
        bool event_fault = events_active && fault > 2;
        assert(result == (fault == 0 || (fault > 2 && !events_active)));
        assert(lease.invalid == event_fault);
        std::vector<unsigned int> expected{101};
        if (fault != 1) {
            if (events_active) expected.push_back(17);
            if (!(events_active && fault == 17)) {
                expected.push_back(102);
                if (fault != 2) {
                    if (events_active) expected.push_back(18);
                    if (!(events_active && fault == 18)) expected.push_back(103);
                }
            }
        }
        assert(trace == expected);
        if (result && events_active) assert(input.recorded && projection.recorded);
        if (!events_active) assert(!input.recorded && !projection.recorded);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-q1-router-profile-") as tmp:
            exe = str(Path(tmp) / "profile-test")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                 "-Werror", "-x", "c++", "-", "-o", exe],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([exe], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
