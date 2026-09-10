"""Exercise streamed dispatch, admission and faults without submitting GPU work."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HawkeyeDispatchPolicyTests(unittest.TestCase):
    def test_real_launch_guards_reject_before_any_hip_call(self):
        provider = (ROOT / "native/providers/whole_provider.cpp").read_text()
        guards = []
        for start, end in (
            ("hipError_t launch_projection_f32_to_bf16_checked(",
             "__global__ void packed_full_attention_kv_f32_to_bf16_kernel("),
            ("hipError_t launch_selected_bf16_projection_wmma_checked(",
             "// Cauchy-Schwarz gives"),
        ):
            guards.append(provider.split(start, 1)[1].split(end, 1)[0])
            guards[-1] = start + guards[-1]
        source = r'''
#include <cstddef>
#include <cstdint>
#include "projection_output_policy.h"
enum hipError_t { hipSuccess, hipErrorInvalidValue };
using hipStream_t = void *;
struct dim3 { dim3(size_t, size_t = 1) {} };
static unsigned int launches = 0, last_error_calls = 0;
constexpr int f32_to_bf16_kernel = 0;
constexpr int selected_bf16_projection_wmma_k16_m64_kernel = 0;
template<class... T> void record_launch(T...) { ++launches; }
#define hipLaunchKernelGGL(...) record_launch(__VA_ARGS__)
hipError_t hipGetLastError() { ++last_error_calls; return hipSuccess; }
''' + "\n".join(guards) + r'''
int main() {
    uint16_t x = 0; float y = 0;
    for (unsigned int i = 0; i < 6; ++i) {
        if (launch_selected_bf16_projection_wmma_checked(
            i == 0 ? nullptr : &x, i == 1 ? nullptr : &x,
            i == 2 ? nullptr : &y, i == 3 ? 0u : 128u,
            i == 4 ? 0u : (i == 5 ? UINT32_MAX : 64u), 0, 0, nullptr
        ) != hipErrorInvalidValue) return 1;
    }
    if (launch_projection_f32_to_bf16_checked(nullptr, &x, 1, nullptr) != hipErrorInvalidValue) return 2;
    if (launch_projection_f32_to_bf16_checked(&y, nullptr, 1, nullptr) != hipErrorInvalidValue) return 3;
    if (launch_projection_f32_to_bf16_checked(&y, &x, 0, nullptr) != hipErrorInvalidValue) return 4;
    if (launch_projection_f32_to_bf16_checked(&y, &x, SIZE_MAX, nullptr) != hipErrorInvalidValue) return 5;
    if (launches || last_error_calls) return 6;
    if (launch_selected_bf16_projection_wmma_checked(&x, &x, &y, 128, 65, 0, 0, nullptr) != hipSuccess) return 7;
    if (launch_projection_f32_to_bf16_checked(&y, &x, 128 * 65, nullptr) != hipSuccess) return 8;
    return launches == 2 && last_error_calls == 2 ? 0 : 9;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-launch-guard-") as tmp:
            exe = str(Path(tmp) / "launch-guard-test")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                 "-Werror", "-I", str(ROOT / "native/providers"), "-x", "c++",
                 "-", "-o", exe],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([exe], check=True, timeout=5)

    def test_actual_stream_launcher_transports_absolute_indices_and_stops_on_fault(self):
        provider = (ROOT / "native/providers/whole_provider.cpp").read_text()
        begin = provider.index("hipError_t launch_selected_bf16_projection_hawkeye_midpoint_correction(")
        end = provider.index("// GB10's cuBLASLt BF16 BA projection", begin)
        source = (ROOT / "tests/native/hawkeye_stream_host_mock.cpp").read_text()
        source = source.replace("// QRT_ACTUAL_LAUNCHER", provider[begin:end])
        with tempfile.TemporaryDirectory(prefix="qrt-hawkeye-stream-") as tmp:
            exe = str(Path(tmp) / "stream-test")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                 "-Werror", "-I", str(ROOT / "native/providers"), "-x", "c++",
                 "-", "-o", exe],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([exe], check=True, timeout=5, capture_output=True, text=True)

    def test_dense_work_and_exhausted_time_are_rejected(self):
        source = r'''
#include "hawkeye_dispatch_policy.h"
#include "projection_output_policy.h"
using namespace qrt_hawkeye_dispatch;
static_assert(admitted(0, 0), "empty candidate set");
static_assert(window_elements(0) == 0, "empty projection");
static_assert(window_elements(65535) == 65535, "partial window");
static_assert(window_elements(65536) == 65536, "full collection grid");
static_assert(window_elements(UINT64_MAX) == 65536, "wide remaining count cannot wrap");
static_assert(maximum_exact_blocks * 16u == maximum_window_elements,
              "a full-window exact batch fits the existing index window");
static_assert(admitted(131072, 64), "inclusive admission boundary");
static_assert(!admitted(131073, 1), "one window cannot exceed scratch capacity");
static_assert(!admitted(1, 65), "one dense block must also be rejected");
static_assert(!admitted(UINT32_MAX, UINT32_MAX), "no integer wraparound");
static_assert(time_remaining(100.0, 10000.0), "inclusive time boundary");
static_assert(!time_remaining(100.001, 1.0), "slow dispatch stops submission");
static_assert(!time_remaining(1.0, 10000.001), "aggregate deadline");
using namespace qrt_projection_output;
static_assert(!needs_f32_buffer(false, false), "ordinary fused BF16 retains its allocation plan");
static_assert(needs_f32_buffer(false, true), "layer 2 WMMA must allocate F32 despite BF16 fusion");
static_assert(needs_f32_buffer(true, false), "ordinary F32 consumer");
static_assert(needs_f32_buffer(true, true), "early-layer override");
int main() {
    float f32 = 0.0f;
    unsigned short bf16 = 0;
    if (valid_buffers(nullptr, &bf16, true)) return 1;
    if (valid_buffers(&f32, nullptr, true)) return 2;
    if (!valid_buffers(&f32, &bf16, true)) return 3;
    if (!valid_buffers(&f32, nullptr, false)) return 4;
    if (valid_buffers(nullptr, nullptr, false)) return 5;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-hawkeye-policy-") as tmp:
            exe = str(Path(tmp) / "policy-test")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                 "-Werror", "-I", str(ROOT / "native/providers"), "-x", "c++",
                 "-", "-o", exe],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([exe], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
