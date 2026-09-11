"""Check actual split-attention bounds and failure ordering without GPU work."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AttentionLaunchPolicyTests(unittest.TestCase):
    def test_scratch_span_is_checked_and_failed_scores_do_not_submit_pv(self):
        header = (ROOT / "native/providers/ck_fmha/blackwell_attention.h").read_text()
        launch = "inline int launch_queries(" + header.split(
            "inline int launch_queries(", 1
        )[1].split("} // namespace qrt_blackwell_attention", 1)[0]
        source = r'''
#include <cstddef>
#include <cstdint>
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t = void*;
struct dim3 { explicit dim3(unsigned, unsigned = 1u) {} };
constexpr unsigned kQueryHeads = 16, kHeadDim = 256, kThreads = 256;
constexpr unsigned kBlackwellSubgroups = 16;
void blackwell_exact_scores_kernel() {}
template<bool SerialValue, bool PrecomputedScores = false>
void blackwell_exact_attention_kernel() {}
unsigned launches = 0, error_queries = 0;
bool fail_scores = false;
template<class... T> void record_launch(T...) { ++launches; }
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(...) record_launch(__VA_ARGS__)
hipError_t hipGetLastError() {
    ++error_queries;
    return fail_scores && launches == 1u ? hipErrorUnknown : hipSuccess;
}
''' + launch + r'''
int main() {
    uint16_t operand = 0u; float output = 0.0f, scratch = 0.0f;
    auto split = [&](unsigned start, unsigned count, float* workspace,
                     size_t elements, unsigned layout = 2u) {
        return launch_queries(&operand, &operand, &operand, &output, nullptr,
            start, count, 0u, nullptr, nullptr, nullptr, true, nullptr,
            layout, workspace, elements);
    };
    if (split(0, 8, nullptr, 1024) != hipErrorInvalidValue) return 1;
    if (split(0, 8, &scratch, 1023) != hipErrorInvalidValue) return 2;
    if (split(7160, 8, &scratch, 8u * 16u * 8u) != hipErrorInvalidValue) return 3;
    if (split(8191, 2, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 4;
    if (split(0, 33, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 5;
    if (split(0, 0, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 6;
    if (split(UINT32_MAX, 8, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 7;
    if (split(0, 8, &scratch, SIZE_MAX, 3) != hipErrorInvalidValue) return 8;
    if (launches || error_queries) return 9;
    if (split(7160, 8, &scratch, 8u * 16u * 7168u) != hipSuccess ||
        launches != 2u || error_queries != 2u) return 10;
    launches = error_queries = 0u; fail_scores = true;
    if (split(0, 8, &scratch, 1024) != hipErrorUnknown ||
        launches != 1u || error_queries != 1u) return 11;
    launches = error_queries = 0u; fail_scores = false;
    if (split(262143, 1, nullptr, 0, 1) != hipSuccess || launches != 1u)
        return 12;  // Original long terminal route needs no new scratch.
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-attention-guard-") as tmp:
            executable = str(Path(tmp) / "launch-guard")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                 "-Werror", "-x", "c++", "-", "-o", executable],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([executable], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
