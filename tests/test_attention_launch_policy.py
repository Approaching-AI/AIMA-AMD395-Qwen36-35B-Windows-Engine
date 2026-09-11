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
        launch = "constexpr unsigned int kSplitMaxTokens" + header.split(
            "constexpr unsigned int kSplitMaxTokens", 1
        )[1].split("} // namespace qrt_blackwell_attention", 1)[0]
        source = r'''
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t = void*;
using hipEvent_t = void*;
struct dim3 { explicit dim3(unsigned, unsigned = 1u, unsigned = 1u) {} };
constexpr unsigned kQueryHeads = 16, kHeadDim = 256, kThreads = 256;
constexpr unsigned kKvHeads = 2, kIntegerMatrixColumns = 128;
constexpr unsigned kBlackwellSubgroups = 16;
constexpr unsigned kCooperativeColumns = 64;
constexpr unsigned kExactTileTokens = 32;
void blackwell_exact_scores_kernel() {}
void blackwell_cooperative_scores_kernel() {}
void blackwell_cooperative_value_kernel() {}
void blackwell_transpose_keys_kernel() {}
template<bool NativeProducts = false> void blackwell_transposed_scores_kernel() {}
void blackwell_online_probability_kernel() {}
void blackwell_probability_value_kernel() {}
template<bool NativeMma = false> void blackwell_mantissa_scores_kernel() {}
template<bool NativeMma = false> void blackwell_mantissa_value_kernel() {}
template<bool SerialValue, bool PrecomputedScores = false, bool SplitDecodeValue = false,
         bool NativeProducts = false>
void blackwell_exact_attention_kernel() {}
unsigned launches = 0, error_queries = 0;
bool fail_scores = false, fail_probability = false;
bool fail_event = false;
unsigned events = 0;
hipError_t hipEventRecord(hipEvent_t, hipStream_t) {
    ++events;
    return fail_event ? hipErrorUnknown : hipSuccess;
}
const char* launch_names[64]{};
template<class... T> void record_launch(const char* name, T...) { launch_names[launches++] = name; }
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel, ...) record_launch(#kernel, kernel, __VA_ARGS__)
hipError_t hipGetLastError() {
    ++error_queries;
    return ((fail_scores && launches == 1u) || (fail_probability && launches == 2u))
        ? hipErrorUnknown : hipSuccess;
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
    if (split(16383, 2, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 4;
    if (split(0, 33, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 5;
    if (split(0, 0, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 6;
    if (split(UINT32_MAX, 8, &scratch, SIZE_MAX) != hipErrorInvalidValue) return 7;
    if (split(0, 8, &scratch, SIZE_MAX, 9) != hipErrorInvalidValue) return 8;
    if (launches || error_queries) return 9;
    if (split(0, 8, &scratch, 1791, 3) != hipErrorInvalidValue || launches || error_queries)
        return 13;
    if (split(7160, 8, &scratch, 8u * 16u * 7168u) != hipSuccess ||
        launches != 2u || error_queries != 2u) return 10;
    launches = error_queries = 0u; fail_scores = true;
    if (split(0, 8, &scratch, 1024) != hipErrorUnknown ||
        launches != 1u || error_queries != 1u) return 11;
    launches = error_queries = 0u; fail_scores = false;
    if (split(262143, 1, nullptr, 0, 1) != hipSuccess || launches != 1u)
        return 12;  // Original long terminal route needs no new scratch.
    launches = error_queries = 0u;
    if (split(0, 8, &scratch, 1792, 3) != hipSuccess || launches != 3u || error_queries != 3u)
        return 14;
    launches = error_queries = 0u; fail_probability = true;
    if (split(0, 8, &scratch, 1792, 3) != hipErrorUnknown || launches != 2u || error_queries != 2u)
        return 15;
    if (events) return 16;  // No instrumentation calls in the provider default.
    launches = error_queries = 0u; fail_probability = false; fail_event = true;
    if (launch_queries(&operand, &operand, &operand, &output, nullptr,
        0, 8, 0, nullptr, nullptr, nullptr, true, nullptr, 2, &scratch, 1024,
        &operand) != hipErrorUnknown || launches != 1u || events != 1u)
        return 17;  // A failed timing event also prevents dependent PV work.
    launches = error_queries = events = 0u; fail_event = false;
    if (transpose_keys(nullptr, &operand, 512u, 1u, nullptr) != hipErrorInvalidValue ||
        transpose_keys(&operand, &operand, 511u, 1u, nullptr) != hipErrorInvalidValue ||
        transpose_keys(&operand, &operand, SIZE_MAX, 16385u, nullptr) != hipErrorInvalidValue ||
        transpose_keys(&operand, &operand, SIZE_MAX, 0u, nullptr) != hipErrorInvalidValue || launches)
        return 18;
    if (transpose_keys(&operand, &operand, 16384u * 512u, 16384u, nullptr) != hipSuccess ||
        launches != 1u || error_queries != 1u) return 19;
    launches = error_queries = 0u;
    auto transposed = [&](const uint16_t* prepared, unsigned stride) {
        return launch_queries(&operand, &operand, &operand, &output, nullptr,
            7160, 8, 0, nullptr, nullptr, nullptr, true, nullptr, 4, &scratch,
            8u * 16u * 7168u, nullptr, nullptr, prepared, stride);
    };
    if (transposed(nullptr, 7168u) != hipErrorInvalidValue ||
        transposed(&operand, 7167u) != hipErrorInvalidValue ||
        transposed(&operand, 16385u) != hipErrorInvalidValue || launches) return 20;
    if (transposed(&operand, 7169u) != hipSuccess || launches != 2u || error_queries != 2u)
        return 21;
    launches = error_queries = 0u; fail_scores = true;
    if (transposed(&operand, 7169u) != hipErrorUnknown || launches != 1u || error_queries != 1u)
        return 22;
    launches = error_queries = 0u; fail_scores = false;
    const size_t continuation_cells = 16u * 8197u;
    if (split(8196, 1, &scratch, continuation_cells - 1u) != hipErrorInvalidValue || launches)
        return 23;
    if (split(8196, 1, &scratch, continuation_cells) != hipSuccess || launches != 2u)
        return 24;
    launches = error_queries = 0u;
    if (split(16383, 1, &scratch, 16u * 16384u) != hipSuccess || launches != 2u)
        return 25;
    if (split_scratch_elements(1u, 16385u, 2u) != 0u) return 26;
    launches = error_queries = 0u;
    if (launch_queries(&operand, &operand, &operand, &output, nullptr,
        0, 8, 0, nullptr, nullptr, nullptr, true, nullptr, 2, &scratch, 1024,
        nullptr, nullptr, &operand, 8u, true) != hipErrorInvalidValue || launches)
        return 27;
    if (launch_queries(&operand, &operand, &operand, &output, nullptr,
        0, 8, 0, nullptr, nullptr, nullptr, true, nullptr, 4, &scratch, 1024,
        nullptr, nullptr, &operand, 8u, true) != hipSuccess || launches != 2u)
        return 28;
    launches = error_queries = 0u; fail_scores = true;
    if (launch_queries(&operand, &operand, &operand, &output, nullptr,
        0, 8, 0, nullptr, nullptr, nullptr, true, nullptr, 4, &scratch, 1024,
        nullptr, nullptr, &operand, 8u, true) != hipErrorUnknown || launches != 1u)
        return 29;
    launches = error_queries = 0u; fail_scores = false;
    auto matrix = [&](size_t elements, const uint16_t* prepared,
                      unsigned stride = 17u, bool native = false) {
        return launch_queries(&operand, &operand, &operand, &output, nullptr,
            1, 16, 0, nullptr, nullptr, nullptr, true, nullptr, 5, &scratch,
            elements, nullptr, nullptr, prepared, stride, native);
    };
    const size_t matrix_elements = split_scratch_elements(16u, 17u, 5u);
    if (matrix_elements != 7040u || matrix(matrix_elements - 1u, &operand) != hipErrorInvalidValue ||
        matrix(matrix_elements, nullptr) != hipErrorInvalidValue ||
        matrix(matrix_elements, &operand, 16u) != hipErrorInvalidValue ||
        matrix(matrix_elements, &operand, 16385u) != hipErrorInvalidValue ||
        matrix(matrix_elements, &operand, 17u, true) != hipErrorInvalidValue || launches)
        return 30;
    if (matrix(matrix_elements, &operand) != hipSuccess || launches != 3u || error_queries != 3u) return 31;
    launches = error_queries = 0u; fail_scores = true;
    if (matrix(matrix_elements, &operand) != hipErrorUnknown || launches != 1u) return 32;
    launches = error_queries = 0u; fail_scores = false; fail_probability = true;
    if (matrix(matrix_elements, &operand) != hipErrorUnknown || launches != 2u) return 33;
    fail_probability = false;
    for (unsigned layout : {6u, 7u}) {
        launches = error_queries = 0u;
        if (launch_queries(&operand, &operand, &operand, &output, nullptr,
            1, 16, 0, nullptr, nullptr, nullptr, true, nullptr, layout, &scratch,
            matrix_elements, nullptr, nullptr, &operand, 17u) != hipSuccess || launches != 3u) return 34;
        const char* expected = layout == 6u ? "blackwell_transposed_scores_kernel<false>" : "blackwell_mantissa_scores_kernel<true>";
        if (!std::strstr(launch_names[0], expected) ||
            !std::strstr(launch_names[1], "blackwell_online_probability_kernel") ||
            !std::strstr(launch_names[2], "blackwell_mantissa_value_kernel<true>")) return 35;
    }
    launches = error_queries = 0u;
    if (split(1, 16, &scratch, matrix_elements - 1u, 8u) != hipErrorInvalidValue || launches) return 36;
    if (split(1, 16, &scratch, matrix_elements, 8u) != hipSuccess || launches != 3u ||
        !std::strstr(launch_names[0], "blackwell_cooperative_scores_kernel") ||
        !std::strstr(launch_names[1], "blackwell_online_probability_kernel") ||
        !std::strstr(launch_names[2], "blackwell_cooperative_value_kernel")) return 37;
    launches = error_queries = 0u; fail_scores = true;
    if (split(1, 16, &scratch, matrix_elements, 8u) != hipErrorUnknown || launches != 1u) return 38;
    launches = error_queries = 0u; fail_scores = false; fail_probability = true;
    if (split(1, 16, &scratch, matrix_elements, 8u) != hipErrorUnknown || launches != 2u) return 39;
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
