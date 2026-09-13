"""Check actual optional matrix profiling boundaries and failure propagation."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MatrixProducerWallTests(unittest.TestCase):
    def test_actual_impl_preserves_submission_and_bounds_profile_waits(self):
        provider = (ROOT / "native/providers/whole_provider.cpp").read_text()
        begin = provider.index("bool resident_bf16_matrix_matmul_impl(")
        end = provider.index("bool resident_bf16_matrix_matmul(\n", begin)
        source = r'''
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using hipStream_t = void*;
static std::vector<std::string> calls;
static int fault = 0, waits = 0;
static uint16_t weights[1], inputs[1];
static float output[1];
static int stream_owner;
struct ResidentBf16MatrixProvider { int handle = 10; void* workspace = output; };
struct ResidentBf16MatrixPlan {
    int operation = 11, weight_layout = 12, input_layout = 13, output_layout = 14, algorithm = 15;
    size_t workspace_bytes = 64;
};
static ResidentBf16MatrixProvider owner;
static ResidentBf16MatrixPlan selected;
int hipStreamSynchronize(hipStream_t stream) {
    assert(stream == &stream_owner);
    calls.push_back(++waits == 1 ? "before" : "after");
    return fault == (waits == 1 ? 1 : 5) ? 1 : 0;
}
bool check_hip(int status, const std::string& stage, std::string* failed, std::string* message) {
    if (status) { *failed = stage; *message = "injected HIP fault"; }
    return !status;
}
bool check_hipblaslt(int status, const std::string& stage, std::string* failed, std::string* message) {
    if (status) { *failed = stage; *message = "injected matrix fault"; }
    return !status;
}
bool ensure_resident_bf16_matrix_provider(ResidentBf16MatrixProvider** out, std::string*, std::string*) {
    calls.push_back("ensure"); *out = &owner; return fault != 2;
}
bool create_resident_bf16_matrix_plan(ResidentBf16MatrixProvider* p, unsigned rows, unsigned k,
    unsigned tokens, bool f32, unsigned index, ResidentBf16MatrixPlan** out, std::string*, std::string*) {
    assert(p == &owner && rows == 8192 && k == 2048 && tokens && f32 && index == 0);
    calls.push_back("plan"); *out = &selected; return fault != 3;
}
int hipblasLtMatmul(int handle, int operation, const float* alpha, const uint16_t* w, int wl,
    const uint16_t* x, int xl, const float* beta, void* c, int cl, void* d, int dl,
    const int* algorithm, void* workspace, size_t bytes, hipStream_t stream) {
    assert(handle == 10 && operation == 11 && *alpha == 1.0f && *beta == 0.0f);
    assert(w == weights && x == inputs && c == output && d == output);
    assert(wl == 12 && xl == 13 && cl == 14 && dl == 14 && *algorithm == 15);
    assert(workspace == output && bytes == 64 && stream == &stream_owner);
    calls.push_back("matmul"); return fault == 4 ? 1 : 0;
}
''' + provider[begin:end] + r'''
void mode(const char* value) {
#ifdef _WIN32
    _putenv_s("QRT_QWEN36_PROFILE_MATRIX_PRODUCER_WALL", value ? value : "");
#else
    if(value) setenv("QRT_QWEN36_PROFILE_MATRIX_PRODUCER_WALL", value, 1);
    else unsetenv("QRT_QWEN36_PROFILE_MATRIX_PRODUCER_WALL");
#endif
}
int main() {
    std::string stage, message;
    auto run = [&](unsigned tokens = 8192, const uint16_t* w = weights) {
        calls.clear(); waits = 0; stage.clear(); message.clear();
        return resident_bf16_matrix_matmul_impl(w, inputs, output, 8192, 2048, tokens,
            true, 1.0f, 0.0f, 0, &stream_owner, "producer", &stage, &message);
    };
    const std::vector<std::string> ordinary{"ensure", "plan", "matmul"};
    for (const char* setting : {static_cast<const char*>(nullptr), "", "0"}) {
        mode(setting); assert(run() && calls == ordinary && waits == 0);
    }
    mode("1");
    for (unsigned tokens : {1u, 65u, 1023u}) assert(run(tokens) && calls == ordinary);
    assert(run(1024) && calls == std::vector<std::string>({"before", "ensure", "plan", "matmul", "after"}));
    for (const char* setting : {"2", "-1", "true", "1junk"}) {
        mode(setting); assert(!run() && calls.empty() && stage == "producer");
    }
    mode("1"); assert(!run(8192, nullptr) && calls.empty());
    const std::vector<std::vector<std::string>> failed{
        {"before"}, {"before", "ensure"}, {"before", "ensure", "plan"},
        {"before", "ensure", "plan", "matmul"}, {"before", "ensure", "plan", "matmul", "after"}};
    for (fault = 1; fault <= 5; ++fault) {
        assert(!run() && calls == failed[size_t(fault - 1)]);
        if(fault == 1) assert(stage == "producer_profile_predecessor");
        if(fault == 4) assert(stage == "producer");
        if(fault == 5) assert(stage == "producer_profile_completed");
    }
    fault = 4; mode("0"); assert(!run() && calls == ordinary && waits == 0);
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-matrix-wall-") as tmp:
            exe = str(Path(tmp) / "matrix-wall")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-x", "c++", "-", "-o", exe],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([exe], check=True, timeout=5, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
