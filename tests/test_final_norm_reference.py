"""Check final-norm validation against distinct BF16/F32 formula endpoints."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function


ROOT = Path(__file__).resolve().parents[1]


class FinalNormReferenceTests(unittest.TestCase):
    def test_selected_mode_rounds_only_the_numerator(self):
        source = (ROOT / "native/providers/whole_provider.cpp").read_text()
        actual = function(source, "float selected_rmsnorm_cpu_value(")
        final_norm = function(source, "bool run_final_norm(")
        call = final_norm.split("const float reference = ", 1)[1].split(";", 1)[0]
        harness = r'''
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
constexpr unsigned QRT_QWEN36_HIDDEN_SIZE = 2048;
constexpr float QRT_QWEN36_RMS_NORM_EPSILON = 1.0e-6f;
float qrt_bf16_to_float(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
float bf16_round_to_float(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return qrt_bf16_to_float(uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16));
}
''' + actual + r'''
struct Residual { std::vector<float> gpu_output; };
struct Norm { std::vector<uint16_t> weights; };
float selected_reference(const Residual& final_output_residual_run, const Norm* run,
                         size_t selected_index, unsigned row,
                         bool use_unrounded_vllm_final_norm) {
    return ''' + call + r''';
}
int main() {
    // A midpoint carrier rounds to 1.0, while its variance stays above 1.0.
    // Legacy and incorrectly rounded-variance formulas both yield 16.0 here.
    // The required fused formula yields 15.9375, differing by more than the
    // existing 0.05 scalar guard. The second token also checks signed inputs
    // and selection offsets; neither token may change the other's variance.
    Residual residual{std::vector<float>(4096, 1.00390625f)};
    for (size_t i = 2048; i < 4096; ++i) residual.gpu_output[i] = -1.00390625f;
    Norm norm{std::vector<uint16_t>(2048, 0x4170u)};  // Gemma weight 15 + 1.
    for (unsigned row : {0u, 1023u, 2047u}) {
        if (selected_reference(residual, &norm, 0, row, true) != 15.9375f) return 1;
        if (selected_reference(residual, &norm, 1, row, true) != -15.9375f) return 2;
        if (selected_reference(residual, &norm, 0, row, false) != 16.0f) return 3;
        if (selected_reference(residual, &norm, 1, row, false) != -16.0f) return 4;
        if (selected_rmsnorm_cpu_value(residual.gpu_output, norm.weights.data(), 0, row)
            != 16.0f) return 5;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-final-norm-reference-") as tmp:
            executable = str(Path(tmp) / "reference")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-ffp-contract=off",
                 "-Wall", "-Wextra", "-Werror", "-x", "c++", "-", "-o", executable],
                input=harness, text=True, check=True, timeout=30,
            )
            subprocess.run([executable], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
