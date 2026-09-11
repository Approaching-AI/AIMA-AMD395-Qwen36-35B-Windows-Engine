"""Keep long corrected projections inside the existing workspace quantum."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class AttentionOutputTileTests(unittest.TestCase):
    def test_real_wrapper_preserves_strides_and_stops_after_failed_tile(self):
        source = (ROOT / "native/providers/whole_provider.cpp").read_text()
        actual = function(source, "bool full_attention_output_projection_bf16(")
        harness = r'''
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
using hipStream_t = void*;
constexpr unsigned QRT_QWEN36_MAX_POSITION_EMBEDDINGS = 262144;
unsigned setting = 1000, fail_call = 0, offset = 0;
const uint16_t *expected_weights, *expected_inputs;
uint16_t *expected_outputs;
std::vector<unsigned> calls;
unsigned env_u32_or_default(const char*, unsigned) { return setting; }
bool full_attention_output_projection_bf16_tile(
    const uint16_t* w, const uint16_t* x, uint16_t* y, unsigned rows,
    unsigned k, unsigned tokens, hipStream_t, const std::string&,
    std::string*, std::string*) {
    if (w != expected_weights || x != expected_inputs + size_t(offset) * 4096 ||
        y != expected_outputs + size_t(offset) * 2048 || rows != 2048 || k != 4096 ||
        (setting && tokens > 8192)) std::abort();
    calls.push_back(tokens);
    if (calls.size() == fail_call) return false;
    y[0] = 71; y[size_t(tokens) * rows - 1] = 93;
    offset += tokens;
    return true;
}
''' + actual + r'''
int main() {
    constexpr unsigned maximum = 16385;
    std::unique_ptr<uint16_t[]> x(new uint16_t[size_t(maximum) * 4096]);
    std::unique_ptr<uint16_t[]> y(new uint16_t[size_t(maximum) * 2048 + 1]);
    uint16_t weight = 1; std::string stage, failure;
    expected_weights = &weight; expected_inputs = x.get(); expected_outputs = y.get();
    auto run = [&](unsigned tokens) {
        calls.clear(); offset = 0;
        return full_attention_output_projection_bf16(&weight, x.get(), y.get(),
            2048, 4096, tokens, nullptr, "test", &stage, &failure);
    };
    for (unsigned tokens : {8191u, 8192u, 8193u, maximum}) {
        y[size_t(tokens) * 2048] = 157;
        if (!run(tokens) || offset != tokens || y[size_t(tokens) * 2048] != 157) return 1;
        const std::vector<unsigned> expected = tokens <= 8192 ? std::vector<unsigned>{tokens}
            : tokens == 8193 ? std::vector<unsigned>{8192, 1}
                            : std::vector<unsigned>{8192, 8192, 1};
        if (calls != expected || y[0] != 71 || y[size_t(tokens) * 2048 - 1] != 93) return 2;
    }
    fail_call = 2;
    if (run(maximum) || offset != 8192 || calls != std::vector<unsigned>{8192, 8192}) return 3;
    fail_call = 0;
    if (run(QRT_QWEN36_MAX_POSITION_EMBEDDINGS + 1) || !calls.empty()) return 4;
    setting = 0;
    if (!run(maximum) || calls != std::vector<unsigned>{maximum}) return 5;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-output-tiles-") as tmp:
            executable = str(Path(tmp) / "tiles")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-x", "c++", "-", "-o", executable],
                           input=harness, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
