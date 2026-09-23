"""The long-context K boundary uses adjacent-pair, single-row reduction."""

from pathlib import Path
import hashlib
import json
import os
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/q1_full_key_norm_row263282_head1.json"


class Q1FullKeyNormSingleRowTests(unittest.TestCase):
    def test_gb10_long_context_key_boundary(self):
        fixture = json.loads(FIXTURE.read_text())
        raw = (FIXTURE.parent / fixture["input_bf16_file"]).read_bytes()
        self.assertEqual(len(raw), 512)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), fixture["input_sha256"])
        words = struct.unpack("<256H", raw)
        source = r'''
#include "native/providers/gdn/sm121_q1_math.h"
#include <array>
using namespace qrt_sm121_q1;
float sum(const float *values, bool single_row) {
    std::array<float, 128> partial{};
    float warp[4];
    for (unsigned lane = 0; lane < 128; ++lane)
        partial[lane] = single_row
            ? head_norm_single_row_key_lane_sumsq(values, lane)
            : head_norm_lane_sumsq(values, lane, true);
    for (unsigned w = 0; w < 4; ++w) {
        for (unsigned offset = 16; offset; offset >>= 1)
            for (unsigned lane = 0; lane < offset; ++lane)
                partial[w * 32 + lane] =
                    add(partial[w * 32 + lane], partial[w * 32 + lane + offset]);
        warp[w] = partial[w * 32];
    }
    return head_norm_warp_sum(warp, true);
}
int main() {
    const uint16_t input[256] = {''' + ",".join(map(str, words)) + r'''};
    float values[256];
    for (unsigned i = 0; i < 256; ++i) values[i] = widen(input[i]);
    const float correct_sum = sum(values, true);
    const float correct_arg = add(multiply(correct_sum, 1.f / 256.f), 1.e-6f);
    const float old_arg = add(multiply(sum(values, false), 1.f / 256.f), 1.e-6f);
    if (qrt_sm121_exp2::bits(correct_sum) != ''' + str(fixture["single_row_sum_bits"]) + r'''u) return 1;
    if (qrt_sm121_exp2::bits(correct_arg) != ''' + str(fixture["single_row_argument_bits"]) + r'''u) return 2;
    if (qrt_sm121_exp2::bits(old_arg) != ''' + str(fixture["old_strided_argument_bits"]) + r'''u) return 3;
    const float scale = add(1.f, widen(''' + str(fixture["channel_weight_bf16"]) + r'''));
    const float expected_inverse = qrt_sm121_exp2::value(''' + str(fixture["single_row_inverse_bits"]) + r'''u);
    const float old_inverse = qrt_sm121_exp2::value(''' + str(fixture["old_strided_inverse_bits"]) + r'''u);
    const float input_value = values[''' + str(fixture["channel"]) + r'''];
    if (bf16(multiply(multiply(input_value, expected_inverse), scale)) !=
        ''' + str(fixture["reference_channel_bf16"]) + r'''u) return 4;
    if (bf16(multiply(multiply(input_value, old_inverse), scale)) !=
        ''' + str(fixture["old_strided_channel_bf16"]) + r'''u) return 5;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-q1-full-key-norm-") as tmp:
            exe = str(Path(tmp) / "replay")
            subprocess.run([os.getenv("CXX", "c++"), "-std=c++17", "-O2",
                            "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT), "-x", "c++", "-", "-o", exe],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
