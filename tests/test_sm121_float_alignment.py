"""Compare exact scalar multiplication/alignment with original ordered sums."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FloatAlignmentTests(unittest.TestCase):
    def test_ordered_groups_with_extrema_and_original_fallback(self):
        code = r'''
#include "float_alignment_cases.h"
#include <cstdio>
int main() {
    unsigned accepted = 0u, fallback = 0u;
    for (unsigned row = 0u; row < 65536u; ++row) {
        uint32_t actual_groups[16], expected_groups[16];
        const auto result = qrt_float_alignment_cases::candidate(row, actual_groups);
        const auto expected = qrt_float_alignment_cases::reference(row, expected_groups);
        for (unsigned group = 0u; group < 16u; ++group) if (actual_groups[group] != expected_groups[group]) {
            std::printf("GROUP_DIFF row=%u group=%u expected=%08x actual=%08x\n", row, group, expected_groups[group], actual_groups[group]);
            return 2;
        }
        if (result.bits != expected) {
            std::printf("DIFF row=%u expected=%08x actual=%08x\n", row, expected, result.bits);
            return 1;
        }
        accepted += result.accepted; fallback += result.fallback;
    }
    std::printf("{\"dots\":65536,\"ordered_groups\":1048576,\"raw_mismatches\":0,\"float_groups\":%u,\"fallback_groups\":%u,\"inference_acceptance\":false}\n", accepted, fallback);
    return !accepted || !fallback;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-float-alignment-") as directory:
            exe = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                            "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                            "-I", str(ROOT / "tests/native"), "-x", "c++", "-", "-o", exe],
                           input=code, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
