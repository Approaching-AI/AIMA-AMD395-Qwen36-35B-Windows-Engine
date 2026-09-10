"""Validate the real host gate snapshot writer without model or GPU work."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GateInputCaptureTests(unittest.TestCase):
    def test_bounded_read_only_snapshot_and_no_overwrite(self):
        source = r'''
#include "gate_input_capture.h"
#include <vector>
#include <cstring>
using namespace qrt_gate_capture;
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    std::filesystem::path root(argv[1]);
    std::vector<float> a(65 * 32), b(65 * 32), parameter(32);
    for (size_t i = 0; i < a.size(); ++i) { a[i] = float(i) / 8.0f; b[i] = -a[i]; }
    for (size_t i = 0; i < parameter.size(); ++i) parameter[i] = float(i) / 16.0f;
    const auto before_a = a, before_b = b, before_p = parameter;
    Span sa{a.data(), a.size()}, sb{b.data(), b.size()}, sp{parameter.data(), parameter.size()};
    std::string error;
    for (unsigned int tokens : {0u, 8193u, 64u}) {
        if (write((root / "invalid").string().c_str(), 0, tokens, sa, sb, sp, sp, &error)) return 2;
        if (std::filesystem::exists(root / "invalid")) return 3;
    }
    if (write((root / "invalid").string().c_str(), 40, 65, sa, sb, sp, sp, &error)) return 4;
    if (write((root / "invalid").string().c_str(), 0, 65, sa, sb, {nullptr, 32}, sp, &error)) return 5;
    if (write((root / "missing" / "child").string().c_str(), 0, 65, sa, sb, sp, sp, &error)) return 6;
    const auto dest = root / "capture";
    if (!write(dest.string().c_str(), 0, 65, sa, sb, sp, sp, &error)) return 7;
    const char *names[] = {"a-f32.bin", "b-f32.bin", "a-log-f32.bin", "dt-bias-f32.bin"};
    const Span spans[] = {sa, sb, sp, sp};
    for (unsigned int i = 0; i < 4; ++i) {
        if (std::filesystem::file_size(dest / names[i]) != spans[i].size * 4) return 8;
        std::ifstream file(dest / names[i], std::ios::binary);
        std::vector<float> actual(spans[i].size); file.read(reinterpret_cast<char *>(actual.data()), actual.size() * 4);
        if (std::memcmp(actual.data(), spans[i].data, actual.size() * 4)) return 9;
    }
    if (!std::filesystem::exists(dest / "capture.json")) return 10;
    if (write(dest.string().c_str(), 0, 65, sa, sb, sp, sp, &error)) return 11;
    if (a != before_a || b != before_b || parameter != before_p) return 12;
    std::ifstream metadata(dest / "capture.json");
    std::string text((std::istreambuf_iterator<char>(metadata)), {});
    return text.find("\"tokens\":65") != std::string::npos &&
           text.find("\"complete\":true") != std::string::npos ? 0 : 13;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-gate-input-capture-") as tmp:
            exe = str(Path(tmp) / "capture-test")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-I", str(ROOT / "native/providers"), "-x", "c++", "-", "-o", exe],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([exe, tmp], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
