from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RsqrtLookupTests(unittest.TestCase):
    def test_domain_exponent_and_layout_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "lookup"
            subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined", "-Wall", "-Wextra", "-Werror",
                            str(ROOT / "tests/native/sm121_rsqrt_lookup_probe.cpp"), "-o", str(binary)],
                           check=True, capture_output=True, text=True, timeout=30)
            result = subprocess.run([str(binary), "--domain-only"], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_normalization_rejects_partial_overlap_and_invalid_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "norm.cpp"
            source.write_text('''#include "native/providers/gdn/blackwell_l2norm.h"
#include <vector>
int main() {
 std::vector<float> raw(8192); std::vector<uint16_t> q(2048), k(2048);
 using qrt_fla_blackwell_norm::valid_normalize;
 if (!valid_normalize(raw.data(), q.data(), k.data(), 1)) return 1;
 if (valid_normalize(raw.data(), q.data(), q.data()+1, 1) ||
     valid_normalize(raw.data(), reinterpret_cast<uint16_t*>(raw.data()+1), k.data(), 1) ||
     valid_normalize(nullptr, q.data(), k.data(), 1) ||
     valid_normalize(raw.data(), q.data(), k.data(), 0) ||
     valid_normalize(raw.data(), q.data(), k.data(), 8193)) return 2;
 return 0;
}
''')
            binary = root / "norm"
            subprocess.run(["c++", "-std=c++17", "-fsanitize=address,undefined",
                            "-I" + str(ROOT / "tests/native/fla_replay_fake_hip"), "-I" + str(ROOT),
                            str(source), "-o", str(binary)], check=True, capture_output=True, text=True, timeout=30)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
