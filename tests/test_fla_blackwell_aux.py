import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("c++"), "requires a portable compiler")
class BlackwellAuxHostTests(unittest.TestCase):
    """Actual host code and address contracts, with no GPU/math acceptance."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.binary = cls.root / "output-host-test"
        cls.flags = ["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                     "-I" + str(ROOT / "tests/native/fla_replay_fake_hip"), "-I" + str(ROOT)]
        subprocess.run(cls.flags + [str(ROOT / "native/providers/gdn/fla_output_capture_replay.cpp"),
                                   str(ROOT / "tests/native/fla_state_fake_launch.cpp"), "-o", str(cls.binary)],
                       check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def fixture(self, tokens):
        directory = self.root / ("input-" + str(tokens))
        directory.mkdir(exist_ok=True)
        sizes = {"q-normalized-bf16": tokens * 2048 * 2, "k-normalized-bf16": tokens * 2048 * 2,
                 "v-new-bf16": tokens * 4096 * 2, "chunk-state-bf16": ((tokens + 63) // 64) * 32 * 128 * 128 * 2,
                 "g-cumsum-f32": tokens * 32 * 4, "native-output-bf16": tokens * 4096 * 2}
        for name, size in sizes.items():
            with (directory / ("full-" + name + ".bin")).open("wb") as stream:
                stream.truncate(size)
        return directory

    def run_output(self, tokens, **environment):
        env = {k: v for k, v in os.environ.items() if k != "QRT_TEST_FAKE_DISPATCH_MS"}
        env.update(environment)
        return subprocess.run([str(self.binary), "-", "native-blackwell-output", "256", "0",
                               str(self.fixture(tokens)), str(tokens), "real"],
                              capture_output=True, text=True, env=env, timeout=15)

    def test_output_chunks_tail_and_readback_fit_one_reused_chunk(self):
        result = self.run_output(1025)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count("FAKE_HIP blackwell_scores tokens=64\n"), 16)
        self.assertEqual(result.stderr.count("FAKE_HIP blackwell_output tokens=64\n"), 16)
        self.assertIn("FAKE_HIP blackwell_output tokens=1\n", result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["segments"], 34)
        self.assertEqual(data["elements"], 1025 * 4096)
        self.assertFalse(data["inference_acceptance"])

    def test_output_admission_rejects_before_second_kernel(self):
        result = self.run_output(128, QRT_TEST_FAKE_DISPATCH_MS="101")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr.count("FAKE_HIP blackwell_scores"), 1)
        self.assertNotIn("FAKE_HIP blackwell_output", result.stderr)

    def test_wu_allows_exact_v_alias_but_rejects_partial_and_other_overlap(self):
        source = self.root / "alias.cpp"
        source.write_text('''#include "native/providers/gdn/blackwell_wu_output.h"
#include <vector>
int main() {
    std::vector<uint16_t> k(64*2048), v(64*4096+1), b(64*32), a(64*2048), w(64*4096), u(64*4096);
    std::vector<float> g(64*32);
    auto valid = [&](uint16_t* out_w, uint16_t* out_u, unsigned count) {
        return qrt_fla_blackwell_aux::valid_wu(k.data(), v.data(), b.data(), a.data(), g.data(), out_w, out_u, count);
    };
    if (!valid(w.data(), u.data(), 64) || !valid(w.data(), v.data(), 64) || !valid(w.data(), v.data(), 1)) return 1;
    if (valid(w.data(), v.data()+1, 64) || valid(w.data(), a.data(), 64) || valid(v.data(), u.data(), 64)) return 2;
    if (valid(w.data(), w.data()+1, 64) || valid(w.data(), u.data(), 0) || valid(w.data(), u.data(), 65)) return 3;
    return 0;
}
''')
        binary = self.root / "alias-test"
        subprocess.run(self.flags + [str(source), "-o", str(binary)], check=True, capture_output=True, timeout=30)
        subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
