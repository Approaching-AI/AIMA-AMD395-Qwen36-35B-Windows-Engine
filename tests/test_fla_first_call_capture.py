import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("c++"), "requires a portable compiler")
class FirstCallCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        source = cls.root / "probe.cpp"
        source.write_text(r'''
#include "native/providers/gdn/first_call_capture.h"
#include <cstring>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 3) return 1;
    const std::string mode(argv[2]);
    unsigned tokens = mode == "invalid" ? 8193 : 65;
    std::vector<float> raw(65*8192, 1), gates(65*64, 2), output(65*4096, 3), state(524288, 4);
    const auto original_raw = raw, original_gates = gates;
    unsigned copies = 0, executions = 0;
    qrt_fla_capture::FirstCall capture;
    auto copy = [&](void* dst, const void* src, size_t count) {
        if (count > 1048576) std::abort();
        ++copies;
        if (mode == "copy_failure" && copies == 2) return false;
        std::memcpy(dst, src, count); return true;
    };
    auto execute = [&] {
        ++executions;
        if (mode != "disabled" && !std::filesystem::exists(std::filesystem::path(argv[1]) / "gates-f32.bin")) std::abort();
        if (mode == "execute_failure") return false;
        output[0] = 7; state.back() = 8; return true;
    };
    const char* path = mode == "disabled" ? nullptr : argv[1];
    bool first = capture.run(path, tokens, raw.data(), gates.data(), output.data(), state.data(), copy, execute);
    unsigned copies_first = copies;
    bool second = capture.run(path, tokens, raw.data(), gates.data(), output.data(), state.data(), copy, execute);
    if (raw != original_raw || gates != original_gates) return 2;
    std::cout << "{\"first\":" << first << ",\"second\":" << second
              << ",\"copies\":" << copies << ",\"copies_first\":" << copies_first
              << ",\"executions\":" << executions << "}\n";
}
''')
        cls.binary = cls.root / "probe"
        subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                        "-I" + str(ROOT), str(source), "-o", str(cls.binary)],
                       check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_case(self, mode, existing=False):
        path = self.root / mode
        if existing:
            path.mkdir()
            (path / "sentinel").write_text("preserve")
        result = subprocess.run([str(self.binary), str(path), mode],
                                check=True, capture_output=True, text=True, timeout=10)
        return path, json.loads(result.stdout)

    def test_logical_tail_complete_output_and_capture_only_first_call(self):
        path, result = self.run_case("success")
        self.assertEqual(result, dict(first=1, second=1, copies=8, copies_first=8, executions=2))
        record = json.loads((path / "capture.json").read_text())
        self.assertTrue(record["complete"])
        self.assertFalse(record["inference_acceptance"])
        self.assertEqual(record["tokens"], 65)
        for name, size in {"raw": 65*8192*4, "gates": 65*64*4, "output": 65*4096*4, "state": 524288*4}.items():
            self.assertEqual((path / (name + "-f32.bin")).stat().st_size, size)
        import struct
        self.assertEqual(struct.unpack("<f", (path / "output-f32.bin").read_bytes()[:4])[0], 7)
        self.assertEqual(struct.unpack("<f", (path / "state-f32.bin").read_bytes()[-4:])[0], 8)

    def test_failures_never_retry_or_write_completion(self):
        for mode, executions in (("copy_failure", 0), ("execute_failure", 1), ("invalid", 0)):
            with self.subTest(mode=mode):
                path, result = self.run_case(mode)
                self.assertEqual((result["first"], result["second"], result["executions"]), (0, 0, executions))
                self.assertEqual(result["copies"], result["copies_first"])
                self.assertFalse((path / "capture.json").exists())

    def test_existing_directory_preserved_and_disabled_capture_has_no_io(self):
        path, result = self.run_case("existing", existing=True)
        self.assertEqual(result["executions"], 0)
        self.assertEqual([p.name for p in path.iterdir()], ["sentinel"])
        self.assertEqual((path / "sentinel").read_text(), "preserve")
        path, result = self.run_case("disabled")
        self.assertEqual(result["executions"], 2)
        self.assertEqual(result["copies"], 0)
        self.assertFalse(path.exists())
