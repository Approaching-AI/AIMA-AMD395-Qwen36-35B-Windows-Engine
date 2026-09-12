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
#include <cassert>
#include <cstring>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 3) return 1;
    const std::string mode(argv[2]);
    if (mode == "window_parse") {
        qrt_fla_capture::Window window;
        assert(qrt_fla_capture::parse_window(nullptr, nullptr, window));
        assert(!window.first_position && !window.tokens);
        assert(qrt_fla_capture::parse_window("24576", "8192", window));
        assert(window.first_position == 24576 && window.tokens == 8192);
        assert(qrt_fla_capture::parse_window("57344", "8192", window));
        for (const auto pair : std::vector<std::pair<const char*, const char*>>{
              std::make_pair(nullptr, "1024"), std::make_pair("1024", nullptr),
              std::make_pair("", "1024"), std::make_pair("0", "1024"),
              std::make_pair("1023", "1024"), std::make_pair("1024", "1023"),
              std::make_pair("1024", "8193"), std::make_pair("65536", "1024"),
              std::make_pair("999999999999", "1024"), std::make_pair("+1024", "1024")})
            assert(!qrt_fla_capture::parse_window(pair.first, pair.second, window));
        std::cout << "{}\n"; return 0;
    }
    if (mode.rfind("window", 0) == 0) {
        constexpr unsigned total = 3072;
        qrt_fla_capture::Window window{mode == "window_tail" ? 2048u : 1024u, 1024u};
        std::vector<float> raw(total*8192u), gates(total*64u), output(total*4096u), state(524288);
        for (unsigned row = 0; row < total; ++row) {
            raw[row*8192u] = float(row); gates[row*64u] = float(row+1);
        }
        qrt_fla_capture::FirstCall capture;
        unsigned executions = 0, state_reads = 0;
        auto copy = [&](void* destination, const void* source, size_t bytes) {
            assert(bytes && bytes <= capture.chunk_bytes);
            const auto address = reinterpret_cast<uintptr_t>(source);
            auto within = [&](const auto& values, size_t first, size_t count) {
                const auto begin = reinterpret_cast<uintptr_t>(values.data()+first);
                return address >= begin && address+bytes <= begin+count*4u;
            };
            if (within(state, 0, state.size())) {
                ++state_reads;
                if (mode == "window_copy_failure") return false;
            } else {
                assert(within(raw, window.first_position*8192u, window.tokens*8192u) ||
                       within(gates, window.first_position*64u, window.tokens*64u) ||
                       within(output, window.first_position*4096u, window.tokens*4096u));
            }
            std::memcpy(destination, source, bytes); return true;
        };
        auto execute = [&] {
            ++executions;
            for (unsigned offset = 0; offset < total; offset += 1024) {
                if (mode != "window_missing" && !capture.before_segment(offset, state.data(), copy)) return false;
                std::fill(state.begin(), state.end(), float(offset+1024));
                for (unsigned row = offset; row < offset+1024; ++row) output[row*4096u] = float(row+2);
            }
            return true;
        };
        const bool ok = capture.run(argv[1], total, raw.data(), gates.data(), output.data(),
                                    state.data(), copy, execute, 0, window);
        assert(executions == 1);
        for (unsigned row = 0; row < total; ++row) {
            assert(raw[row*8192u] == float(row) && gates[row*64u] == float(row+1));
        }
        const unsigned before = state_reads;
        assert(capture.before_segment(window.first_position, state.data(), copy));
        assert(state_reads == before);
        std::cout << "{\"success\":" << ok << ",\"state_reads\":" << state_reads << "}\n";
        return 0;
    }
    if (mode == "parse") {
        unsigned value = 99;
        if (!qrt_fla_capture::parse_call_index(nullptr, value) || value != 0) return 3;
        for (unsigned i = 0; i <= 63; ++i) {
            if (!qrt_fla_capture::parse_call_index(std::to_string(i).c_str(), value) || value != i) return 4;
        }
        for (const char* bad : {"", "-1", "+3", "3x", " 3", "64", "999999999999"}) {
            if (qrt_fla_capture::parse_call_index(bad, value)) return 5;
        }
        std::cout << "{}\n"; return 0;
    }
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
        if (mode != "disabled" && (mode != "selected" || executions >= 4) &&
            !std::filesystem::exists(std::filesystem::path(argv[1]) / "gates-f32.bin")) std::abort();
        if (mode == "execute_failure") return false;
        output[0] = mode == "selected" ? 6.0f + executions : 7; state.back() = 8; return true;
    };
    const char* path = mode == "disabled" ? nullptr : argv[1];
    if (mode == "selected") {
        for (unsigned i = 0; i < 5; ++i) {
            if (!capture.run(path, tokens, raw.data(), gates.data(), output.data(), state.data(), copy, execute, 3)) return 6;
            if (i < 3 && (copies || std::filesystem::exists(path))) return 7;
            if (i >= 3 && copies != 8) return 8;
        }
        if (executions != 5 || raw != original_raw || gates != original_gates) return 9;
        std::cout << "{}\n"; return 0;
    }
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

    def test_selected_call_has_no_earlier_reads_and_records_actual_ordinal(self):
        path, _ = self.run_case("selected")
        record = json.loads((path / "capture.json").read_text())
        self.assertEqual(record["kind"], "selected_native_gdn_call")
        self.assertEqual(record["call_index"], 3)
        self.assertTrue(record["complete"])
        import struct
        self.assertEqual(struct.unpack("<f", (path / "output-f32.bin").read_bytes()[:4])[0], 10)

    def test_call_index_parsing_rejects_ambiguous_or_unbounded_selection(self):
        self.run_case("parse")

    def test_original_window_keeps_its_initial_and_final_states_and_exact_slice(self):
        import struct
        for mode, start in (("window_middle", 1024), ("window_tail", 2048)):
            path, result = self.run_case(mode)
            try:
                self.assertEqual(result, dict(success=1, state_reads=4))
                record = json.loads((path / "capture.json").read_text())
                self.assertEqual((record["tokens"], record["first_position"], record["captured_tokens"]),
                                 (3072, start, 1024))
                self.assertEqual(record["state_tokens"], start + 1024)
                self.assertTrue(record["initial_state_captured"])
                for name, width, add in (("raw", 8192, 0), ("gates", 64, 1), ("output", 4096, 2)):
                    p = path / (name + "-f32.bin")
                    self.assertEqual(p.stat().st_size, 1024 * width * 4)
                    with p.open("rb") as stream:
                        self.assertEqual(struct.unpack("<f", stream.read(4))[0], start + add)
                        stream.seek(1023 * width * 4)
                        self.assertEqual(struct.unpack("<f", stream.read(4))[0], start + 1023 + add)
                for filename, expected in (("initial-state-f32.bin", start), ("state-f32.bin", start+1024)):
                    p = path / filename
                    self.assertEqual(p.stat().st_size, 524288*4)
                    with p.open("rb") as stream:
                        self.assertEqual(struct.unpack("<f", stream.read(4))[0], expected)
                        stream.seek(-4, 2)
                        self.assertEqual(struct.unpack("<f", stream.read(4))[0], expected)
            finally:
                shutil.rmtree(path)

    def test_missing_or_failed_original_state_read_never_completes_a_window(self):
        for mode in ("window_missing", "window_copy_failure"):
            path, result = self.run_case(mode)
            try:
                self.assertFalse(result["success"])
                self.assertFalse((path / "capture.json").exists())
            finally:
                shutil.rmtree(path)

    def test_window_parser_requires_both_bounded_original_segment_positions(self):
        self.run_case("window_parse")
