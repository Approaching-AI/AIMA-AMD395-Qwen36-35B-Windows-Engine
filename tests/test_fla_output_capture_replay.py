import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlaOutputCaptureContractTests(unittest.TestCase):
    def test_standalone_native_replay_has_fixed_allocation_and_dispatch_bounds(self) -> None:
        text = (ROOT / "native/providers/gdn/fla_output_capture_replay.cpp").read_text()
        for contract in ("number(argv[6], 8192)", "offset += native ? 64u : 1024u",
                         "allocation_bytes > 512u * 1024u * 1024u",
                         "allocation_bytes + 512u * 1024u * 1024u",
                         "hipEventSynchronize(end.handle)", "milliseconds <= 100.0f",
                         'properties.gcnArchName', 'capture size mismatch'):
            self.assertIn(contract, text)
        self.assertIn('return mismatch || nonfinite ? 3 : 0', text)
        self.assertNotIn("whole_provider", text)
        self.assertIn("&global_scratch, &profile_scratch", text)
        self.assertIn("sizeof(arguments) / sizeof(arguments[0]) == 9", text)
        self.assertIn("phase=dispatch", text)

    def test_builder_fingerprints_replay_and_supervisor_detects_its_process(self) -> None:
        builder = (ROOT / "scripts/baiying_build_fla_gdn.ps1").read_text()
        self.assertIn("$outputReplay = Join-Path", builder)
        self.assertIn("$blackwellAccumulator, $outputReplay", builder)
        self.assertIn("fla-output-capture-replay.exe", builder)
        guard = (ROOT / "scripts/baiying_guarded_inference.ps1").read_text()
        self.assertIn("fla-output-capture-replay|hipcc", guard)
        self.assertIn("q\\d+-fla-smoke", guard)


@unittest.skipUnless(shutil.which("c++"), "requires the portable C++ compiler")
class FlaOutputCpuProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.probe = cls.root / "probe"
        subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off",
                        str(ROOT / "tests/native/fla_output_accumulator_probe.cpp"),
                        "-o", str(cls.probe)], check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def fixture(self) -> list[str]:
        surfaces = (("q", [0x3F80] * 2048), ("k", [0x3F80] * 2048),
                    ("v", [0x3F00] * 4096), ("h", [0] * (32 * 128 * 128)),
                    ("g", [0.0] * 32), ("reference", [0x40B5] * 4096))
        paths = []
        for name, values in surfaces:
            path = self.root / (name + ".bin")
            path.write_bytes(struct.pack("<" + ("f" if name == "g" else "H") * len(values), *values))
            paths.append(str(path))
        return paths

    def test_single_token_has_known_causal_output_and_is_not_inference_acceptance(self) -> None:
        result = subprocess.run([str(self.probe), *self.fixture(), "1"], check=True,
                                capture_output=True, text=True, timeout=5)
        record = json.loads(result.stdout)
        self.assertFalse(record["inference_acceptance"])
        self.assertTrue(all(v["elements"] == 4096 and v["bit_mismatches"] == 0 for v in record["variants"]))

    def test_shape_and_short_file_fail_before_computation(self) -> None:
        paths = self.fixture()
        for tokens in ("0", "8193", "64invalid"):
            result = subprocess.run([str(self.probe), *paths, tokens], capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 2)
        Path(paths[0]).write_bytes(b"")
        result = subprocess.run([str(self.probe), *paths, "1"], capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 3)


if __name__ == "__main__":
    unittest.main()
