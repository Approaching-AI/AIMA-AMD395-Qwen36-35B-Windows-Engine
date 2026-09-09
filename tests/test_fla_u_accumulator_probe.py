import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("c++"), "requires the portable C++ compiler")
class FlaUAccumulatorProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.probe = cls.root / "probe"
        subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off",
                        str(ROOT / "tests/native/fla_u_accumulator_probe.cpp"),
                        "-o", str(cls.probe)], check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def fixture(self, zero: bool = False) -> list[str]:
        v = ([0x3F80, 0xBF00] * 2048) if not zero else [0] * 4096
        expected = ([0x3F00, 0xBE80] * 2048) if not zero else v
        inverse = [0x3F80 if i % 64 == 0 else 0 for i in range(32 * 64)]
        paths = []
        for name, values in (("v", v), ("beta", [0x3F00] * 32),
                             ("inverse", inverse), ("reference", expected)):
            path = self.root / (name + ".bin")
            path.write_bytes(struct.pack("<" + "H" * len(values), *values))
            paths.append(str(path))
        return paths

    def test_single_token_uses_zero_padding_without_reading_beyond_capture(self) -> None:
        result = subprocess.run([str(self.probe), *self.fixture(), "1"],
                                check=True, capture_output=True, text=True, timeout=5)
        data = json.loads(result.stdout)
        self.assertFalse(data["inference_acceptance"])
        for variant in data["variants"]:
            self.assertEqual(variant["elements"], 4096)
            self.assertEqual(variant["bit_mismatches"], 0)

    def test_zero_reference_and_rejected_input_size_are_bounded(self) -> None:
        paths = self.fixture(zero=True)
        result = subprocess.run([str(self.probe), *paths, "1"], check=True,
                                capture_output=True, text=True, timeout=5)
        self.assertTrue(all(x["bit_mismatches"] == 0 for x in json.loads(result.stdout)["variants"]))
        Path(paths[0]).write_bytes(b"")
        rejected = subprocess.run([str(self.probe), *paths, "1"], capture_output=True, timeout=5)
        self.assertEqual(rejected.returncode, 3)

    def test_invalid_shape_is_rejected_before_loading_files(self) -> None:
        for tokens in ("0", "8193", "64garbage"):
            result = subprocess.run([str(self.probe), "v", "b", "a", "u", tokens],
                                    capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
