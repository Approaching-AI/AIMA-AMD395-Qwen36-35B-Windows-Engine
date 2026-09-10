import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FRAME = 32 * 128 * 128


class RawPrefixProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.directory.name) / "probe"
        subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/native/fla_raw_prefix_probe.cpp"), "-o", str(cls.binary)],
                       check=True, capture_output=True, text=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_nonzero_seed_is_carried_and_raw_reference_never_feeds_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, width, count in (("k-normalized-bf16", 2, 128 * 2048),
                                       ("w-bf16", 2, 128 * 4096), ("u-bf16", 2, 128 * 4096),
                                       ("v-new-bf16", 2, 128 * 4096), ("g-cumsum-f32", 4, 128 * 32)):
                (root / (name + ".bin")).write_bytes(b"\0" * (width * count))
            (root / "initial_state-f32.bin").write_bytes(struct.pack("<f", 1) * FRAME)
            checkpoints = struct.pack("<f", 1) * FRAME + struct.pack("<f", .5) * FRAME
            (root / "checkpoint-f32.bin").write_bytes(checkpoints)
            (root / "terminal-state-f32.bin").write_bytes(struct.pack("<f", .25) * FRAME)
            (root / "state-exponent-values-f32.bin").write_bytes(
                struct.pack("<f", 1) * (128 * 32) + struct.pack("<f", .5) * (2 * 32))
            command = [str(self.binary), str(root), str(root), "128"]
            def run():
                result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
                return next(v for v in json.loads(result.stdout)["variants"] if v["name"] == "captured_exp_fma")
            before = run()
            for surface in ("checkpoints", "v_new", "terminal", "isolated_step_state"):
                self.assertEqual(before[surface]["raw_bit_mismatches"], 0)
            changed = checkpoints[:FRAME * 4] + struct.pack("<I", 0x3F000001) + checkpoints[FRAME * 4 + 4:]
            (root / "checkpoint-f32.bin").write_bytes(changed)
            after = run()
            self.assertEqual(after["checkpoints"]["raw_bit_mismatches"], 1)
            self.assertEqual(after["checkpoints"]["bf16_bit_mismatches"], 0)
            self.assertEqual(after["terminal"], before["terminal"])
            self.assertGreater(after["isolated_step_state"]["raw_bit_mismatches"], 0)

    def test_partial_chunk_is_rejected_before_reading_files(self):
        result = subprocess.run([str(self.binary), "missing", "missing", "65"],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid tokens", result.stderr)


if __name__ == "__main__":
    unittest.main()
