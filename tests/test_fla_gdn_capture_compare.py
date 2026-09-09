import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fla_compare", ROOT / "scripts/compare_fla_gdn_capture.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FlaGdnCompareTests(unittest.TestCase):
    def test_bf16_difference_is_not_hidden_by_matching_self_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.write_bytes(struct.pack("<HHH", 0x3f80, 0x4000, 0xbf80))
            b.write_bytes(struct.pack("<HHH", 0x3f80, 0x4001, 0xbf80))
            result = MODULE.compare(a, b, "bf16")
            self.assertEqual(result["elements"], 3)
            self.assertEqual(result["mismatch_count"], 1)
            self.assertEqual(result["maximum_absolute_error"], 0.015625)
            self.assertFalse(result["exact_match"])

    def test_nonfinite_and_wrong_length_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.write_bytes(struct.pack("<ff", float("nan"), 1.0))
            b.write_bytes(a.read_bytes())
            result = MODULE.compare(a, b, "f32")
            self.assertEqual(result["nonfinite_count"], 1)
            self.assertFalse(result["exact_match"])
            b.write_bytes(b"short")
            with self.assertRaises(ValueError):
                MODULE.compare(a, b, "f32")


if __name__ == "__main__":
    unittest.main()
