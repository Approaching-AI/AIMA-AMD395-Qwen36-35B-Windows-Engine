import importlib.util
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch


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

    def test_captured_pre_decay_dot_is_not_silently_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("output-bf16", "state-f32", "q-normalized-bf16", "k-normalized-bf16",
                     "g-cumsum-f32", "a-f32", "a-inverse-bf16", "w-bf16", "u-bf16",
                     "chunk-state-bf16", "v-new-bf16", "a-dot-f32")
            files = {}
            for name in names:
                source = root / (name + ".bin")
                payload = struct.pack("<H", 0x3f80) if name.endswith("bf16") else struct.pack("<f", 1.0)
                source.write_bytes(payload)
                files[name] = {"bytes": len(payload), "sha256": MODULE.fingerprint(source)}
                prefix = "native" if name in names[:2] else "stage"
                (root / (prefix + "-" + name + ".bin")).write_bytes(
                    struct.pack("<f", 2.0) if name == "a-dot-f32" else payload)
            manifest = root / "reference.json"
            manifest.write_text(json.dumps({"results": [{"forward_method": "forward_native", "tokens": 64, "files": files}]}))
            output = root / "comparison.json"
            argv = ["compare", "--reference-manifest", str(manifest), "--reference-dir", str(root),
                    "--candidate-prefix", str(root / "native"), "--stage-prefix", str(root / "stage"), "--output", str(output)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
                MODULE.main()
            self.assertEqual(caught.exception.code, 3)
            record = json.loads(output.read_text())
            self.assertEqual(record["surfaces"]["a-dot-f32"]["mismatch_count"], 1)
            self.assertFalse(record["inference_acceptance"])


if __name__ == "__main__":
    unittest.main()
