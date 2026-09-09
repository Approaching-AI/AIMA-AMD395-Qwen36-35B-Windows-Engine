import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("state_prefix", ROOT / "scripts/prepare_fla_state_prefix.py")
prefix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefix)


class StatePrefixTests(unittest.TestCase):
    def fixture(self, root: Path) -> None:
        for index, (name, (suffix, dtype, shape)) in enumerate(prefix.layouts(129).items()):
            filename = f"full-{name}-{suffix}.bin"
            size = math.prod(shape) * (2 if suffix == "bf16" else 4)
            data = bytes([index]) * size
            if name == "chunk-state":
                data = b"\x00" * (size // 3) + b"\x01" * (size // 3) + b"\x02" * (size // 3)
            (root / filename).write_bytes(data)
            descriptor = dict(surface="full-" + name, file=dict(path="/capture/" + filename,
                              dtype=dtype, shape=shape, bytes=size, sha256=hashlib.sha256(data).hexdigest()))
            (root / f"detail-full-{name}.json").write_text(json.dumps(descriptor))

    def test_prefix_preserves_inputs_and_selects_the_following_bf16_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            output = root / "prefix"
            record = prefix.prepare(root, output, 64)
            self.assertEqual((record["source_tokens"], record["tokens"], record["chunks"]), (129, 64, 1))
            self.assertEqual(len(record["parent_files"]), 7)
            self.assertEqual(len(record["files"]), 8)
            self.assertFalse(record["kernel_executed"])
            self.assertFalse(record["reference_raw_f32_checkpoints_available"])
            self.assertFalse(record["reference_checkpoints_are_recurrent_inputs"])
            self.assertTrue(record["future_raw_trace_requires_saved_bf16_parity"])
            self.assertEqual((output / "next-chunk-state-bf16.bin").read_bytes(), b"\x01" * 1048576)
            for name, item in record["files"].items():
                data = (output / item["file"]).read_bytes()
                self.assertEqual(len(data), item["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
                self.assertEqual(item["reference_only"], name in ("v-new-bf16", "chunk-state-bf16", "next-chunk-state-bf16"))
            # Nonzero initial state must be copied, not silently zeroed.
            self.assertEqual((output / "initial_state-f32.bin").read_bytes(), (root / "full-initial_state-f32.bin").read_bytes())
            with self.assertRaisesRegex(ValueError, "already exists"):
                prefix.prepare(root, output, 64)

    def test_all_parent_bytes_are_verified_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root / "full-w-bf16.bin"
            data = bytearray(path.read_bytes())
            data[-1] ^= 1  # Outside the exported prefix: still must be checked.
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                prefix.prepare(root, root / "prefix", 64)
            self.assertFalse((root / "prefix").exists())

    def test_incomplete_or_oversized_prefix_and_inconsistent_layout_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            for tokens in (0, 1, 63, 65, 192, 1088):
                with self.assertRaises(ValueError):
                    prefix.prepare(root, root / "prefix", tokens)
            path = root / "detail-full-u.json"
            item = json.loads(path.read_bytes())
            item["file"]["dtype"] = "float32"
            path.write_text(json.dumps(item))
            with self.assertRaisesRegex(ValueError, "layout mismatch"):
                prefix.prepare(root, root / "prefix", 64)
            self.assertFalse((root / "prefix").exists())


if __name__ == "__main__":
    unittest.main()
