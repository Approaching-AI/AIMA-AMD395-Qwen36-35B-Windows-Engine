import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capture_sm121_exp2_table as table


class Exp2TableTests(unittest.TestCase):
    def test_preflight_requires_no_gpu_or_numpy_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for module in ("numpy", "torch", "triton"):
                (root / (module + ".py")).write_text('raise RuntimeError("GPU build library must not import")\n')
            command = [sys.executable, str(ROOT / "scripts/capture_sm121_exp2_table.py"),
                       "--output-dir", str(root / "out"), "--source-commit", "0" * 40]
            p = subprocess.run(command, env=dict(os.environ, PYTHONPATH=str(root)),
                               capture_output=True, text=True, timeout=5)
            self.assertEqual(p.returncode, 0, p.stderr)
            import json
            record = json.loads(p.stdout)
            self.assertFalse(record["kernel_executed"])
            self.assertFalse(record["model_or_prompt_inputs"])
            self.assertEqual(record["exhaustive_input_count"], 0x7F800001)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "packing uses offline builder NumPy; also tested in CUDA image on CPU")
    def test_lossless_packing_all_widths_and_offset_bound(self):
        import numpy as np
        pages = np.stack([np.full(256, 7, dtype=np.uint32),
                          np.arange(256, dtype=np.uint32) + 9,
                          np.arange(256, dtype=np.uint32) * 200 + 80000,
                          np.arange(256, dtype=np.uint32) * 100000 + 90000])
        directory, chunks, end = table.pack_pages(pages, 0)
        payload = b"".join(chunks)
        self.assertEqual(end, len(payload))
        for page in range(4):
            base, tag = struct.unpack_from("<II", directory, page * 8)
            kind, offset = tag >> 30, tag & 0x3FFFFFFF
            self.assertEqual(kind, page)
            for index in range(256):
                delta = 0 if kind == 0 else struct.unpack_from("<" + ("B", "H", "I")[kind - 1], payload,
                                                               offset + index * (1, 2, 4)[kind - 1])[0]
                self.assertEqual(base + delta, int(pages[page, index]))
        with self.assertRaisesRegex(ValueError, "30-bit"):
            table.pack_pages(pages, (1 << 30) - 1)


if __name__ == "__main__":
    unittest.main()
