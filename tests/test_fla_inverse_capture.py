import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capture_fla_inverse as capture


class InverseCaptureTests(unittest.TestCase):
    def fixture(self, root):
        files = {}
        for name, width, dtype in (("a-input", 4, "float32"), ("a-reference", 2, "bfloat16")):
            path = root / (name + ".bin")
            path.write_bytes(b"\0" * (64 * 32 * 64 * width))
            files[name] = dict(file=path.name, bytes=path.stat().st_size, dtype=dtype,
                               shape=[1, 64, 32, 64], sha256=capture.file_sha(path))
        manifest = dict(tokens=64, files=files)
        self.save(root, manifest)
        source = root / "source.py"
        source.write_text('raise RuntimeError("top-level import forbidden")\n@unknown_autotuner()\n'
                          'def ' + capture.FUNCTION + '(A, Ai, T, H: tl.constexpr):\n    return A\n')
        return manifest, source

    def save(self, root, manifest):
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return capture.file_sha(path)

    def test_default_preflight_and_mismatched_host_are_cpu_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, source = self.fixture(root)
            for name in ("torch", "triton", "numpy"):
                (root / (name + ".py")).write_text('raise RuntimeError("GPU library import forbidden")\n')
            command = [sys.executable, str(ROOT / "scripts/capture_fla_inverse.py"),
                       "--input-dir", str(root), "--manifest-sha256", self.save(root, manifest),
                       "--source", str(source), "--source-sha256", capture.file_sha(source),
                       "--source-commit", "0" * 40, "--output-dir", str(root / "preflight")]
            environment = dict(os.environ, PYTHONPATH=str(root))
            result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertFalse(record["kernel_executed"])
            self.assertFalse(record["model_loaded"])
            self.assertFalse(record["inference_acceptance"])
            again = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
            self.assertIn("existing output", again.stderr)
            command[-1] = str(root / "wrong-host")
            result = subprocess.run(command + ["--execute", "--expected-host", "not-this-host"],
                                    env=environment, capture_output=True, text=True, timeout=10)
            self.assertIn("execution host mismatch", result.stderr)
            self.assertFalse((root / "wrong-host").exists())

    def test_extracted_function_source_is_real_and_never_runs_autotuners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source = self.fixture(root)
            triton = types.SimpleNamespace(jit=lambda f, **kw: (inspect.getsource(f), kw))
            seen, options = capture.load_kernel(source, root, triton, types.SimpleNamespace(constexpr=object()))
            self.assertEqual(seen, capture.extract(source))
            self.assertEqual(options, dict(do_not_specialize=["T"]))

    def test_input_hash_and_finiteness_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "manifest fingerprint"):
                capture.validate(root, "0" * 64)
            item = manifest["files"]["a-input"]
            item["file"] = "../outside.bin"
            with self.assertRaisesRegex(ValueError, "layout"):
                capture.validate(root, self.save(root, manifest))
            item["file"] = "a-input.bin"
            path = root / item["file"]
            raw = bytearray(path.read_bytes()); raw[:4] = b"\0\0\xc0\x7f"; path.write_bytes(raw)
            item["sha256"] = capture.file_sha(path)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                capture.validate(root, self.save(root, manifest))


if __name__ == "__main__":
    unittest.main()
