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
import capture_fla_l2norm as capture


class L2NormCaptureTests(unittest.TestCase):
    def fixture(self, root):
        payload = b"\0" * (16 * 128 * 2)
        files = {}
        for name in ("q-input", "q-reference", "k-input", "k-reference"):
            (root / (name + ".bin")).write_bytes(payload)
            files[name] = dict(file=name + ".bin", bytes=len(payload), dtype="bfloat16",
                               shape=[1, 1, 16, 128], sha256=capture.digest(payload))
        manifest = dict(tokens=1, files=files)
        self.save(root, manifest)
        source = root / "source.py"
        source.write_text('raise RuntimeError("top level must not run")\n'
                          '@unknown_decorator()\ndef l2norm_fwd_kernel2(X, Y, eps, M, N: tl.constexpr, BD: tl.constexpr, MBLOCK: tl.constexpr):\n'
                          '    square_sum = X\n    rsqrt = Y\n    row_idx = M\n    xmask = True\n')
        return manifest, source

    def save(self, root, manifest):
        raw = json.dumps(manifest).encode()
        (root / "manifest.json").write_bytes(raw)
        return capture.digest(raw)

    def test_preflight_is_cpu_only_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, source = self.fixture(root)
            for name in ("torch", "triton"):
                (root / (name + ".py")).write_text('raise RuntimeError("GPU import forbidden")\n')
            command = [sys.executable, str(ROOT / "scripts/capture_fla_l2norm.py"),
                       "--input-dir", str(root), "--manifest-sha256", self.save(root, manifest),
                       "--source", str(source), "--source-sha256", capture.digest(source.read_bytes()),
                       "--source-commit", "0" * 40, "--output-dir", str(root / "result")]
            environment = dict(os.environ, PYTHONPATH=str(root))
            result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertFalse(record["kernel_executed"])
            self.assertFalse(record["model_loaded"])
            self.assertFalse(record["inference_acceptance"])
            before = (root / "result/capture.json").read_bytes()
            again = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(before, (root / "result/capture.json").read_bytes())
            command[-1] = str(root / "wrong-host")
            result = subprocess.run(command + ["--execute", "--expected-host", "not-this-host"],
                                    env=environment, text=True, capture_output=True, timeout=10)
            self.assertIn("execution host mismatch", result.stderr)
            self.assertFalse((root / "wrong-host").exists())

    def test_source_seen_by_jit_matches_generated_function_without_top_level(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source = self.fixture(root)
            fake_triton = types.SimpleNamespace(jit=lambda function: inspect.getsource(function))
            fake_tl = types.SimpleNamespace(constexpr=object())
            for trace in (False, True):
                seen, source_sha = capture.load_kernel(source, trace, root, fake_triton, fake_tl)
                self.assertEqual(seen, capture.kernel_source(source, trace))
                self.assertEqual(source_sha, capture.digest(seen.encode()))
                self.assertNotIn("unknown_decorator", seen)
                self.assertEqual("tl.store(RawSum" in seen, trace)
                self.assertEqual("tl.store(RawRsqrt" in seen, trace)

    def test_manifest_and_file_identity_shape_and_finiteness_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "manifest fingerprint"):
                capture.validate(root, "0" * 64)
            item = manifest["files"]["q-input"]
            item["file"] = "../outside.bin"
            with self.assertRaisesRegex(ValueError, "basename"):
                capture.validate(root, self.save(root, manifest))
            item["file"] = "q-input.bin"
            item["shape"] = [1, 1, 32, 64]
            with self.assertRaisesRegex(ValueError, "shape/type"):
                capture.validate(root, self.save(root, manifest))
            item["shape"] = [1, 1, 16, 128]
            raw = b"\xc0\x7f" + b"\0" * (item["bytes"] - 2)
            (root / item["file"]).write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "fingerprint/nonfinite"):
                capture.validate(root, self.save(root, manifest))
            item["sha256"] = capture.digest(raw)
            with self.assertRaisesRegex(ValueError, "fingerprint/nonfinite"):
                capture.validate(root, self.save(root, manifest))

    def test_saved_baseline_failure_prevents_all_trace_dispatch(self):
        payloads = {"q-reference": b"q", "k-reference": b"k"}
        for failed in ("q", "k"):
            calls = []
            def launch(name, trace):
                calls.append((name, trace))
                return dict(normalized=b"bad" if name == failed else name.encode())
            with self.assertRaisesRegex(ValueError, "no trace dispatch"):
                capture.run_controls(launch, payloads)
            self.assertTrue(all(not trace for _, trace in calls))
            self.assertEqual(calls[-1], (failed, False))

    def test_trace_must_preserve_saved_bf16_and_stops_on_first_failure(self):
        payloads = {"q-reference": b"q", "k-reference": b"k"}
        calls = []
        def launch(name, trace):
            calls.append((name, trace))
            return dict(normalized=b"changed" if trace else name.encode())
        with self.assertRaisesRegex(ValueError, "no further dispatch"):
            capture.run_controls(launch, payloads)
        self.assertEqual(calls, [("q", False), ("k", False), ("q", True)])
        baselines, traces = capture.run_controls(lambda name, trace: dict(normalized=name.encode()), payloads)
        self.assertEqual(baselines, traces)


if __name__ == "__main__":
    unittest.main()
