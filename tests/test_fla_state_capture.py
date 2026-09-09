import importlib.util
import json
import math
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("state_capture", ROOT / "scripts/capture_fla_state_prefix.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class StateCaptureTests(unittest.TestCase):
    def fixture(self, root: Path):
        files = {}
        expected = {name + "-" + suffix: (dtype, shape, name in ("v-new", "chunk-state"))
                    for name, (suffix, dtype, shape) in capture.layouts(64).items()}
        expected["next-chunk-state-bf16"] = ("bfloat16", [1, 32, 128, 128], True)
        for name, (dtype, shape, reference_only) in expected.items():
            data = b"\0" * (math.prod(shape) * (2 if dtype == "bfloat16" else 4))
            (root / (name + ".bin")).write_bytes(data)
            files[name] = dict(file=name + ".bin", dtype=dtype, shape=shape, bytes=len(data),
                               sha256=capture.sha256(data), reference_only=reference_only)
        manifest = dict(kind="cpu_prepared_gdn_state_prefix", tokens=64, chunks=1,
                        files=files, total_bytes=sum(item["bytes"] for item in files.values()),
                        reference_checkpoints_are_recurrent_inputs=False,
                        future_raw_trace_requires_saved_bf16_parity=True)
        self.save_manifest(root, manifest)
        return manifest

    def save_manifest(self, root, manifest):
        (root / "manifest.json").write_text(json.dumps(manifest))
        return capture.sha256((root / "manifest.json").read_bytes())

    def arguments(self, root, output):
        source = root / "not-imported.py"
        source.write_text('raise RuntimeError("source must not execute in preflight")\n')
        return [sys.executable, str(ROOT / "scripts/capture_fla_state_prefix.py"),
                "--prefix-dir", str(root), "--manifest-sha256", capture.sha256((root / "manifest.json").read_bytes()),
                "--source", str(source), "--source-sha256", capture.sha256(source.read_bytes()),
                "--output-dir", str(output), "--source-commit", "0" * 40]

    def test_preflight_never_imports_gpu_libraries_or_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            for name in ("torch", "triton"):
                (root / (name + ".py")).write_text('raise RuntimeError("GPU import forbidden")\n')
            output = root / "preflight"
            command = self.arguments(root, output)
            environment = dict(os.environ, PYTHONPATH=str(root))
            result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertFalse(record["kernel_executed"])
            self.assertFalse(record["inference_acceptance"])
            self.assertFalse(record["model_loaded"])
            self.assertEqual(record["tokens"], 64)
            self.assertEqual(len(record["helper_sha256"]), 2)
            before = (output / "capture.json").read_bytes()
            again = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("never overwrite", again.stderr)
            self.assertEqual((output / "capture.json").read_bytes(), before)

    def test_execute_requires_the_explicit_matching_host_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            output = root / "wrong-host"
            command = self.arguments(root, output) + ["--execute", "--expected-host", "deliberately-not-this-host"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("execution host mismatch", result.stderr)
            self.assertFalse(output.exists())

    def test_hash_roles_and_path_contract_fail_before_gpu_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "manifest fingerprint"):
                capture.validate_prefix(root, "0" * 64)
            manifest["files"]["v-new-bf16"]["reference_only"] = False
            with self.assertRaisesRegex(ValueError, "layout mismatch"):
                capture.validate_prefix(root, self.save_manifest(root, manifest))
            manifest["files"]["v-new-bf16"]["reference_only"] = True
            manifest["files"]["w-bf16"]["file"] = "../outside.bin"
            with self.assertRaisesRegex(ValueError, "layout mismatch"):
                capture.validate_prefix(root, self.save_manifest(root, manifest))

    def test_nonfinite_payload_is_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            item = manifest["files"]["w-bf16"]
            data = b"\xc1\x7f" + b"\0" * (item["bytes"] - 2)
            (root / item["file"]).write_bytes(data)
            item["sha256"] = capture.sha256(data)
            with self.assertRaisesRegex(ValueError, "finite-value"):
                capture.validate_prefix(root, self.save_manifest(root, manifest))

    def pair_fixture(self):
        baseline = dict(v_new=struct.pack("<H", 0x3F80), h=b"\0\0", final=struct.pack("<f", 1.0))
        raw = dict(v_new=baseline["v_new"], h=struct.pack("<f", 0.0), final=baseline["final"])
        inputs = {"v-new-bf16": baseline["v_new"], "chunk-state-bf16": baseline["h"],
                  "next-chunk-state-bf16": struct.pack("<H", 0x3F80)}
        return baseline, raw, inputs

    def test_failed_baseline_never_dispatches_raw_trace(self):
        baseline, raw, inputs = self.pair_fixture()
        baseline["h"] = struct.pack("<H", 0x3F80)
        calls = []
        def launch(dtype):
            calls.append(dtype)
            return baseline if dtype == "bf16" else raw
        record, retained = capture.run_pair(launch, inputs)
        self.assertEqual(calls, ["bf16"])
        self.assertFalse(record["raw_trace_valid"])
        self.assertIsNone(retained)

    def test_trace_requires_raw_terminal_identity_not_just_bf16_rounding(self):
        baseline, raw, inputs = self.pair_fixture()
        record, retained = capture.run_pair(lambda dtype: baseline if dtype == "bf16" else raw, inputs)
        self.assertTrue(record["raw_trace_valid"])
        self.assertIs(retained, raw)
        raw["final"] = struct.pack("<I", 0x3F800001)
        record, retained = capture.run_pair(lambda dtype: baseline if dtype == "bf16" else raw, inputs)
        self.assertFalse(record["raw_trace_valid"])
        self.assertIsNone(retained)

    def test_bf16_conversion_is_nearest_even_and_rejects_nonfinite(self):
        values = (0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000)
        self.assertEqual(capture.rounded_bf16(struct.pack("<4I", *values)), struct.pack("<4H", 0x3F80, 0x3F82, 0xBF80, 0xBF82))
        for value in (0x7F800000, 0x7FC00000, 0xFF800000):
            with self.assertRaises(ValueError):
                capture.rounded_bf16(struct.pack("<I", value))

    def test_extraction_does_not_execute_source_top_level_or_decorators(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            source.write_text('raise RuntimeError("top-level forbidden")\n@unknown_autotuner()\n'
                              'def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(T):\n    return exp(T)\n')
            fake_triton = types.SimpleNamespace(jit=lambda function, **kwargs: (function, kwargs))
            fake_tl = types.SimpleNamespace(exp=lambda value: value + 2)
            function, options = capture.load_state_kernel(source, fake_triton, fake_tl)
            self.assertEqual(function(3), 5)
            self.assertEqual(options, {"do_not_specialize": ["T"]})

    def test_worker_cannot_bypass_its_supervisor(self):
        with self.assertRaisesRegex(ValueError, "no live matching supervisor"):
            capture.arm_parent_death(0)

    @unittest.skipUnless(os.name == "posix", "reference execution is POSIX-only")
    def test_supervisor_deadline_kills_and_reaps_its_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "pid"
            command = [sys.executable, "-c", "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)", str(pidfile)]
            self.assertEqual(capture.supervise(command, 0.5), 124)
            pid = int(pidfile.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    @unittest.skipUnless(sys.platform == "linux", "requires Linux parent-death signals")
    def test_worker_dies_when_only_its_supervisor_is_killed(self):
        child_code = ("import os,sys,time; sys.path.insert(0,sys.argv[1]); "
                      "from capture_fla_state_prefix import arm_parent_death; "
                      "arm_parent_death(int(sys.argv[2])); print(os.getpid(),flush=True); time.sleep(30)")
        parent_code = ("import os,subprocess,sys,time; "
                       "p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],str(os.getpid())],stdout=subprocess.PIPE,text=True); "
                       "print(p.stdout.readline().strip(),flush=True); time.sleep(30)")
        parent = subprocess.Popen([sys.executable, "-c", parent_code, child_code, str(ROOT / "scripts")],
                                  stdout=subprocess.PIPE, text=True, start_new_session=True)
        child_pid, terminal = None, False
        try:
            self.assertTrue(select.select([parent.stdout], [], [], 5)[0], "worker did not arm its death signal")
            child_pid = int(parent.stdout.readline())
            parent.kill()  # Deliberately do not kill the process group.
            parent.wait(timeout=5)
            for _ in range(100):
                status = Path(f"/proc/{child_pid}/stat")
                try:
                    terminal = status.read_text().split(")", 1)[1].split()[0] == "Z"
                except FileNotFoundError:
                    terminal = True
                if terminal:
                    break
                time.sleep(0.01)
            self.assertTrue(terminal, "orphaned worker survived its supervisor")
        finally:
            if parent.poll() is None:
                os.killpg(parent.pid, signal.SIGKILL)
                parent.wait(timeout=5)
            elif child_pid is not None and not terminal:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            parent.stdout.close()


if __name__ == "__main__":
    unittest.main()
