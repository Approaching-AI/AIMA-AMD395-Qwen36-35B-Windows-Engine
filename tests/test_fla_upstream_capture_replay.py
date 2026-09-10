import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/providers/gdn/fla_upstream_capture_replay.cpp"
SYMBOLS = {"solve-blackwell": "native-blackwell-solve", "norm-blackwell": "native-blackwell-norm", "solve": "_fla_solve_tril_64_kernel", "wu": "_fla_recompute_w_u_kernel", "wu-blackwell": "native-blackwell-wu", "state": "_fla_chunk_state_kernel", "state-blackwell": "native-blackwell-state"}


class FlaUpstreamIntegrationContractTests(unittest.TestCase):
    def test_builder_fingerprints_probe_and_supervisor_detects_it(self) -> None:
        builder = (ROOT / "scripts/baiying_build_fla_gdn.ps1").read_text()
        self.assertIn("$upstreamReplay = Join-Path", builder)
        self.assertIn("$outputReplay, $upstreamReplay", builder)
        self.assertIn("fla-upstream-capture-replay.exe", builder)
        guard = (ROOT / "scripts/baiying_guarded_inference.ps1").read_text()
        self.assertIn("fla-upstream-capture-replay|fla-output-capture-replay", guard)
        self.assertIn("allocation > 512u * 1024u * 1024u", SOURCE.read_text())

    def test_native_state_uses_compiler_owned_launches_and_explicit_runtime_opt_in(self) -> None:
        source = (ROOT / "native/providers/gdn/blackwell_state.cpp").read_text()
        self.assertEqual(source.count("hipLaunchKernelGGL("), 2)
        self.assertNotIn("hipModuleLaunchKernel", source)
        self.assertIn("fmaf(initial[index]", source)
        self.assertIn("if (token == 0) h[state_index] = state", source)
        self.assertIn("if (token >= count)", source)
        builder = (ROOT / "scripts/baiying_build_fla_gdn.ps1").read_text()
        self.assertIn("$(Quote-Arg $upstreamReplay) $(Quote-Arg $blackwellState)", builder)
        runtime = (ROOT / "native/providers/gdn/qrt_fla_chunk_gdn_q8192_provider.cpp").read_text()
        self.assertIn('std::getenv("QRT_FLA_GDN_STATE_BLACKWELL")', runtime)
        self.assertIn('setting && std::strcmp(setting, "1") == 0', runtime)
        self.assertIn('if (blackwell_state_enabled())', runtime)
        self.assertIn('if (initial != state)', runtime)
        self.assertIn('kBlackwellStateScratchBytes', runtime)


@unittest.skipUnless(shutil.which("c++"), "requires the portable C++ compiler")
class FlaUpstreamHostSafetyTests(unittest.TestCase):
    """Checks real host code against fake HIP; does NOT validate GPU arithmetic."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.executable = cls.root / "host-replay-test"
        subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                        "-I" + str(ROOT / "tests/native/fla_replay_fake_hip"), str(SOURCE),
                        str(ROOT / "tests/native/fla_state_fake_launch.cpp"),
                        "-o", str(cls.executable)], check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def fixture(self, source_tokens: int) -> Path:
        directory = self.root / ("source-" + str(source_tokens))
        directory.mkdir(exist_ok=True)
        state = 32 * 128 * 128
        sizes = {"a-f32": source_tokens * 2048 * 4, "a-inverse-bf16": source_tokens * 2048 * 2,
                 "k-normalized-bf16": source_tokens * 2048 * 2, "g-cumsum-f32": source_tokens * 32 * 4,
                 "beta-bf16": source_tokens * 32 * 2, "initial_state-f32": state * 4,
                 "native-final-state-f32": state * 4, "chunk-state-bf16": ((source_tokens + 63) // 64) * state * 2}
        sizes.update({name: source_tokens * 4096 * 2 for name in ("v-bf16", "w-bf16", "u-bf16", "v-new-bf16")})
        sizes.update({name: source_tokens * 2048 * 2 for name in ("q-bf16", "k-bf16", "q-normalized-bf16")})
        for name, size in sizes.items():
            with (directory / ("full-" + name + ".bin")).open("wb") as file:
                file.truncate(size)
        return directory

    def run_probe(self, stage, directory, source_tokens, tokens, **environment):
        env = {k: v for k, v in os.environ.items() if k not in ("QRT_TEST_FAKE_DISPATCH_MS", "QRT_FLA_UPSTREAM_DUMP_Q64_DIR")}
        env.update(environment)
        native = stage in ("state-blackwell", "wu-blackwell", "norm-blackwell", "solve-blackwell")
        return subprocess.run([str(self.executable), stage, "-" if native else "unused.hsaco", SYMBOLS[stage], "256" if native else "128", "0",
                               str(directory), str(source_tokens), str(tokens)], env=env,
                              capture_output=True, text=True, timeout=10)

    def test_all_stage_launches_have_complete_abi_and_in_bounds_segmented_buffers(self) -> None:
        directory = self.fixture(1025)
        for stage, slots in (("solve", 5), ("wu", 10), ("state", 11)):
            result = self.run_probe(stage, directory, 1025, 1025)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertFalse(record["inference_acceptance"])
            self.assertFalse(record["model_loaded"])
            self.assertEqual(record["segments"], 2)
            self.assertIn(f"FAKE_HIP launch slots={slots} tokens=1024", result.stderr)
            self.assertIn(f"FAKE_HIP launch slots={slots} tokens=64", result.stderr)

    def test_native_wu_uses_production_inplace_alias_and_bounded_tail(self) -> None:
        result = self.run_probe("wu-blackwell", self.fixture(1025), 1025, 1025)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count("FAKE_HIP blackwell_wu_inplace tokens=64\n"), 16)
        self.assertIn("FAKE_HIP blackwell_wu_inplace tokens=1\n", result.stderr)
        self.assertEqual(json.loads(result.stdout)["segments"], 17)

    def test_normalization_segments_raw_qkv_and_preserves_one_token_tail(self) -> None:
        result = self.run_probe("norm-blackwell", self.fixture(1025), 1025, 1025)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FAKE_HIP blackwell_norm tokens=1024\n", result.stderr)
        self.assertIn("FAKE_HIP blackwell_norm tokens=1\n", result.stderr)
        self.assertEqual(json.loads(result.stdout)["segments"], 2)
        result = self.run_probe("norm-blackwell", self.fixture(1025), 1025, 1025, QRT_TEST_FAKE_DISPATCH_MS="101")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr.count("FAKE_HIP blackwell_norm"), 1)

    def test_inverse_segments_and_one_token_tail_stop_on_admission_failure(self) -> None:
        result = self.run_probe("solve-blackwell", self.fixture(1025), 1025, 1025)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FAKE_HIP blackwell_inverse tokens=1024\n", result.stderr)
        self.assertIn("FAKE_HIP blackwell_inverse tokens=1\n", result.stderr)
        self.assertEqual(json.loads(result.stdout)["segments"], 2)
        result = self.run_probe("solve-blackwell", self.fixture(1025), 1025, 1025, QRT_TEST_FAKE_DISPATCH_MS="101")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr.count("FAKE_HIP blackwell_inverse"), 1)

    def test_native_wu_stops_before_next_dispatch_on_admission_failure(self) -> None:
        result = self.run_probe("wu-blackwell", self.fixture(128), 128, 128, QRT_TEST_FAKE_DISPATCH_MS="101")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr.count("FAKE_HIP blackwell_wu_inplace"), 1)

    def test_prefix_view_preserves_parent_shape_and_uses_next_chunk_reference(self) -> None:
        result = self.run_probe("state", self.fixture(128), 128, 64)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual((record["source_tokens"], record["tokens"]), (128, 64))
        self.assertIn("next-chunk-state-bf16", record["surfaces"])
        self.assertNotIn("final-state-f32", record["surfaces"])

    def test_native_state_uses_two_compiler_owned_calls_per_chunk_and_logical_tail(self) -> None:
        result = self.run_probe("state-blackwell", self.fixture(1025), 1025, 1025)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["segments"], 34)
        self.assertLess(record["allocation_bytes"], 512 * 1024 * 1024)
        self.assertEqual(result.stderr.count("FAKE_HIP blackwell_project tokens=64"), 16)
        self.assertEqual(result.stderr.count("FAKE_HIP blackwell_update tokens=64"), 16)
        self.assertIn("FAKE_HIP blackwell_project tokens=1\n", result.stderr)
        self.assertIn("FAKE_HIP blackwell_update tokens=1\n", result.stderr)
        self.assertNotIn("FAKE_HIP launch", result.stderr)

    def test_native_invalid_arguments_return_before_any_hip_dispatch(self) -> None:
        result = subprocess.run([str(self.executable), "--blackwell-host-only"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["cases"], 8)
        self.assertEqual(json.loads(result.stdout)["gpu_dispatches"], 0)
        self.assertNotIn("FAKE_HIP", result.stderr)

    def test_native_admission_checks_projection_before_submitting_update(self) -> None:
        result = self.run_probe("state-blackwell", self.fixture(128), 128, 128, QRT_TEST_FAKE_DISPATCH_MS="101")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr.count("FAKE_HIP blackwell_project"), 1)
        self.assertNotIn("FAKE_HIP blackwell_update", result.stderr)
        self.assertIn("100 ms dispatch admission exceeded", result.stderr)

    def test_invalid_shape_nonfinite_and_short_input_fail_before_hip(self) -> None:
        directory = self.fixture(128)
        for source, view in ((0, 64), (128, 0), (128, 65), (8193, 64), (128, 129)):
            result = self.run_probe("solve", directory, source, view)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("FAKE_HIP", result.stderr)
        path = directory / "full-a-f32.bin"
        with path.open("r+b") as file:
            file.write(struct.pack("<f", float("nan")))
        result = self.run_probe("solve", directory, 128, 64)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("FAKE_HIP", result.stderr)
        path.write_bytes(b"")
        result = self.run_probe("solve", directory, 128, 64)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("FAKE_HIP", result.stderr)

    def test_dispatch_admission_stops_before_next_segment(self) -> None:
        result = self.run_probe("solve", self.fixture(1025), 1025, 1025, QRT_TEST_FAKE_DISPATCH_MS="101")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr.count("FAKE_HIP launch"), 1)
        self.assertIn("100 ms dispatch admission exceeded", result.stderr)

    def test_dump_is_q64_only_and_never_overwrites(self) -> None:
        directory = self.fixture(128)
        destination = self.root / "dumps"
        destination.mkdir()
        env = {"QRT_FLA_UPSTREAM_DUMP_Q64_DIR": str(destination)}
        result = self.run_probe("solve", directory, 128, 128, **env)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("FAKE_HIP", result.stderr)
        result = self.run_probe("solve", directory, 128, 64, **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = destination / "solve-a-inverse-bf16.bin"
        self.assertEqual(output.stat().st_size, 64 * 2048 * 2)
        result = self.run_probe("solve", directory, 128, 64, **env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing capture overwrite", result.stderr)


if __name__ == "__main__":
    unittest.main()
