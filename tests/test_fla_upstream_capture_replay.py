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
SYMBOLS = {"solve": "_fla_solve_tril_64_kernel", "wu": "_fla_recompute_w_u_kernel", "state": "_fla_chunk_state_kernel"}


class FlaUpstreamIntegrationContractTests(unittest.TestCase):
    def test_builder_fingerprints_probe_and_supervisor_detects_it(self) -> None:
        builder = (ROOT / "scripts/baiying_build_fla_gdn.ps1").read_text()
        self.assertIn("$upstreamReplay = Join-Path", builder)
        self.assertIn("$outputReplay, $upstreamReplay", builder)
        self.assertIn("fla-upstream-capture-replay.exe", builder)
        guard = (ROOT / "scripts/baiying_guarded_inference.ps1").read_text()
        self.assertIn("fla-upstream-capture-replay|fla-output-capture-replay", guard)
        self.assertIn("allocation > 512u * 1024u * 1024u", SOURCE.read_text())


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
        for name, size in sizes.items():
            with (directory / ("full-" + name + ".bin")).open("wb") as file:
                file.truncate(size)
        return directory

    def run_probe(self, stage, directory, source_tokens, tokens, **environment):
        env = {k: v for k, v in os.environ.items() if k not in ("QRT_TEST_FAKE_DISPATCH_MS", "QRT_FLA_UPSTREAM_DUMP_Q64_DIR")}
        env.update(environment)
        return subprocess.run([str(self.executable), stage, "unused.hsaco", SYMBOLS[stage], "128", "0",
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

    def test_prefix_view_preserves_parent_shape_and_uses_next_chunk_reference(self) -> None:
        result = self.run_probe("state", self.fixture(128), 128, 64)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual((record["source_tokens"], record["tokens"]), (128, 64))
        self.assertIn("next-chunk-state-bf16", record["surfaces"])
        self.assertNotIn("final-state-f32", record["surfaces"])

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
