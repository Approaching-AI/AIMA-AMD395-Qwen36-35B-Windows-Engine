import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("c++"), "requires the portable C++ compiler")
class FlaStateAccumulatorProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.executable = cls.root / "probe"
        subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off",
                        str(ROOT / "tests/native/fla_state_accumulator_probe.cpp"), "-o", str(cls.executable)],
                       check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def fixture(self, captured=False) -> None:
        state = 32 * 128 * 128
        (self.root / "full-k-normalized-bf16.bin").write_bytes(struct.pack("<H", 0x3F80) * (128 * 2048))
        (self.root / "full-u-bf16.bin").write_bytes(struct.pack("<H", 0x3F00) * (128 * 4096))
        (self.root / "full-w-bf16.bin").write_bytes(b"\0" * (128 * 4096 * 2))
        (self.root / "full-v-new-bf16.bin").write_bytes(struct.pack("<H", 0x3F00) * (128 * 4096))
        (self.root / "full-g-cumsum-f32.bin").write_bytes(b"\0" * (128 * 32 * 4))
        (self.root / "full-initial_state-f32.bin").write_bytes(b"\0" * (state * 4))
        (self.root / "full-chunk-state-bf16.bin").write_bytes(b"\0" * (state * 2) + struct.pack("<H", 0x4200) * state)
        (self.root / "full-native-final-state-f32.bin").write_bytes(struct.pack("<f", 64.0) * state)
        if captured:
            for name in ("a-dot", "a"):
                (self.root / ("full-" + name + "-f32.bin")).write_bytes(struct.pack("<f", 1.0) * (128 * 32 * 64))

    def run_probe(self, tokens="128", trajectory=False, captured=False):
        return subprocess.run([str(self.executable), str(self.root), tokens, "-"] + (["--trajectory-sample"] if trajectory or captured else []) + (["--captured-exp-control"] if captured else []),
                              capture_output=True, text=True, timeout=20)

    def single_chunk_fixture(self, captured=False) -> None:
        state = 32 * 128 * 128
        (self.root / "k-normalized-bf16.bin").write_bytes(struct.pack("<H", 0x3F80) * (64 * 2048))
        for name in ("u", "v-new"):
            (self.root / (name + "-bf16.bin")).write_bytes(struct.pack("<H", 0x3F00) * (64 * 4096))
        (self.root / "g-cumsum-f32.bin").write_bytes(b"\0" * (64 * 32 * 4))
        (self.root / "chunk-state-bf16.bin").write_bytes(b"\0" * (state * 2))
        (self.root / "state-f32.bin").write_bytes(struct.pack("<f", 32.0) * state)
        if captured:
            for name in ("a-dot", "a"):
                (self.root / (name + "-f32.bin")).write_bytes(struct.pack("<f", 1.0) * (64 * 32 * 64))

    def run_single_chunk(self, tokens="64", captured=False):
        return subprocess.run([str(self.executable), str(self.root), tokens, "--single-chunk-raw"] +
                              (["--captured-exp-control"] if captured else []),
                              capture_output=True, text=True, timeout=20)

    def test_single_chunk_raw_reference_exposes_bf16_hidden_difference(self) -> None:
        self.single_chunk_fixture()
        # One F32 ULP is hidden by the BF16 conversion; both boundaries must be
        # reported instead of claiming raw-state equality from rounded equality.
        with (self.root / "state-f32.bin").open("r+b") as file:
            file.write(struct.pack("<I", 0x42000001))
        result = self.run_single_chunk()
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["kind"], "cpu_single_chunk_raw_state_attribution")
        self.assertFalse(record["inference_acceptance"])
        self.assertFalse(record["raw_initial_state_available"])
        for variant in record["variants"]:
            self.assertEqual(variant["reference_bf16"]["mismatch_count"], 0)
            raw = variant["reference_raw_f32"]
            self.assertEqual(raw["numeric"]["elements"], 32 * 128 * 128)
            self.assertEqual(raw["numeric"]["mismatch_count"], 1)
            self.assertEqual(raw["bit_mismatch_count"], 1)
            self.assertEqual(raw["first_bit_index"], 0)
            self.assertEqual(raw["first_actual_bits"], 0x42000000)
            self.assertEqual(raw["first_expected_bits"], 0x42000001)
            self.assertEqual(raw["differing_rows"], [{"head": 0, "value": 0, "bit_mismatch_count": 1}])

    def test_single_chunk_captured_control_never_uses_terminal_state_to_infer_exponents(self) -> None:
        self.single_chunk_fixture(captured=True)
        with (self.root / "state-f32.bin").open("r+b") as file:
            file.write(struct.pack("<f", 64.0))
        result = self.run_single_chunk(captured=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertTrue(record["captured_exp_control"]["reference_derived"])
        self.assertFalse(record["captured_exp_control"]["production_implementation"])
        self.assertEqual(record["captured_exp_calls"], record["captured_exp_hits"])
        self.assertEqual(record["changed_gated_bf16_product_count"], 0)
        for variant in record["variants"]:
            self.assertEqual(variant["reference_raw_f32"]["bit_mismatch_count"], 1)

    def test_single_chunk_raw_stats_distinguish_signed_zero(self) -> None:
        self.single_chunk_fixture()
        (self.root / "k-normalized-bf16.bin").write_bytes(b"\0" * (64 * 2048 * 2))
        (self.root / "state-f32.bin").write_bytes(struct.pack("<I", 0x80000000) + b"\0" * ((32 * 128 * 128 - 1) * 4))
        result = self.run_single_chunk()
        self.assertEqual(result.returncode, 0, result.stderr)
        for variant in json.loads(result.stdout)["variants"]:
            raw = variant["reference_raw_f32"]
            self.assertEqual(raw["numeric"]["mismatch_count"], 0)
            self.assertEqual(raw["bit_mismatch_count"], 1)
            self.assertEqual(raw["first_actual_bits"], 0)
            self.assertEqual(raw["first_expected_bits"], 0x80000000)

    def test_single_chunk_rejects_wrong_shape_seed_residual_and_nonfinite(self) -> None:
        self.single_chunk_fixture()
        for tokens in ("0", "63", "65", "128", "8192", "64bad"):
            self.assertEqual(self.run_single_chunk(tokens).returncode, 2)
        for name, data, error in (
            ("chunk-state-bf16.bin", struct.pack("<H", 0x3F80), "requires captured zero"),
            ("v-new-bf16.bin", struct.pack("<H", 0x3F80), "V-new must equal"),
            ("state-f32.bin", struct.pack("<f", float("nan")), "nonfinite capture"),
        ):
            self.single_chunk_fixture()
            with (self.root / name).open("r+b") as file:
                file.write(data)
            result = self.run_single_chunk()
            self.assertEqual(result.returncode, 3)
            self.assertIn(error, result.stderr)
        self.single_chunk_fixture()
        (self.root / "state-f32.bin").write_bytes(b"")
        self.assertEqual(self.run_single_chunk().returncode, 3)

    def test_known_state_sum_has_parent_and_non_acceptance_labels(self) -> None:
        self.fixture()
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual((record["source_tokens"], record["replay_tokens"]), (128, 64))
        self.assertFalse(record["inference_acceptance"])
        self.assertEqual(record["exponent"], "host_exp2f_not_sm121_sfu")
        for variant in record["variants"]:
            self.assertEqual(variant["reference_bf16"]["elements"], 32 * 128 * 128)
            self.assertEqual(variant["reference_bf16"]["mismatch_count"], 0)

    def test_nonzero_seed_and_bad_shapes_cannot_use_zero_seed_simplification(self) -> None:
        self.fixture()
        for tokens in ("0", "64", "8193", "128bad"):
            self.assertEqual(self.run_probe(tokens).returncode, 2)
        with (self.root / "full-initial_state-f32.bin").open("r+b") as file:
            file.write(struct.pack("<f", 1.0))
        result = self.run_probe()
        self.assertEqual(result.returncode, 3)
        self.assertIn("requires captured zero initial state", result.stderr)

    def test_short_reference_is_rejected(self) -> None:
        self.fixture()
        (self.root / "full-chunk-state-bf16.bin").write_bytes(b"")
        self.assertEqual(self.run_probe().returncode, 3)

    def test_sampled_trajectory_carries_state_across_both_chunks(self) -> None:
        self.fixture()
        result = self.run_probe(trajectory=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)["trajectory"]
        self.assertFalse(record["reference_state_injected"])
        self.assertEqual(record["sampled_state_rows"], 16)
        for variant in record["variants"]:
            for name in ("chunk_state_bf16", "v_new_bf16", "final_state_f32"):
                self.assertEqual(variant[name]["mismatch_count"], 0)
            self.assertEqual(variant["chunk_state_bf16"]["elements"], 16 * 2 * 128)
            self.assertEqual(variant["v_new_bf16"]["elements"], 16 * 128)
            self.assertEqual(variant["final_state_f32"]["elements"], 16 * 128)
            self.assertFalse(variant["same_input_projection"]["feeds_trajectory"])
            self.assertEqual(variant["same_input_projection"]["v_new_bf16"]["mismatch_count"], 0)
            if "first_state_boundaries" in variant:
                self.assertEqual(variant["first_state_boundaries"], [])

    def test_reference_checkpoint_changes_do_not_feed_the_carried_state(self) -> None:
        self.fixture()
        state = 32 * 128 * 128
        with (self.root / "full-chunk-state-bf16.bin").open("r+b") as file:
            file.seek(state * 2)
            file.write(struct.pack("<H", 0x4240) * state)
        result = self.run_probe(trajectory=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for variant in json.loads(result.stdout)["trajectory"]["variants"]:
            self.assertEqual(variant["chunk_state_bf16"]["mismatch_count"], 16 * 128)
            self.assertEqual(variant["final_state_f32"]["mismatch_count"], 0)
            if "first_state_boundaries" in variant:
                self.assertEqual(len(variant["first_state_boundaries"]), 16)
                for boundary in variant["first_state_boundaries"]:
                    self.assertEqual((boundary["chunk"], boundary["key"]), (1, 0))
                    self.assertEqual(boundary["actual_f32"], 32.0)
                    self.assertEqual(boundary["expected_bf16"], 48.0)
                    self.assertEqual(boundary["previous_f32"], 0.0)
                    self.assertEqual(boundary["previous_decay_f32"], 1.0)
                    self.assertEqual(boundary["previous_update_f32"], 32.0)
                    self.assertFalse(boundary["reference_raw_state_available"])

    def test_captured_exponent_control_is_explicit_and_has_unique_constraints(self) -> None:
        self.fixture(captured=True)
        result = self.run_probe(captured=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)["trajectory"]
        control = record["captured_exp_control"]
        self.assertTrue(control["reference_derived"])
        self.assertFalse(control["production_implementation"])
        self.assertEqual(control["resolved_inputs"], 1)
        self.assertEqual(control["inconsistent_inputs"], 0)
        self.assertEqual(control["resolved_host_differences"], 0)
        for variant in record["variants"][-3:]:
            self.assertGreater(variant["reference_derived_exp_mask"], 0)
            self.assertEqual(variant["captured_exp_calls"], variant["captured_exp_hits"])
            self.assertEqual(variant["final_state_f32"]["mismatch_count"], 0)

    def test_conflicting_exponent_constraints_are_not_chosen_to_fit_state(self) -> None:
        self.fixture(captured=True)
        with (self.root / "full-a-f32.bin").open("r+b") as file:
            file.seek(32 * 64 * 4)
            file.write(struct.pack("<f", 0.5))
        result = self.run_probe(captured=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)["trajectory"]
        self.assertEqual(record["captured_exp_control"]["inconsistent_inputs"], 1)
        self.assertEqual(record["captured_exp_control"]["resolved_inputs"], 0)
        for variant in record["variants"][-3:]:
            self.assertEqual(variant["captured_exp_hits"], 0)
            self.assertEqual(variant["final_state_f32"]["mismatch_count"], 0)

    def test_subnormal_products_cannot_claim_a_unique_exponent(self) -> None:
        self.fixture(captured=True)
        for name in ("a-dot", "a"):
            (self.root / ("full-" + name + "-f32.bin")).write_bytes(struct.pack("<I", 1) * (128 * 32 * 64))
        result = self.run_probe(captured=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)["trajectory"]
        control = record["captured_exp_control"]
        self.assertEqual(control["resolved_inputs"], 0)
        self.assertEqual(control["unobserved_inputs"], 1)
        self.assertGreater(control["unusable_products"], 0)
        for variant in record["variants"][-3:]:
            self.assertEqual(variant["captured_exp_hits"], 0)
            self.assertEqual(variant["final_state_f32"]["mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
