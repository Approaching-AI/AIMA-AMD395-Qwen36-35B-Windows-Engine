import hashlib
import json
import random
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(tokens):
    packed = b"".join(struct.pack("<I", token) for token in tokens)
    fnv = 1469598103934665603
    for byte in packed:
        fnv = ((fnv ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return hashlib.sha256(packed).hexdigest(), f"{fnv:016x}"


class Q7169OracleTests(unittest.TestCase):
    def setUp(self):
        self.oracle = json.loads((ROOT / "contracts/arbitrary_q7169_gb10_oracle.json").read_text())

    def test_exact_fixture_and_continuation(self):
        prompt = self.oracle["prompt"]
        self.assertEqual((prompt["token_count"], prompt["seed"], prompt["first_token_id"]), (7169, 50536152, 30199))
        rng = random.Random(prompt["seed"])
        tokens = [prompt["first_token_id"]] + [rng.randrange(256) + 32 for _ in range(7168)]
        self.assertEqual(fingerprint(tokens), (prompt["u32le_sha256"], prompt["u32le_fnv1a64"]))
        expected = self.oracle["expected"]
        self.assertEqual(expected["output_token_ids"][0], 82)
        self.assertEqual(len(expected["output_token_ids"]), 32)
        self.assertEqual(fingerprint(expected["output_token_ids"]), (expected["output_token_ids_u32le_sha256"], expected["output_token_ids_u32le_fnv1a64"]))

    def test_source_and_model_binding_does_not_claim_windows_acceptance(self):
        capture = self.oracle["capture"]
        for source, digest in [("fixture_adapter", "fixture_adapter_sha256"), ("raw_capture_source", "raw_capture_source_sha256"), ("continuation_capture_source", "continuation_capture_source_sha256")]:
            self.assertEqual(hashlib.sha256((ROOT / capture[source]).read_bytes()).hexdigest(), capture[digest])
        self.assertFalse(self.oracle["windows_acceptance"])
        self.assertFalse(capture["continuation_capture_hook_armed"])
        self.assertEqual(self.oracle["expected"]["first_token_raw_logit"], 9.25)
        self.assertEqual(self.oracle["expected"]["first_token_raw_logit_tolerance"], 0.125)
        q8192 = json.loads((ROOT / "contracts/hprefill_q8192_gb10_oracle.json").read_text())
        self.assertEqual(self.oracle["model_evidence"]["weight_shard_sha256"], q8192["model_evidence"]["weight_shard_sha256"])
