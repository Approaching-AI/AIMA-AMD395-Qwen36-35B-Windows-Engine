from __future__ import annotations

import hashlib
import json
import random
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "contracts" / "hprefill_q8192_gb10_oracle.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv1a64(payload: bytes) -> str:
    digest = 1_469_598_103_934_665_603
    for value in payload:
        digest ^= value
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


class GB10Q8192OracleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))

    def test_oracle_binds_exact_real_token_prompt(self) -> None:
        prompt = self.oracle["prompt"]
        self.assertEqual(prompt["token_count"], 8192)
        self.assertEqual(prompt["seed"], 395518)
        self.assertEqual(prompt["first_token_id"], 84411)
        generator = random.Random(prompt["seed"])
        token_ids = [prompt["first_token_id"]]
        token_ids.extend(
            32 + generator.randrange(256)
            for _ in range(prompt["token_count"] - 1)
        )
        packed = b"".join(struct.pack("<I", token) for token in token_ids)
        self.assertEqual(hashlib.sha256(packed).hexdigest(), prompt["u32le_sha256"])
        self.assertEqual(fnv1a64(packed), prompt["u32le_fnv1a64"])
        generator_path = ROOT / prompt["generator"]["path"]
        self.assertEqual(sha256_file(generator_path), prompt["generator"]["sha256"])

    def test_oracle_binds_live_gb10_token_and_raw_logit(self) -> None:
        oracle = self.oracle
        self.assertEqual(oracle["schema_version"], 1)
        self.assertEqual(oracle["record_type"], "qrt_hprefill_q8192_gb10_oracle")
        self.assertEqual(oracle["status"], "pass")
        self.assertEqual(oracle["correctness_authority"]["host"], "gb10-4t")
        self.assertEqual(
            oracle["correctness_authority"]["model"], "qwen3.6-35b-a3b"
        )
        expected = oracle["expected"]
        self.assertEqual(expected["first_token_id"], 144)
        self.assertEqual(expected["first_token_raw_logit"], 10.375)
        self.assertEqual(expected["first_token_raw_logit_tolerance"], 0.125)
        capture = oracle["capture"]
        self.assertEqual(capture["http_status"], 200)
        self.assertEqual(capture["usage"]["prompt_tokens"], 8192)
        self.assertEqual(capture["output_token_ids"], [expected["first_token_id"]])
        base = capture["base_lm_head_capture"]
        self.assertEqual(base["argmax_token_id"], expected["first_token_id"])
        self.assertEqual(base["selected_token_id"], expected["first_token_id"])
        self.assertEqual(
            base["selected_token_raw_logit_bf16"],
            expected["first_token_raw_logit"],
        )
        self.assertEqual(base["hidden_bf16_sha256"], "3a00370d" + "2d4a550d" + "ebba9ecd" + "403beb8a" + "f806851b" + "cb653063" + "59cd93f5" + "52644950")
        self.assertGreater(
            base["bf16_rounding"]["nearest_midpoint_distance"], 0.02
        )
        self.assertNotEqual(
            capture["mtp_capture_diagnostic"]["argmax_token_id"],
            expected["first_token_id"],
        )

    def test_oracle_binds_capture_command_and_model_weight(self) -> None:
        capture = self.oracle["capture"]
        command_file = ROOT / capture["command_file"]
        self.assertEqual(sha256_file(command_file), capture["command_file_sha256"])
        for name in ("request_sha256", "response_sha256", "command_file_sha256"):
            self.assertRegex(capture[name], r"^[0-9a-f]{64}$")
        model = self.oracle["model_evidence"]
        self.assertEqual(model["weight_tensor_key"], "lm_head.weight")
        self.assertEqual(model["weight_shape"], [248320, 2048])
        self.assertEqual(model["weight_dtype"], "torch.bfloat16")
        self.assertEqual(model["weight_shard_bytes"], 2231416848)
        self.assertRegex(model["weight_shard_sha256"], r"^[0-9a-f]{64}$")
        policy = self.oracle["diagnostic_policy"]
        self.assertFalse(policy["openai_logprob_is_raw_logit"])
        self.assertFalse(policy["engine_self_hashes_are_correctness_authority"])
        self.assertTrue(policy["continuation_captured_in_this_contract"])
        self.assertTrue(
            policy["decode_and_prefix_continuation_correctness_active"]
        )

    def test_oracle_binds_live_gb10_continuation_token_for_token(self) -> None:
        expected = self.oracle["expected"]
        capture = self.oracle["continuation_capture"]
        output_tokens = expected["output_token_ids"]
        self.assertEqual(expected["continuation_comparison"], "token_for_token")
        self.assertEqual(expected["continuation_token_count"], 32)
        self.assertEqual(expected["decode_step_count"], 31)
        self.assertEqual(len(output_tokens), expected["continuation_token_count"])
        self.assertEqual(output_tokens[0], expected["first_token_id"])
        self.assertEqual(capture["record_type"], "gb10_q8192_continuation_capture")
        self.assertEqual(capture["http_status"], 200)
        self.assertFalse(capture["capture_hook_armed"])
        self.assertEqual(capture["usage"]["prompt_tokens"], 8192)
        self.assertEqual(capture["usage"]["completion_tokens"], 32)
        self.assertEqual(capture["output_token_ids"], output_tokens)
        packed = b"".join(struct.pack("<I", token) for token in output_tokens)
        output_sha256 = hashlib.sha256(packed).hexdigest()
        output_fnv1a64 = fnv1a64(packed)
        self.assertEqual(output_sha256, expected["output_token_ids_u32le_sha256"])
        self.assertEqual(output_sha256, capture["output_token_ids_u32le_sha256"])
        self.assertEqual(output_fnv1a64, expected["output_token_ids_u32le_fnv1a64"])
        self.assertEqual(output_fnv1a64, capture["output_token_ids_u32le_fnv1a64"])
        repeat = capture["reproducibility"]
        self.assertEqual(repeat["request_count"], 2)
        self.assertTrue(repeat["token_for_token_repeat_pass"])
        self.assertEqual(repeat["repeat_request_sha256"], capture["request_sha256"])
        self.assertEqual(
            repeat["repeat_output_token_ids_u32le_sha256"], output_sha256
        )
        self.assertEqual(
            repeat["repeat_output_token_ids_u32le_fnv1a64"], output_fnv1a64
        )
        self.assertRegex(repeat["repeat_response_sha256"], r"^[0-9a-f]{64}$")
        capture_script = ROOT / capture["command_file"]
        self.assertEqual(
            sha256_file(capture_script), capture["command_file_sha256"]
        )
        for name in ("request_sha256", "response_sha256", "command_file_sha256"):
            self.assertRegex(capture[name], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
