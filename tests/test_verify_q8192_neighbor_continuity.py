import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / (
    "scripts/verify_q8192_neighbor_continuity.py"
)
SPEC = importlib.util.spec_from_file_location("neighbor_continuity", SCRIPT)
NEIGHBOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(NEIGHBOR)


class NeighborContinuityTests(unittest.TestCase):
    def test_command_redacts_api_keys(self) -> None:
        self.assertEqual(
            NEIGHBOR.redacted_command(
                ["verify.py", "--api-key", "secret", "--api-key=also-secret"]
            ),
            [
                "verify.py",
                "--api-key",
                "<redacted>",
                "--api-key=<redacted>",
            ],
        )

    def test_cases_are_stable_cold_and_cover_both_generation_shapes(self) -> None:
        first = NEIGHBOR.build_cases(3, 81928193)
        second = NEIGHBOR.build_cases(3, 81928193)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 18)
        self.assertEqual(len({case["first_token_id"] for case in first}), 18)
        self.assertEqual(
            {(case["prompt_tokens"], case["max_tokens"]) for case in first},
            {
                (length, max_tokens)
                for length in NEIGHBOR.PROMPT_LENGTHS
                for max_tokens in NEIGHBOR.MAX_TOKEN_COUNTS
            },
        )
        for case in first:
            prompt = NEIGHBOR.build_prompt(case)
            self.assertEqual(len(prompt), case["prompt_tokens"])
            self.assertEqual(prompt[0], case["first_token_id"])

    def test_response_record_uses_explicit_token_ids_and_stays_compact(self) -> None:
        case = {
            "ordinal": 0,
            "prompt_tokens": 2,
            "max_tokens": 2,
            "repetition": 0,
            "prompt_seed": 1,
            "first_token_id": 1024,
        }
        value = {
            "choices": [
                {
                    "text": "transport-text" * 1000,
                    "token_ids": [42, 43],
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }
        body = json.dumps(value).encode()
        record = NEIGHBOR.response_record(case, [1024, 32], body, value, 1.0)
        self.assertEqual(record["token_ids"], [42, 43])
        self.assertNotIn("text", record)
        self.assertEqual(record["text_char_count"], 14_000)

    def test_response_record_rejects_missing_token_ids(self) -> None:
        case = {
            "ordinal": 0,
            "prompt_tokens": 1,
            "max_tokens": 1,
            "repetition": 0,
            "prompt_seed": 1,
            "first_token_id": 1024,
        }
        value = {
            "choices": [{"text": "x", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        with self.assertRaises(NEIGHBOR.VerificationError):
            NEIGHBOR.response_record(case, [1024], b"{}", value, 1.0)

    def records(self, neighbor_ms: float) -> list[dict]:
        records = []
        for max_tokens in NEIGHBOR.MAX_TOKEN_COUNTS:
            for prompt_tokens in NEIGHBOR.PROMPT_LENGTHS:
                value = 4000.0 if prompt_tokens == 8192 else neighbor_ms
                for _ in range(3):
                    records.append(
                        {
                            "max_tokens": max_tokens,
                            "prompt_tokens": prompt_tokens,
                            "ttft_ms": value,
                        }
                    )
        return records

    def test_continuity_gate_accepts_smooth_neighbors(self) -> None:
        summary = NEIGHBOR.continuity_summary(self.records(4500.0))
        self.assertTrue(summary["all_gates_passed"])
        self.assertTrue(all(gate["passed"] for gate in summary["gates"]))

    def test_continuity_gate_rejects_the_reported_cliff(self) -> None:
        summary = NEIGHBOR.continuity_summary(self.records(61_000.0))
        self.assertFalse(summary["all_gates_passed"])
        self.assertTrue(all(not gate["passed"] for gate in summary["gates"]))


if __name__ == "__main__":
    unittest.main()
