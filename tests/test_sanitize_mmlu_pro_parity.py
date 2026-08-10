from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sanitize_mmlu_pro_parity import public_row, sanitize  # noqa: E402


def fixture(side: str) -> dict:
    return {
        "sequence": 0,
        "question_id": 42,
        "category": "physics",
        "source": "fixture",
        "answer": "B",
        "prompt_sha256": "a" * 64,
        "messages_sha256": "b" * 64,
        "prediction": "B" if side == "candidate" else "C",
        "parsed": True,
        "correct": side == "candidate",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        "elapsed_ms": 1.25,
        "qrt_metrics": {"ttft_ms": 1.0},
        "endpoint": {"base_url": "http://private.invalid"},
        "content": "private response",
    }


class SanitizeMmluParityTests(unittest.TestCase):
    def test_public_row_removes_content_endpoint_and_reference_timing(self) -> None:
        row = public_row(fixture("candidate"), fixture("reference"))
        encoded = json.dumps(row)
        self.assertNotIn("endpoint", encoded)
        self.assertNotIn("content", encoded)
        self.assertNotIn("elapsed_ms", row["reference"])
        self.assertFalse(row["agreement"]["prediction"])

    def test_mismatched_binding_is_rejected(self) -> None:
        reference = fixture("reference")
        reference["question_id"] = 43
        with self.assertRaisesRegex(ValueError, "question_id mismatch"):
            public_row(fixture("candidate"), reference)

    def test_sanitize_reports_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.jsonl"
            reference = root / "reference.jsonl"
            candidate.write_text(json.dumps(fixture("candidate")) + "\n", encoding="utf-8")
            reference.write_text(json.dumps(fixture("reference")) + "\n", encoding="utf-8")
            output = StringIO()
            summary = sanitize(candidate, reference, output, expected_rows=1)
        self.assertEqual(summary.rows, 1)
        self.assertEqual(summary.candidate_correct, 1)
        self.assertEqual(summary.reference_correct, 0)
        self.assertEqual(summary.prediction_agreement, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
