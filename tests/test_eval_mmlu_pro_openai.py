#!/usr/bin/env python3

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_mmlu_pro_openai as subject


def row(question_id, category="math", answer="B"):
    return {
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "options": ["zero", "one", "two"],
        "answer": answer,
        "answer_index": subject.CHOICES.index(answer),
        "cot_content": f"A: Reasoning. The answer is ({answer}).",
        "category": category,
        "src": "unit-test",
    }


class MmluProOpenAiTest(unittest.TestCase):
    def test_public_cli_defaults_to_local_candidate_and_supports_short_aliases(self):
        defaults = subject.parse_args([])
        self.assertEqual(defaults.side, "candidate")
        self.assertEqual(defaults.candidate_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(
            defaults.reference_url, "https://reference.example.invalid/v1"
        )
        aliased = subject.parse_args(
            [
                "--base-url",
                "http://localhost:9000/v1",
                "--model",
                "local-model",
                "--output",
                "candidate.jsonl",
            ]
        )
        self.assertEqual(aliased.candidate_url, "http://localhost:9000/v1")
        self.assertEqual(aliased.candidate_model, "local-model")
        self.assertEqual(aliased.results_file, Path("candidate.jsonl"))

    def test_extract_answer_uses_official_fallback_order(self):
        self.assertEqual(subject.extract_answer("The answer is (C)."), "C")
        self.assertEqual(subject.extract_answer("Answer: D"), "D")
        self.assertEqual(subject.extract_answer("First A, finally J"), "J")
        self.assertIsNone(subject.extract_answer("no letter choice"))

    def test_direct_prompt_has_shared_five_shot_prefix(self):
        validation = [row(index, answer="B") for index in range(5)]
        tests = [row(70), row(71)]
        tasks = subject.prepare_tasks(
            tests,
            validation,
            mode="direct",
            categories=None,
            question_ids=None,
            limit=0,
        )
        self.assertEqual(len(tasks), 2)
        self.assertEqual(
            tasks[0].shared_prefix_sha256, tasks[1].shared_prefix_sha256
        )
        self.assertTrue(tasks[0].prompt.endswith("Answer: The answer is ("))
        self.assertIn('Answer: The answer is (B).', tasks[0].prompt)
        self.assertEqual(
            subject.request_messages(tasks[0], "direct"),
            [
                {"role": "system", "content": subject.DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": tasks[0].prompt},
            ],
        )

    def test_official_cot_prompt_matches_harness_shape(self):
        validation = [row(index, answer="B") for index in range(5)]
        task = subject.prepare_tasks(
            [row(70)],
            validation,
            mode="official-cot",
            categories=None,
            question_ids=None,
            limit=0,
        )[0]
        self.assertIn("Think step by step", task.prompt)
        self.assertTrue(task.prompt.endswith("Answer: Let's think step by step.\n\n"))
        self.assertNotIn("A: Reasoning", task.prompt)
        self.assertEqual(
            subject.request_messages(task, "official-cot"),
            [{"role": "user", "content": task.prompt}],
        )

    def test_limit_per_category_is_stratified_and_keeps_sequence_dense(self):
        validation = [
            row(index, category=category, answer="B")
            for category in ("biology", "math")
            for index in range(5)
        ]
        tests = [
            row(base + index, category=category)
            for category, base in (("biology", 100), ("math", 200))
            for index in range(3)
        ]
        tasks = subject.prepare_tasks(
            tests,
            validation,
            mode="direct",
            categories=None,
            question_ids=None,
            limit=0,
            limit_per_category=2,
        )
        self.assertEqual(
            [task.category for task in tasks],
            ["biology", "biology", "math", "math"],
        )
        self.assertEqual([task.sequence for task in tasks], [0, 1, 2, 3])

    def test_question_ids_select_exact_rows_across_categories(self):
        validation = [
            row(index, category=category, answer="B")
            for category in ("biology", "math")
            for index in range(5)
        ]
        tests = [
            row(base + index, category=category)
            for category, base in (("biology", 100), ("math", 200))
            for index in range(3)
        ]
        tasks = subject.prepare_tasks(
            tests,
            validation,
            mode="direct",
            categories=None,
            question_ids={101, 202},
            limit=0,
        )
        self.assertEqual([task.question_id for task in tasks], [101, 202])
        self.assertEqual([task.sequence for task in tasks], [0, 1])

    def test_question_id_files_are_strict_and_hashed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "questions.txt"
            payload = b"101\n\n202\r\n"
            path.write_bytes(payload)
            question_ids, evidence = subject.load_question_id_files([path])
        self.assertEqual(question_ids, [101, 202])
        self.assertEqual(evidence[0]["question_ids"], 2)
        self.assertEqual(evidence[0]["bytes"], len(payload))
        self.assertEqual(evidence[0]["sha256"], hashlib.sha256(payload).hexdigest())

    def test_question_id_files_reject_duplicates_and_non_decimal_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            duplicate = root / "duplicate.txt"
            duplicate.write_text("101\n101\n", encoding="utf-8")
            with self.assertRaisesRegex(subject.EvalError, "duplicate"):
                subject.load_question_id_files([duplicate])
            invalid = root / "invalid.txt"
            invalid.write_text("101\n-1\n", encoding="utf-8")
            with self.assertRaisesRegex(subject.EvalError, "non-negative"):
                subject.load_question_id_files([invalid])

    def test_summary_requires_full_equal_score(self):
        validation = [row(index, answer="B") for index in range(5)]
        tasks = subject.prepare_tasks(
            [row(70), row(71)],
            validation,
            mode="direct",
            categories=None,
            question_ids=None,
            limit=0,
        )
        candidate = subject.Endpoint("candidate", "http://candidate/v1", "m", None)
        reference = subject.Endpoint("reference", "http://reference/v1", "m", None)
        config = {"config_id": "unit"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            results = Path(temporary_directory) / "results.jsonl"
            with results.open("w", encoding="utf-8") as handle:
                for endpoint in (candidate, reference):
                    for task in tasks:
                        handle.write(
                            json.dumps(
                                {
                                    "config_id": "unit",
                                    "endpoint": endpoint.public_identity,
                                    "question_id": task.question_id,
                                    "category": task.category,
                                    "answer": task.answer,
                                    "prediction": task.answer,
                                    "correct": True,
                                    "parsed": True,
                                    "ok": True,
                                }
                            )
                            + "\n"
                        )
            runtime_evidence = {"candidate": [{"sha256": "0" * 64}]}
            summary = subject.summarize(
                tasks,
                [results],
                [candidate, reference],
                config,
                runtime_evidence,
            )
        self.assertTrue(summary["parity"]["passed"])
        self.assertEqual(summary["parity"]["prediction_agreement"], 1.0)
        self.assertEqual(summary["runtime_evidence"], runtime_evidence)

    def test_runtime_evidence_is_hashed_and_attached(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "state.json"
            path.write_text(
                json.dumps({"host": "baiying", "repo_commit": "abc"}),
                encoding="utf-8",
            )
            evidence = subject.load_runtime_evidence([path])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["payload"]["host"], "baiying")
        self.assertEqual(len(evidence[0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
