"""Bind target logits to real input histories, including rejected draft rows."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from capture_gb10_runtime_boundaries import prepared_token_ids, qualify_transaction, target_rows  # noqa: E402


class RuntimeBoundaryTests(unittest.TestCase):
    def test_embedded_text_uses_the_prepared_real_token_buffer(self):
        self.assertEqual(prepared_token_ids(None, [82, 220], 2, [2, 2048]), [82, 220])
        self.assertEqual(prepared_token_ids([82, 220], [82, 220], 2, None), [82, 220])
        for arguments in ((None, [82, 220], 2, None), (None, [82, 220], 2, [1, 2048]),
                          ([82, 196], [82, 220], 2, None), (None, [82], 2, [2, 2048])):
            with self.assertRaises(ValueError):
                prepared_token_ids(*arguments)

    def test_partial_prefill_and_two_target_rows_keep_actual_logit_indices(self):
        rows = target_rows(list(range(8192)), [32] * 8192, [8191], {8190, 8191})
        self.assertEqual([r["logit_row"] for r in rows], [None, 0])
        rows = target_rows([8192, 8193], [144, 255], [0, 1], {8192, 8193})
        self.assertEqual([r["logit_row"] for r in rows], [0, 1])
        self.assertEqual([r["input_token_id"] for r in rows], [144, 255])
        for positions, ids, indices in (([0, 2], [32, 33], [1]), ([0], [32, 33], [0]),
                                        ([0], [32], [1]), ([0], [32], [0, 0])):
            with self.assertRaises(ValueError):
                target_rows(positions, ids, indices, {0})

    def test_draft_row_requires_every_input_through_that_row(self):
        history = [10, 20, 30, 40]
        for inputs, expected in (([30, 40], [True, True]), ([30, 99], [True, False]),
                                 ([99, 40], [False, False])):
            transaction = dict(first_position=2, input_token_ids=inputs,
                               rows=target_rows([2, 3], inputs, [0, 1], {2, 3}))
            actual = qualify_transaction(transaction, history)
            self.assertEqual([r["matches_generated_history"] for r in actual], expected)


if __name__ == "__main__":
    unittest.main()
