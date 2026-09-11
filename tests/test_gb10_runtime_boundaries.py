"""Bind target logits to real input histories, including rejected draft rows."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from capture_gb10_runtime_boundaries import (  # noqa: E402
    full_cache_observation_offset, full_cache_observation_row, full_cache_row_is_qualified,
    prepared_token_ids, qualify_transaction,
    recurrent_state_selection, target_rows,
)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_cache_can_bind_an_accepted_second_target_row(self):
        with patch.dict(os.environ, {'QRT_GB10_Q7169_FULL_CACHE_OFFSET': '37',
                                    'QRT_GB10_Q7169_FULL_CACHE_ROW': '1'}, clear=True):
            self.assertEqual(full_cache_observation_row('q7169-out512'), 1)
            self.assertEqual(full_cache_observation_row('q7169-out32'), 0)
        for offset, row in (('0', '1'), ('37', '2'), ('37', '-1')):
            with patch.dict(os.environ, {'QRT_GB10_Q7169_FULL_CACHE_OFFSET': offset,
                                        'QRT_GB10_Q7169_FULL_CACHE_ROW': row}, clear=True):
                with self.assertRaises(ValueError):
                    full_cache_observation_row('q7169-out512')
        transaction = dict(ordinal=1, first_position=2, input_token_ids=[30, 40],
                           rows=target_rows([2, 3], [30, 40], [0, 1], {3}))
        cache = dict(transaction=1, row=1, tokens=4, input_token_id=40)
        for history, expected in (([10, 20, 30, 40], True), ([10, 20, 30, 99], False),
                                  ([10, 20, 99, 40], False)):
            transaction['qualified_rows'] = qualify_transaction(transaction, history)
            self.assertEqual(full_cache_row_is_qualified(cache, [transaction]), expected)
        transaction['qualified_rows'] = qualify_transaction(transaction, [10, 20, 30, 40])
        for field, value in (('transaction', 2), ('row', 0), ('tokens', 5), ('input_token_id', 99)):
            self.assertFalse(full_cache_row_is_qualified(dict(cache, **{field: value}), [transaction]))

    def test_cache_position_is_scoped_to_its_frozen_case(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(full_cache_observation_offset('q8191-out32'), 0)
        with patch.dict(os.environ, {
            'QRT_GB10_Q8191_FULL_CACHE_OFFSET': '3',
            'QRT_GB10_Q7169_FULL_CACHE_OFFSET': '37',
            'QRT_GB10_Q8192_FULL_CACHE_OFFSET': '119',
        }, clear=True):
            self.assertEqual(full_cache_observation_offset('q8191-out32'), 3)
            self.assertEqual(full_cache_observation_offset('q7169-out512'), 37)
            self.assertEqual(full_cache_observation_offset('q8192-out512'), 119)
            self.assertEqual(full_cache_observation_offset('q7169-out32'), 0)
            self.assertEqual(full_cache_observation_offset('q8192-out32'), 0)
        for case, name, values in (
            ('q8191-out32', 'QRT_GB10_Q8191_FULL_CACHE_OFFSET', ['-1', '32', 'bad']),
            ('q7169-out512', 'QRT_GB10_Q7169_FULL_CACHE_OFFSET', ['-1', '512']),
        ):
            for value in values:
                with patch.dict(os.environ, {name: value}, clear=True):
                    with self.assertRaises(ValueError):
                        full_cache_observation_offset(case)

    def test_recurrent_input_uses_the_previous_accepted_slot(self):
        for accepted, initial in (([1], 7), ([2], 11)):
            selection = recurrent_state_selection([[7, 11]], accepted, 2, 20)
            self.assertEqual(selection["initial_slot"], initial)
            self.assertEqual(selection["final_slots"], [7, 11])
        self.assertEqual(recurrent_state_selection([7], None, 1, 20)["initial_slot"], 7)
        for arguments in (([[7, 11]], [0], 2, 20), ([[7, 11]], [3], 2, 20),
                          ([[7, -1]], [1], 2, 20), ([[7, 20]], [1], 2, 20),
                          ([[7, 11], [9, 10]], [1], 2, 20), ([7, 11], None, 2, 20)):
            with self.assertRaises(ValueError):
                recurrent_state_selection(*arguments)

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
