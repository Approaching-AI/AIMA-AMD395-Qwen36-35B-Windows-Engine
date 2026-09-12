"""Bind target logits to real input histories, including rejected draft rows."""
from pathlib import Path
import os
import json
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from capture_gb10_runtime_boundaries import (  # noqa: E402
    full_attention_observation_layer, full_cache_observation_offset,
    full_cache_observation_row, full_cache_row_is_qualified,
    full_prefill_attention_window, full_prefill_linear_window, matches_linear_window,
    observation_positions, prepared_token_ids, qualify_transaction,
    recurrent_state_selection, short_prefill_moe_observation, target_rows,
)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_suffix_attention_window_binds_original_extent_and_leaves_controls_unselected(self):
        case = 'long-prefix16384-suffix1024-out512'
        plan = dict(layer=3, first_position=16384, tokens=1024)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(full_prefill_attention_window(case, 17408))
        with patch.dict(os.environ, {'QRT_GB10_FULL_PREFILL_ATTENTION_WINDOWS':
                                    json.dumps({case: plan})}, clear=True):
            self.assertEqual(full_prefill_attention_window(case, 17408), plan)
            self.assertIsNone(full_prefill_attention_window('q8192-out32', 8192))
            self.assertTrue(matches_linear_window(plan, 3, dict(first_position=16384, token_count=1024)))
            self.assertFalse(matches_linear_window(plan, 3, dict(first_position=17408, token_count=2)))
            with self.assertRaises(ValueError):
                full_prefill_attention_window(case, 17407)
        for value in ([], {}, {case: dict(plan, layer=2)}, {case: dict(plan, layer=43)},
                      {case: dict(plan, first_position=0)}, {case: dict(plan, tokens=1025)},
                      {case: dict(plan, first_position=32700)}, {case: dict(plan, tokens=True)},
                      {case: dict(plan, extra=1)}, {'undeclared-extent': plan}):
            with patch.dict(os.environ, {'QRT_GB10_FULL_PREFILL_ATTENTION_WINDOWS':
                                        json.dumps(value)}, clear=True):
                with self.assertRaises(ValueError):
                    full_prefill_attention_window(case, 17408)

    def test_seeded_prefill_window_matches_original_batch_and_preserves_controls(self):
        case = 'long-prefix16384-suffix1024-out512'
        plan = dict(layer=0, first_position=16384, tokens=1024)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(full_prefill_linear_window(case, 17408))
        with patch.dict(os.environ, {'QRT_GB10_FULL_PREFILL_LINEAR_WINDOWS':
                                    json.dumps({case: plan})}, clear=True):
            window = full_prefill_linear_window(case, 17408)
            self.assertEqual(window, plan)
            self.assertIsNone(full_prefill_linear_window('q8192-out32', 8192))
            self.assertEqual(observation_positions('q8192-out32', 8192), {8191,8192})
            self.assertTrue(matches_linear_window(window, 0, dict(first_position=16384, token_count=1024)))
            for layer, start, count in ((2,16384,1024),(0,16383,1024),(0,16384,1023),(0,17408,2)):
                self.assertFalse(matches_linear_window(window, layer, dict(first_position=start, token_count=count)))
            with self.assertRaises(ValueError):
                full_prefill_linear_window(case, 17407)
        invalid = [[], {}, {case: []}, {case: dict(plan, extra=1)},
                   {case: dict(plan, layer=3)}, {case: dict(plan, layer=40)},
                   {case: dict(plan, tokens=1025)}, {case: dict(plan, tokens=0)},
                   {case: dict(plan, first_position=-1)}, {case: dict(plan, first_position=0)},
                   {case: dict(plan, first_position=263167)},
                   {case: dict(plan, layer=True)}, {'undeclared': plan}]
        for plans in invalid:
            with patch.dict(os.environ, {'QRT_GB10_FULL_PREFILL_LINEAR_WINDOWS': json.dumps(plans)},
                            clear=True):
                with self.assertRaises(ValueError):
                    full_prefill_linear_window(case, 17408)

    def test_short_moe_capture_requires_the_complete_original_prefill(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(short_prefill_moe_observation(5, 0, 5))
        with patch.dict(os.environ, {'QRT_GB10_SHORT_PREFILL_MOE': '1'}, clear=True):
            for length in (1, 5, 85, 128, 294, 384):
                self.assertTrue(short_prefill_moe_observation(length, 0, length))
            for args in ((0, 0, 0), (385, 0, 385), (7169, 0, 7169),
                         (5, 0, 4), (5, 1, 5), (5, 5, 2)):
                self.assertFalse(short_prefill_moe_observation(*args))

    def test_all_short_prefill_rows_preserve_actual_history_and_long_controls(self):
        with patch.dict(os.environ, {'QRT_GB10_SHORT_PREFILL_ALL_ROWS_MAX_TOKENS': '85'},
                        clear=True):
            self.assertEqual(observation_positions('http-plain-q5-out32', 5), set(range(6)))
            self.assertEqual(observation_positions('http-tool-continuation-q85-out32', 85),
                             set(range(86)))
            self.assertEqual(observation_positions('q7169-out32', 7169), {7168, 7169})
        with patch.dict(os.environ, {'QRT_GB10_SHORT_PREFILL_ALL_ROWS_MAX_TOKENS': '294'},
                        clear=True):
            self.assertEqual(observation_positions('http-tool-call-q294-out32', 294),
                             set(range(295)))
            self.assertEqual(observation_positions('q7169-out32', 7169), {7168, 7169})
        for value in ('0', '-1', '385', 'bad'):
            with patch.dict(os.environ, {'QRT_GB10_SHORT_PREFILL_ALL_ROWS_MAX_TOKENS': value},
                            clear=True):
                with self.assertRaises(ValueError):
                    observation_positions('http-plain-q5-out32', 5)

    def test_actual_case_offsets_keep_prefill_and_bind_declared_history(self):
        name = 'http-plain-q5-out32'
        with patch.dict(os.environ, {'QRT_GB10_CASE_BOUNDARY_OFFSETS':
                                     json.dumps({name: [6, 7]})}, clear=True):
            self.assertEqual(observation_positions(name, 5), {4, 5, 11, 12})
            self.assertEqual(observation_positions('q7169-out32', 7169), {7168, 7169})
            rows = target_rows([10, 11], [100, 200], [0, 1],
                               observation_positions(name, 5))
            self.assertEqual(rows, [dict(row=1, position=11, input_token_id=200, logit_row=1)])
            transaction = dict(first_position=10, input_token_ids=[100, 200], rows=rows)
            self.assertTrue(qualify_transaction(transaction, [0] * 10 + [100, 200])[0]
                            ['matches_generated_history'])
            self.assertFalse(qualify_transaction(transaction, [0] * 10 + [99, 200])[0]
                             ['matches_generated_history'])
        for plan in ([], {}, {name: []}, {name: [-1]}, {name: [31]},
                     {name: [True]}, {name: [1, 1]}, {name: [1, 2, 3, 4]},
                     {'no-declared-extent': [1]}, {'case-out513': [1]},
                     {'case-out1': [0]}, {name: '6'}):
            with patch.dict(os.environ, {'QRT_GB10_CASE_BOUNDARY_OFFSETS':
                                         json.dumps(plan)}, clear=True):
                with self.assertRaises(ValueError):
                    observation_positions(name, 5)

    def test_full_attention_capture_has_one_valid_owner(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(full_attention_observation_layer(), 3)
        for layer in (3, 7, 19, 39):
            with patch.dict(os.environ, {'QRT_GB10_BOUNDARY_FULL_LAYER': str(layer)}, clear=True):
                self.assertEqual(full_attention_observation_layer(), layer)
        for value in ('-1', '0', '18', '40', '3,19', 'bad'):
            with patch.dict(os.environ, {'QRT_GB10_BOUNDARY_FULL_LAYER': value}, clear=True):
                with self.assertRaises(ValueError):
                    full_attention_observation_layer()

    def test_bounded_positions_do_not_guess_the_accepted_mtp_row(self):
        with patch.dict(os.environ, {'QRT_GB10_Q7169_BOUNDARY_OFFSETS': '94'}, clear=True):
            positions = observation_positions('q7169-out512', 7169)
            self.assertEqual(positions, {7168, 7169, 7263})
            self.assertEqual(observation_positions('q7169-out32', 7169), {7168, 7169})
            # The target can occur as an accepted second row or as a first
            # row after a rejected draft. Selection retains either identity.
            self.assertEqual(target_rows([7262, 7263], [100, 200], [0, 1], positions),
                             [dict(row=1, position=7263, input_token_id=200, logit_row=1)])
            self.assertEqual(target_rows([7263, 7264], [200, 300], [0, 1], positions),
                             [dict(row=0, position=7263, input_token_id=200, logit_row=0)])
        for value in ('-1', '512', '1,1', '1,2,3,4', '', 'bad'):
            with patch.dict(os.environ, {'QRT_GB10_Q7169_BOUNDARY_OFFSETS': value}, clear=True):
                with self.assertRaises(ValueError):
                    observation_positions('q7169-out512', 7169)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(observation_positions('q7169-out512', 7169),
                             {7168, 7169, 7199, 7200, 7201, 7287, 7288, 7289})

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
