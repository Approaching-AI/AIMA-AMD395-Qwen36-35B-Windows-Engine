"""Bind target logits to real input histories, including rejected draft rows."""
from pathlib import Path
import os
import json
import sys
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from capture_gb10_runtime_boundaries import (  # noqa: E402
    full_attention_observation_layer, full_attention_observation_layers, full_cache_observation_offset,
    full_cache_observation_row, full_cache_row_is_qualified,
    full_prefill_attention_window, full_prefill_linear_window, matches_linear_window,
    observation_byte_limit, linear_observation_layers,
    full_prefill_linear_core_only, full_prefill_linear_labels, product_prefill_operands,
    observation_positions, observation_timeout_seconds, prepared_token_ids, qualify_transaction,
    recurrent_observation_window, recurrent_state_selection, selected_prefill_moe_observation,
    short_prefill_moe_observation, target_rows,
    observe_original_moe_routed, prefill_moe_routed_rows_enabled, observe_original_attention_owner,
)
from capture_gb10_token_matrix import qualify_runtime_capture  # noqa: E402


class RuntimeBoundaryTests(unittest.TestCase):
    def test_recurrent_window_keeps_original_history_and_other_cases(self):
        case = 'long-prefix262144-suffix1024-out512'
        plan = dict(layer=5, first_offset=0, tokens=124)
        env = {'QRT_GB10_CASE_RECURRENT_WINDOWS': json.dumps({case: plan}),
               'QRT_GB10_CASE_BOUNDARY_LINEAR_LAYERS': json.dumps({case: [5]})}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(recurrent_observation_window(case, 263168), plan)
            self.assertEqual(observation_positions(case, 263168), {263167} | set(range(263168, 263292)))
            self.assertEqual(observation_byte_limit(None, case, 263168), 1024 << 20)
            self.assertEqual(linear_observation_layers(case), [5])
            self.assertIsNone(recurrent_observation_window('q7169-out32', 7169))
            self.assertEqual(observation_positions('q7169-out32', 7169), {7168, 7169})
            self.assertEqual(observation_byte_limit(None, 'q7169-out32', 7169), 512 << 20)
            for invalid in (0, 263169, True, 263168.0):
                with self.assertRaises(ValueError):
                    recurrent_observation_window(case, invalid)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(recurrent_observation_window(case, 263168))
            self.assertEqual(observation_positions(case, 263168), {263167, 263168})
            self.assertEqual(observation_byte_limit(None, case, 263168), 512 << 20)

    def test_recurrent_window_rejects_unbounded_or_nonoriginal_rows(self):
        case = 'long-prefix262144-suffix1024-out512'
        good = dict(layer=5, first_offset=0, tokens=124)
        bad = [[], {}, {case: []}, {case: dict(good, extra=1)}, {'not-a-case': good}]
        for key, values in [('layer', [-1, 3, 40, True, 5.0]),
                            ('first_offset', [-1, 500, True, 0.0]),
                            ('tokens', [0, 129, True, 124.0])]:
            bad.extend({case: dict(good, **{key: value})} for value in values)
        for plan in bad:
            with self.subTest(plan=plan), patch.dict(os.environ,
                    {'QRT_GB10_CASE_RECURRENT_WINDOWS': json.dumps(plan)}, clear=True):
                with self.assertRaises(ValueError):
                    observation_positions(case, 263168)
        with patch.dict(os.environ, {'QRT_GB10_CASE_RECURRENT_WINDOWS':
                json.dumps({'q8192-out32': dict(layer=5, first_offset=30, tokens=1)})}, clear=True):
            self.assertEqual(observation_positions('q8192-out32', 8192), {8191, 8192, 8222})

    def test_recurrent_window_requires_one_layer_and_excludes_bulk_captures(self):
        case = 'long-prefix262144-suffix1024-out512'
        env = {'QRT_GB10_CASE_RECURRENT_WINDOWS': json.dumps({case: dict(layer=5, first_offset=0, tokens=124)})}
        for layers in (None, [0], [0, 5]):
            current = dict(env)
            if layers is not None:
                current['QRT_GB10_CASE_BOUNDARY_LINEAR_LAYERS'] = json.dumps({case: layers})
            with self.subTest(layers=layers), patch.dict(os.environ, current, clear=True):
                with self.assertRaises(ValueError):
                    observation_byte_limit(None, case, 263168)
        env['QRT_GB10_CASE_BOUNDARY_LINEAR_LAYERS'] = json.dumps({case: [5]})
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                observation_byte_limit(dict(layer=3, first_position=0, tokens=8192), case, 263168)
        env['QRT_GB10_CASE_FULL_CACHE'] = json.dumps({case: dict(offset=0, row=0)})
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                observation_byte_limit(None, case, 263168)

    def test_product_prefill_operands_bind_the_original_complete_q8192_transaction(self):
        key = 'QRT_GB10_FULL_PREFILL_PRODUCT_OPERANDS'
        case = 'q8192-out512'
        with patch.dict(os.environ, {key: '1'}, clear=True):
            self.assertTrue(product_prefill_operands(case, 8192))
            linear = full_prefill_linear_window(case, 8192)
            attention = full_prefill_attention_window(case, 8192)
            self.assertEqual(linear, dict(layer=0, first_position=0, tokens=8192))
            self.assertEqual(attention, dict(layer=3, first_position=0, tokens=8192))
            self.assertEqual(full_prefill_linear_labels(0, False, True),
                             {'layer-00-input-rmsnorm', 'linear-00-qkv'})
            self.assertEqual(observation_byte_limit(attention), 1024 << 20)
            for other, count in [('q7169-out32', 7169), ('q8192-out32', 8192),
                                 ('q7169-out512', 7169), ('q8193-out32', 8193)]:
                self.assertFalse(product_prefill_operands(other, count))
                self.assertIsNone(full_prefill_linear_window(other, count))
                self.assertIsNone(full_prefill_attention_window(other, count))
            for count in (8191, 8193, True, 8192.0):
                with self.assertRaisesRegex(ValueError, 'original q8192 extent'):
                    product_prefill_operands(case, count)
        for name, value in [(key, '2'), ('QRT_GB10_FULL_PREFILL_LINEAR_WINDOWS', '{}'),
                ('QRT_GB10_FULL_PREFILL_ATTENTION_WINDOWS', '{}'),
                ('QRT_GB10_FULL_PREFILL_LINEAR_CORE_ONLY', '1')]:
            with patch.dict(os.environ, {key: '1', name: value}, clear=True):
                with self.assertRaises(ValueError):
                    full_prefill_linear_window(case, 8192)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(product_prefill_operands(case, 8192))
            self.assertIsNone(full_prefill_linear_window(case, 8192))
            self.assertIsNone(full_prefill_attention_window(case, 8192))
            with self.assertRaises(ValueError):
                full_prefill_linear_labels(0, True, True)

    def test_all_full_attention_owners_remain_scoped_to_the_named_case(self):
        layers = list(range(3, 40, 4))
        with patch.dict(os.environ, {'QRT_GB10_CASE_BOUNDARY_FULL_LAYERS':
                json.dumps({'q8192-out32': layers})}, clear=True):
            self.assertEqual(full_attention_observation_layers('q8192-out32'), layers)
            self.assertEqual(full_attention_observation_layers('q7169-out32'), [3])
        for plan in ([], {}, {'q8192-out32': []}, {'q8192-out32': [3, 3]},
                {'q8192-out32': [0]}, {'q8192-out32': [43]}, {'q8192-out32': [True]},
                {'q8192-out32': ['3']}, {'q8192-out32': [3.0]}, {'q8192-out32': [[]]},
                {'q8192': [3]}, {'q8192-out513': [3]}):
            with self.subTest(plan=plan), patch.dict(os.environ,
                    {'QRT_GB10_CASE_BOUNDARY_FULL_LAYERS': json.dumps(plan)}, clear=True):
                with self.assertRaises(ValueError):
                    full_attention_observation_layers('q8192-out32')

    def test_shared_rotary_owner_scope_preserves_results_and_unwinds_errors(self):
        active, seen = [], []
        result = object()
        def shared_rotary():
            seen.append([layer for layer in range(3, 40, 4) if active and active[-1] == layer])
        def inner(argument, *, marker):
            self.assertEqual((argument, marker), (13, 29))
            shared_rotary()
            return result
        def outer():
            shared_rotary()
            self.assertIs(observe_original_attention_owner(active, 39, inner, 13, marker=29), result)
            shared_rotary()
            return result
        self.assertIs(observe_original_attention_owner(active, 3, outer), result)
        self.assertEqual((active, seen), ([], [[3], [39], [3]]))
        error = ValueError('original forward error')
        def fails():
            shared_rotary()
            raise error
        with self.assertRaises(ValueError) as caught:
            observe_original_attention_owner(active, 7, fails)
        self.assertIs(caught.exception, error)
        self.assertEqual(active, [])
        self.assertEqual(seen[-1], [7])

    def test_case_cache_plan_binds_a_real_target_row_within_its_output_extent(self):
        key = 'QRT_GB10_CASE_FULL_CACHE'
        with patch.dict(os.environ, {key: json.dumps({'q8192-out32': dict(offset=1, row=1)})}, clear=True):
            self.assertEqual(full_cache_observation_offset('q8192-out32'), 1)
            self.assertEqual(full_cache_observation_row('q8192-out32'), 1)
            self.assertEqual(full_cache_observation_offset('q7169-out32'), 0)
            self.assertEqual(full_cache_observation_row('q7169-out32'), 0)
            self.assertEqual(observation_positions('q8192-out32', 8192), {8191, 8192, 8193})
        for plan in ([], {}, {'q8192': dict(offset=0, row=0)},
                {'q8192-out32': dict(offset=31, row=0)}, {'q8192-out32': dict(offset=-1, row=0)},
                {'q8192-out32': dict(offset=0, row=1)}, {'q8192-out32': dict(offset=1, row=2)},
                {'q8192-out32': dict(offset=True, row=0)}, {'q8192-out32': dict(offset=0, row=False)},
                {'q8192-out32': dict(offset=0, row=0.0)}, {'q8192-out32': dict(offset=0)},
                {'q8192-out32': dict(offset=0, row=0, unknown=1)}, {'q8192-out32': []}):
            with self.subTest(plan=plan), patch.dict(os.environ, {key: json.dumps(plan)}, clear=True):
                with self.assertRaises(ValueError):
                    full_cache_observation_offset('q8192-out32')

    def test_every_cache_owner_is_qualified_against_actual_generated_history(self):
        transactions = [dict(ordinal=1, first_position=2, input_token_ids=[30, 40],
            rows=target_rows([2, 3], [30, 40], [0, 1], {2, 3}))]
        layers = list(range(3, 40, 4))
        caches = [dict(layer=layer, transaction=1, row=1, tokens=4, input_token_id=40) for layer in layers]
        boundary = dict(transactions=transactions, selected_positions=[2, 3],
            full_attention_cache=caches[0], full_attention_caches=caches,
            full_attention_layers=layers, full_attention_cache_required=True)
        def qualify(value):
            qualify_runtime_capture(dict(runtime_boundaries=value), [10, 20], [30, 40, 50])
        qualify(deepcopy(boundary))
        # A later owner with a stale/rejected input cannot borrow the first
        # owner's qualification, even when every selected position is covered.
        for field, value in (('transaction', 2), ('row', 0), ('tokens', 5), ('input_token_id', 99)):
            changed = deepcopy(boundary)
            changed['full_attention_caches'][-1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'owner caches'):
                qualify(changed)
        for kind in ('missing', 'duplicate', 'reordered', 'primary', 'unrequested'):
            changed = deepcopy(boundary)
            if kind == 'missing': changed['full_attention_caches'].pop()
            if kind == 'duplicate': changed['full_attention_caches'][-1]['layer'] = 3
            if kind == 'reordered': changed['full_attention_caches'].reverse()
            if kind == 'primary': changed['full_attention_cache'] = dict(caches[-1])
            if kind == 'unrequested': changed['full_attention_cache_required'] = False
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'owner caches'):
                qualify(changed)
        empty = dict(boundary, full_attention_cache=None, full_attention_caches=[],
                     full_attention_cache_required=False)
        qualify(deepcopy(empty))
        legacy = {k: v for k, v in boundary.items() if k not in
                  {'full_attention_caches', 'full_attention_layers', 'full_attention_cache_required'}}
        qualify(deepcopy(legacy))

    def test_all_linear_layers_are_scoped_to_the_named_original_case(self):
        case = 'long-prefix262144-suffix1024-out512'
        layers = [layer for layer in range(40) if layer % 4 != 3]
        with patch.dict(os.environ, {'QRT_GB10_CASE_BOUNDARY_LINEAR_LAYERS':
                json.dumps({case: layers})}, clear=True):
            self.assertEqual(linear_observation_layers(case), layers)
            self.assertEqual(linear_observation_layers('q8192-out32'), [0, 2])
            self.assertEqual(linear_observation_layers('q8191-out32'), [0, 2, 4])
            self.assertEqual(observation_byte_limit(None), 512 << 20)

    def test_case_linear_layer_plan_rejects_invalid_extents_and_types(self):
        case = 'long-prefix262144-suffix1024-out512'
        for plan in ([], {}, {case: []}, {case: [0, 0]}, {case: [3]},
                {case: [40]}, {case: [-1]}, {case: [True]}, {case: ['0']},
                {case: [[]]}, {case: [0.0]},
                {'missing-output-extent': [0]}, {'q5-out513': [0]}):
            with self.subTest(plan=plan), patch.dict(os.environ,
                    {'QRT_GB10_CASE_BOUNDARY_LINEAR_LAYERS': json.dumps(plan)}, clear=True):
                with self.assertRaises(ValueError):
                    linear_observation_layers(case)

    def test_routed_moe_option_requires_bounded_selected_rows(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(prefill_moe_routed_rows_enabled())
        with patch.dict(os.environ, {'QRT_GB10_PREFILL_MOE_ROUTED_ROWS': '1'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'requires selected prefill rows'):
                prefill_moe_routed_rows_enabled()
        with patch.dict(os.environ, {'QRT_GB10_PREFILL_MOE_ROUTED_ROWS': '1',
                'QRT_GB10_PREFILL_MOE_SELECTED_ROWS': '1'}, clear=True):
            self.assertTrue(prefill_moe_routed_rows_enabled())
        with patch.dict(os.environ, {'QRT_GB10_PREFILL_MOE_ROUTED_ROWS': 'yes'}, clear=True):
            with self.assertRaises(ValueError):
                prefill_moe_routed_rows_enabled()

    def test_routed_moe_observer_preserves_results_arguments_and_restores_on_errors(self):
        original_calls, observed = [], []
        projection_result, forward_result = object(), object()
        def dispatch(*args, **kwargs):
            original_calls.append((args, kwargs))
            return projection_result
        module = SimpleNamespace(invoke_fused_moe_triton_kernel=dispatch)
        tensors = [SimpleNamespace(shape=shape) for shape in
                   ((8192, 8, 1024), (8192 * 8, 512), (8192, 8, 2048))]
        def forward(argument, *, marker):
            self.assertEqual((argument, marker), ('original-input', 13))
            self.assertIs(module.invoke_fused_moe_triton_kernel('a', 'b', tensors[0], mode=4), projection_result)
            self.assertIs(module.invoke_fused_moe_triton_kernel(tensors[1], 'd', tensors[2], mode=5), projection_result)
            return forward_result
        def observe(*args):
            observed.append(args)
        result = observe_original_moe_routed(forward, module, observe, 8192, 'original-input', marker=13)
        self.assertIs(result, forward_result)
        self.assertEqual([x[0] for x in observed], ['routed-gate-up', 'routed-activated', 'routed-weighted'])
        self.assertEqual([x[2] for x in observed], [8192, 4096, 16384])
        self.assertEqual(original_calls[0], (('a', 'b', tensors[0]), {'mode': 4}))
        self.assertIs(module.invoke_fused_moe_triton_kernel, dispatch)
        def fail(*args):
            raise RuntimeError('original forward failed')
        with self.assertRaisesRegex(RuntimeError, 'original forward failed'):
            observe_original_moe_routed(fail, module, observe, 8192)
        self.assertIs(module.invoke_fused_moe_triton_kernel, dispatch)
        tensors[2].shape = (8191, 8, 2048)
        with self.assertRaisesRegex(ValueError, 'projection shape changed'):
            observe_original_moe_routed(forward, module, observe, 8192, 'original-input', marker=13)
        self.assertIs(module.invoke_fused_moe_triton_kernel, dispatch)

    def test_complete_large_request_has_a_bounded_observation_deadline(self):
        for count in (1, 7169, 8192, 32768, 65536, 66560, 131072, 132096):
            self.assertEqual(observation_timeout_seconds(count), 180)
        for count in (132097, 262144, 263168):
            self.assertEqual(observation_timeout_seconds(count), 900)
        for count in (0, -1, 263169, True, 262144.0, "262144"):
            with self.assertRaisesRegex(ValueError, "original observation prompt extent"):
                observation_timeout_seconds(count)

    def test_interior_prefill_positions_bind_real_unsampled_rows_and_preserve_controls(self):
        case = 'long-prefix32768-owner-out32'
        setting = 'QRT_GB10_CASE_PREFILL_POSITIONS'
        with patch.dict(os.environ, {setting: json.dumps({case: [18553, 18554, 18555]})}, clear=True):
            selected = observation_positions(case, 32768)
            self.assertEqual(selected, {18553, 18554, 18555, 32767, 32768})
            rows = target_rows(list(range(16384, 24576)), [16602]*8192, [], selected)
            self.assertEqual([row['row'] for row in rows], [2169, 2170, 2171])
            self.assertTrue(all(row['logit_row'] is None for row in rows))
            self.assertEqual(observation_positions('q7169-out32', 7169), {7168, 7169})
            self.assertEqual(observation_positions('q8192-out32', 8192), {8191, 8192})
            with self.assertRaises(ValueError):
                observation_positions(case, 18555)
        for plan in ([], {}, {case: []}, {case: list(range(97))}, {case: [1, 1]},
                     {case: [-1]}, {case: [263168]}, {case: [True]}, {case: '18554'},
                     {'undeclared': [1]}, {'case-out513': [1]}):
            with patch.dict(os.environ, {setting: json.dumps(plan)}, clear=True):
                with self.assertRaises(ValueError):
                    observation_positions(case, 32768)

        positions = list(range(64)) + [1023, 2111, 3391, 4159, 6079, 7167, 8191]
        with patch.dict(os.environ, {setting: json.dumps({'q8192-out512': positions})}, clear=True):
            selected = observation_positions('q8192-out512', 8192)
            rows = target_rows(list(range(8192)), list(range(8192)), [8191], selected)
            self.assertEqual([row['position'] for row in rows], positions)
            self.assertEqual([row['input_token_id'] for row in rows], positions)
            self.assertEqual(observation_positions('q7169-out32', 7169), {7168, 7169})

    def test_long_chunk_end_observations_preserve_real_row_identity(self):
        case = 'long-prefix262144-owner-out512'
        ends = list(range(8191, 262144, 8192))
        environment = {'QRT_GB10_CASE_PREFILL_POSITIONS': json.dumps({case: ends})}
        with patch.dict(os.environ, environment, clear=True):
            selected = observation_positions(case, 262144)
            self.assertEqual(selected, set(ends) | {262144})
            for start in range(0, 262144, 8192):
                ids = list(range(start, start + 8192))
                rows = target_rows(ids, ids, [], selected)
                self.assertEqual(rows, [dict(row=8191, position=start + 8191,
                    input_token_id=start + 8191, logit_row=None)])
            self.assertEqual(observation_positions('q8192-out32', 8192), {8191, 8192})
            self.assertEqual(observation_positions('q7169-out32', 7169), {7168, 7169})
            with self.assertRaisesRegex(ValueError, 'beyond the original prompt'):
                observation_positions(case, 262143)
            self.assertEqual(observation_timeout_seconds(262144), 900)

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
                      {case: dict(plan, first_position=0)}, {case: dict(plan, tokens=8193)},
                      {case: dict(plan, first_position=32700)}, {case: dict(plan, tokens=True)},
                      {case: dict(plan, extra=1)}, {'undeclared-extent': plan}):
            with patch.dict(os.environ, {'QRT_GB10_FULL_PREFILL_ATTENTION_WINDOWS':
                                        json.dumps(value)}, clear=True):
                with self.assertRaises(ValueError):
                    full_prefill_attention_window(case, 17408)

    def test_long_decode_cache_budget_is_derived_and_bounded(self):
        case = 'long-prefix262144-suffix1024-out512'
        plan = {case: dict(offset=0, row=0)}
        with patch.dict(os.environ, {'QRT_GB10_CASE_FULL_CACHE': json.dumps(plan)}, clear=True):
            self.assertEqual(observation_byte_limit(None, case, 263168), 1024 << 20)
            self.assertEqual(observation_byte_limit(None, 'q8192-out32', 8192), 512 << 20)
            for value in (0, True, 263169):
                with self.assertRaises(ValueError):
                    observation_byte_limit(None, case, value)
            with patch.dict(os.environ, {'QRT_GB10_CASE_BOUNDARY_FULL_LAYERS': json.dumps({case: [3, 7]})}):
                with self.assertRaisesRegex(ValueError, '1 GiB'):
                    observation_byte_limit(None, case, 263168)

    def test_full_attention_cold_chunk_keeps_complete_cache_bounded(self):
        case = 'long-prefix131072-owner-out512'
        plan = dict(layer=15, first_position=90112, tokens=8192)
        with patch.dict(os.environ, {'QRT_GB10_FULL_PREFILL_ATTENTION_WINDOWS':
                                    json.dumps({case: plan})}, clear=True):
            self.assertEqual(full_prefill_attention_window(case, 131072), plan)
            self.assertEqual(observation_byte_limit(plan), 1024 << 20)
            control = full_prefill_attention_window('q8192-out32', 8192)
            self.assertIsNone(control)
            self.assertEqual(observation_byte_limit(control), 512 << 20)
            self.assertTrue(matches_linear_window(plan, 15, dict(first_position=90112, token_count=8192)))
            self.assertFalse(matches_linear_window(plan, 15, dict(first_position=81920, token_count=8192)))
        for change in (dict(first_position=131073), dict(tokens=8193), dict(tokens=True)):
            with patch.dict(os.environ, {'QRT_GB10_FULL_PREFILL_ATTENTION_WINDOWS':
                                        json.dumps({case: dict(plan, **change)})}, clear=True):
                with self.assertRaises(ValueError):
                    full_prefill_attention_window(case, 139264)

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

    def test_original_8192_core_window_keeps_seed_and_output_without_large_projections(self):
        case = 'long-prefix32768-owner-out32'
        plan = dict(layer=12, first_position=24576, tokens=8192)
        environment = {'QRT_GB10_FULL_PREFILL_LINEAR_WINDOWS': json.dumps({case: plan}),
                       'QRT_GB10_FULL_PREFILL_LINEAR_CORE_ONLY': '1'}
        with patch.dict(os.environ, environment, clear=True):
            self.assertTrue(full_prefill_linear_core_only())
            window = full_prefill_linear_window(case, 32768)
            self.assertEqual(window, plan)
            self.assertTrue(matches_linear_window(window, 12,
                dict(first_position=24576, token_count=8192)))
            self.assertFalse(matches_linear_window(window, 12,
                dict(first_position=24576, token_count=1024)))
            self.assertIsNone(full_prefill_linear_window('q8192-out32', 8192))
            self.assertEqual(observation_positions('q8192-out32', 8192), {8191, 8192})
            labels = full_prefill_linear_labels(12, full_prefill_linear_core_only())
            self.assertEqual(labels, {'linear-12-' + name for name in (
                'q-core-input', 'k-core-input', 'v-core-input', 'g-core-input',
                'beta-core-input', 'initial-state', 'final-state', 'core')})
            with self.assertRaises(ValueError):
                full_prefill_linear_window(case, 32767)
        with patch.dict(os.environ, dict(environment,
                QRT_GB10_FULL_PREFILL_LINEAR_WINDOWS=json.dumps({case: dict(plan, first_position=8192)})),
                clear=True):
            self.assertEqual(observation_positions(case, 32768), {16383, 32767, 32768})
            self.assertEqual(observation_positions('q7169-out32', 7169), {7168, 7169})
            rows = target_rows(list(range(8192, 16384)), [16602]*8192, [],
                               observation_positions(case, 32768))
            self.assertEqual(rows, [dict(row=8191, position=16383, input_token_id=16602,
                                        logit_row=None)])
        with patch.dict(os.environ, dict(environment,
                QRT_GB10_FULL_PREFILL_LINEAR_CORE_ONLY='0'), clear=True):
            with self.assertRaises(ValueError):
                full_prefill_linear_window(case, 32768)
            labels = full_prefill_linear_labels(0, full_prefill_linear_core_only())
            self.assertEqual(len(labels), 17)
            self.assertIn('linear-00-qkv', labels)
            self.assertIn('layer-00-post-attention-rmsnorm', labels)
        for value in ('', 'true', '-1', '2'):
            with patch.dict(os.environ, {'QRT_GB10_FULL_PREFILL_LINEAR_CORE_ONLY': value},
                            clear=True):
                with self.assertRaises(ValueError):
                    full_prefill_linear_window('q8192-out32', 8192)
        with patch.dict(os.environ, dict(environment,
                QRT_GB10_FULL_PREFILL_LINEAR_WINDOWS=json.dumps({case: dict(plan, tokens=8193)})),
                clear=True):
            with self.assertRaises(ValueError):
                full_prefill_linear_window(case, 32769)

    def test_short_moe_capture_requires_the_complete_original_prefill(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(short_prefill_moe_observation(5, 0, 5))
        with patch.dict(os.environ, {'QRT_GB10_SHORT_PREFILL_MOE': '1'}, clear=True):
            for length in (1, 5, 85, 128, 294, 384):
                self.assertTrue(short_prefill_moe_observation(length, 0, length))
            for args in ((0, 0, 0), (385, 0, 385), (7169, 0, 7169),
                         (5, 0, 4), (5, 1, 5), (5, 5, 2)):
                self.assertFalse(short_prefill_moe_observation(*args))

    def test_selected_long_prefill_moe_keeps_actual_chunk_row_identity(self):
        case = 'long-prefix131072-owner-out512'
        plan = {case: [90111, 98303]}
        with patch.dict(os.environ, {'QRT_GB10_CASE_PREFILL_POSITIONS': json.dumps(plan)}, clear=True):
            before = observation_positions(case, 131072)
            self.assertFalse(selected_prefill_moe_observation(131072, 90112, 8192))
            with patch.dict(os.environ, {'QRT_GB10_PREFILL_MOE_SELECTED_ROWS': '1'}):
                self.assertTrue(selected_prefill_moe_observation(131072, 90112, 8192))
                selected = observation_positions(case, 131072)
                self.assertEqual(selected, before)
                rows = target_rows(list(range(90112, 98304)), [42] * 8192, [8191], selected)
                self.assertEqual(rows, [dict(row=8191, position=98303, input_token_id=42, logit_row=0)])
                transaction = dict(first_position=90112, input_token_ids=[42] * 8192, rows=rows)
                self.assertTrue(qualify_transaction(transaction, [42] * 131072)[0]['matches_generated_history'])
                self.assertFalse(selected_prefill_moe_observation(131072, 131072, 2))

    def test_selected_prefill_moe_rejects_unbounded_or_cross_prompt_chunks(self):
        with patch.dict(os.environ, {'QRT_GB10_PREFILL_MOE_SELECTED_ROWS': '1'}, clear=True):
            for args in ((0, 0, 1), (263169, 0, 8192), (131072, -1, 8192),
                         (131072, 90112, 0), (131072, 90112, 8193),
                         (131072, 131071, 2), (131072, True, 8192), (131072, 0, 8192.0)):
                self.assertFalse(selected_prefill_moe_observation(*args))
            self.assertTrue(selected_prefill_moe_observation(263168, 262144, 1024))
        for value in ('', 'true', '2', '-1'):
            with patch.dict(os.environ, {'QRT_GB10_PREFILL_MOE_SELECTED_ROWS': value}, clear=True):
                with self.assertRaisesRegex(ValueError, 'requires 0 or 1'):
                    selected_prefill_moe_observation(131072, 90112, 8192)

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
