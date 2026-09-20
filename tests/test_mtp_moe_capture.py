"""Reject broken draft lineage and tampered captures; fixtures are not model evidence."""
from pathlib import Path
import copy
import hashlib
import math
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from capture_gb10_mtp_moe import (  # noqa: E402
    FRONTIERS, WEIGHTS, observe_original_moe_routed, qualify_moe_capture, qualify_moe_configuration)


CONFIGURATION = dict(tp_size=1, ep_size=1, shared_expert_present=True,
    enable_eplb=False, sequence_parallel=False, use_overlapped=True,
    internal_router=True, router_is_original_module=True,
    shared_is_original_module=True, shared_gate_is_original_module=True)


class MtpMoeConfigurationTests(unittest.TestCase):
    def test_pinned_shared_overlap_and_original_internal_router_are_accepted(self):
        original = copy.deepcopy(CONFIGURATION)
        qualify_moe_configuration(original)
        self.assertEqual(original, CONFIGURATION)

    def test_foreign_router_module_or_distributed_configuration_is_rejected(self):
        for changed in (dict(tp_size=2), dict(ep_size=2), dict(enable_eplb=True),
                dict(router_is_original_module=False), dict(shared_is_original_module=False),
                dict(shared_gate_is_original_module=False)):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'configuration changed'):
                qualify_moe_configuration(dict(CONFIGURATION, **changed))


class MtpRoutedCallTests(unittest.TestCase):
    def test_original_calls_and_objects_are_preserved_and_restored(self):
        for tokens in (1, 2, 7169, 8192):
            with self.subTest(tokens=tokens):
                calls, copies = [], []
                original_result, projection_result = object(), object()
                def dispatch(*args, **kwargs):
                    calls.append((args, kwargs))
                    return projection_result
                module = SimpleNamespace(invoke_fused_moe_triton_kernel=dispatch)
                gate = SimpleNamespace(shape=(tokens, 8, 1024))
                activated = SimpleNamespace(shape=(tokens * 8, 512))
                weighted = SimpleNamespace(shape=(tokens, 8, 2048))
                def forward(argument, *, marker):
                    self.assertEqual((argument, marker), ('original', 13))
                    self.assertIs(module.invoke_fused_moe_triton_kernel('input', 'weights', gate, mode=4), projection_result)
                    self.assertIs(module.invoke_fused_moe_triton_kernel(activated, 'down', weighted, mode=5), projection_result)
                    return original_result
                result = observe_original_moe_routed(forward, module, lambda *args: copies.append(args),
                                                     tokens, 'original', marker=13)
                self.assertIs(result, original_result)
                self.assertIs(module.invoke_fused_moe_triton_kernel, dispatch)
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0], (('input', 'weights', gate), {'mode': 4}))
                self.assertEqual(copies, [('routed-gate-up', gate, 8192),
                    ('routed-activated', activated, 4096), ('routed-weighted', weighted, 16384)])

    def test_copy_error_restores_original_dispatcher_and_prevents_second_call(self):
        calls = []
        def dispatch(*args):
            calls.append(args)
        module = SimpleNamespace(invoke_fused_moe_triton_kernel=dispatch)
        def forward():
            module.invoke_fused_moe_triton_kernel(None, None, SimpleNamespace(shape=(2, 8, 1024)))
            self.fail('second projection must not run after a failed observation')
        def fail(*args):
            raise RuntimeError('copy failed')
        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
            observe_original_moe_routed(forward, module, fail, 2)
        self.assertIs(module.invoke_fused_moe_triton_kernel, dispatch)
        self.assertEqual(len(calls), 1)


class MtpMoeCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'control'
        self.root.mkdir()
        weights_root = self.root.parent / 'mtp-moe-original-weights'
        weights_root.mkdir()
        weights = {label: self.save(weights_root / (label + '.bin'), b'\0' * (math.prod(shape) * 2),
                                   list(shape), 'torch.bfloat16') for label, shape in WEIGHTS.items()}
        files = {}
        for label, (width, dtype) in FRONTIERS.items():
            data = (struct.pack('<16i', *range(8), *range(8)) if label == 'topk-ids' else
                    struct.pack('<16f', *([0.125] * 16)) if label == 'topk-weights' else
                    b'\0' * (2 * width * (2 if dtype == 'torch.bfloat16' else 4)))
            key = 'mtp-moe0000-' + label
            files[key] = dict(self.save(self.root / (key + '.bin'), data, [2, width], dtype),
                              transaction=0, label=label)
        norms = b'\1\0' * 2048 + b'\2\0' * 2048
        original = {
            'mtp0000-moe-output': self.save(self.root / 'moe.bin', b'\0' * 8192, [2, 2048], 'torch.bfloat16'),
            'mtp0000-final-norm': self.save(self.root / 'norm.bin', norms, [2, 2048], 'torch.bfloat16'),
        }
        logits = bytearray(248320 * 4)
        struct.pack_into('<f', logits, 14 * 4, 3.5)
        draft = dict(transaction=0, sampled_row=0, original_result_returned_unchanged=True,
            hidden=self.save(self.root / 'hidden.bin', norms[:4096], [1, 2048], 'torch.bfloat16'),
            logits=self.save(self.root / 'logits.bin', logits, [1, 248320], 'torch.float32'))
        transaction = dict(ordinal=0, sampled_rows=[0], draft_token_ids=[[14]], rows=[
            dict(row=0, selected_for_sampling=True, input_provenance='accepted_history'),
            dict(row=1, selected_for_sampling=False, input_provenance='rejected_padding')])
        self.moe = dict(files=files, draft_logits={'0': draft}, weights=weights, configuration=CONFIGURATION.copy(),
            weights_directory='../mtp-moe-original-weights', original_results_returned_unchanged=True,
            bytes=sum(meta['bytes'] for meta in files.values()) + draft['hidden']['bytes'] + draft['logits']['bytes'])
        self.worker = dict(mtp_moe=self.moe, mtp_boundaries=dict(original_history_qualified=True,
                           transactions=[transaction], files=original))

    def save(self, path, data, shape, dtype):
        path.write_bytes(data)
        return dict(file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    shape=shape, dtype=dtype)

    def qualify(self):
        qualify_moe_capture(self.worker, self.root)

    def change(self, meta, offset, data, update_hash=True):
        path = self.root / meta['file']
        content = bytearray(path.read_bytes())
        content[offset:offset + len(data)] = data
        path.write_bytes(content)
        if update_hash:
            meta['sha256'] = hashlib.sha256(content).hexdigest()

    def test_original_sampling_selects_accepted_row_and_preserves_padding_identity(self):
        before = copy.deepcopy(self.worker['mtp_boundaries'])
        self.qualify()
        self.assertTrue(self.moe['original_frontiers_qualified'])
        self.assertTrue(self.moe['sampled_draft_logits_qualified'])
        self.assertEqual(self.moe['draft_checks'][0]['draft_token_id'], 14)
        self.assertEqual(self.worker['mtp_boundaries'], before)

    def test_full_logit_corruption_away_from_original_argmax_is_detected(self):
        self.change(self.moe['draft_logits']['0']['logits'], 100000 * 4, struct.pack('<f', 4.0), False)
        with self.assertRaisesRegex(ValueError, 'file identity'):
            self.qualify()

    def test_changed_actual_draft_token_is_rejected(self):
        self.worker['mtp_boundaries']['transactions'][0]['draft_token_ids'] = [[15]]
        with self.assertRaisesRegex(ValueError, 'draft token differs'):
            self.qualify()

    def test_rejected_padding_cannot_supply_lm_head_hidden(self):
        self.change(self.moe['draft_logits']['0']['hidden'], 0, b'\2\0' * 2048)
        with self.assertRaisesRegex(ValueError, 'sampled final norm'):
            self.qualify()

    def test_non_finite_original_logits_rejected(self):
        self.change(self.moe['draft_logits']['0']['logits'], 4, struct.pack('<f', float('nan')))
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            self.qualify()

    def test_shared_and_routed_tuple_order_is_checked(self):
        self.change(self.moe['files']['mtp-moe0000-expert-part-0'], 0, b'\1\0')
        with self.assertRaisesRegex(ValueError, 'tuple ordering'):
            self.qualify()

    def test_final_norm_input_is_bound_to_actual_moe_output(self):
        self.change(self.moe['files']['mtp-moe0000-final-hidden'], 0, b'\1\0')
        with self.assertRaisesRegex(ValueError, 'differs from MoE output'):
            self.qualify()

    def test_duplicate_expert_and_broken_renormalization_rejected(self):
        for label, content in (('topk-ids', struct.pack('<i', 1)), ('topk-weights', struct.pack('<f', 0.25))):
            with self.subTest(label=label):
                meta = self.moe['files']['mtp-moe0000-' + label]
                old = (self.root / meta['file']).read_bytes()[:4]
                self.change(meta, 0, content)
                with self.assertRaisesRegex(ValueError, 'routing identities'):
                    self.qualify()
                self.change(meta, 0, old)

    def test_file_traversal_is_rejected(self):
        self.moe['files']['mtp-moe0000-router']['file'] = '../router.bin'
        with self.assertRaisesRegex(ValueError, 'file identity'):
            self.qualify()

    def test_unaccounted_bytes_cannot_qualify(self):
        self.moe['bytes'] += 2
        with self.assertRaisesRegex(ValueError, 'byte accounting'):
            self.qualify()


class MtpMoeCliTests(unittest.TestCase):
    def test_optional_capture_uses_only_both_original_short_controls(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'capture'
            command = [sys.executable, str(ROOT / 'scripts/capture_gb10_token_matrix.py'),
                '--mtp-moe-frontiers', '--mtp-kernel-launches', '--mtp-boundaries', '--runtime-boundaries',
                '--oracle-q7169', str(ROOT / 'contracts/arbitrary_q7169_gb10_oracle.json'),
                '--oracle-q8192', str(ROOT / 'contracts/hprefill_q8192_gb10_oracle.json'),
                '--source-commit', 'a' * 40, '--output-dir', str(directory)]
            subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
            record = json.loads((directory / 'capture.json').read_text())
            self.assertEqual([case['name'] for case in record['fixtures']], ['q7169-out32', 'q8192-out32'])
            self.assertTrue(record['mtp_moe_frontiers'])
            self.assertFalse(record['completed'])
            self.assertFalse(record['windows_acceptance'])

    def test_dependent_flags_and_case_scope_are_required_before_capture(self):
        for extra in ([], ['--mtp-kernel-launches', '--mtp-boundaries', '--runtime-boundaries',
                          '--mtp-full-prefill-frontiers'],
                      ['--mtp-kernel-launches', '--mtp-boundaries', '--runtime-boundaries',
                       '--additional-cases', 'unexpected.json']):
            with self.subTest(extra=extra):
                command = [sys.executable, str(ROOT / 'scripts/capture_gb10_token_matrix.py'),
                    '--mtp-moe-frontiers', '--oracle-q7169', 'unused.json', '--oracle-q8192', 'unused.json',
                    '--source-commit', 'a' * 40, '--output-dir', 'unused', *extra]
                run = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 2)
                self.assertIn('--mtp-moe-frontiers requires', run.stderr)


if __name__ == '__main__':
    unittest.main()
