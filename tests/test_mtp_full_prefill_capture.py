"""Validate copied frontier identity; synthetic bytes are not model evidence."""
from pathlib import Path
import hashlib
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from capture_gb10_mtp_full_prefill import (  # noqa: E402
    FULL_FRONTIERS, MAXIMUM_FULL_BYTES, full_prefill_layout, qualify_full_prefill_capture)


class MtpFullPrefillCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompt, self.outputs = [10, 11, 12], [13, 14]
        rows = [dict(row=i, position=i, input_token_id=(11, 12, 13)[i]) for i in (0, 2)]
        self.transaction = dict(ordinal=0, first_position=0, token_count=3,
                                discarded_target=False, rows=rows)
        full_files, selected = {}, {}
        for ordinal, (label, (name, original_width, begin, width)) in enumerate(FULL_FRONTIERS.items()):
            values = [struct.pack('<' + str(original_width) + 'H',
                *((i * 97 + j * 31 + ordinal) % 65536 for j in range(original_width)))
                for i in range(3)]
            data = b''.join(row[begin * 2:(begin + width) * 2] for row in values)
            full_files[name] = dict(self.save('full-' + name, data, [3, width]),
                source_label=label, source_width=original_width, first_feature=begin, transaction=0)
            selected[label] = dict(self.save('selected-' + label,
                b''.join(values[row['row']] for row in rows), [2, original_width]),
                label=label, transaction=0)
        ids = self.save('ids', struct.pack('<3I', 11, 12, 13), [3], 'torch.int32')
        self.full = dict(files=full_files, shifted_input_ids=ids, rows=3, first_position=0,
            transaction=0, bytes=full_prefill_layout(3, self.transaction),
            original_results_returned_unchanged=True)
        self.worker = dict(mtp_full_prefill=self.full, mtp_boundaries=dict(
            original_history_qualified=True, transactions=[self.transaction], files=selected))

    def save(self, name, data, shape, dtype='torch.bfloat16'):
        path = self.root / (name + '.bin')
        path.write_bytes(data)
        return dict(file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    shape=shape, dtype=dtype)

    def alter(self, metadata, offset, update_hash=True):
        path = self.root / metadata['file']
        data = bytearray(path.read_bytes())
        data[offset] ^= 1
        path.write_bytes(data)
        if update_hash:
            metadata['sha256'] = hashlib.sha256(data).hexdigest()

    def qualify(self):
        qualify_full_prefill_capture(self.worker, self.prompt, self.outputs, self.root)

    def test_complete_input_shift_and_all_selected_frontiers_match(self):
        self.qualify()
        self.assertTrue(self.full['original_full_frontiers_qualified'])
        self.assertTrue(self.full['actual_shifted_prompt_ids_qualified'])
        self.assertEqual(self.full['selected_observation_rows_compared'], 22)

    def test_single_prefill_extent_and_byte_bound(self):
        self.assertEqual(full_prefill_layout(8192, dict(self.transaction, token_count=8192)),
                         310411264)
        self.assertLess(310411264, MAXIMUM_FULL_BYTES)
        for size, updates in ((0, {}), (8193, {}), (True, {}),
                (3, dict(ordinal=1)), (3, dict(first_position=1)),
                (3, dict(token_count=2)), (3, dict(discarded_target=True))):
            with self.subTest(size=size, updates=updates), self.assertRaises(ValueError):
                full_prefill_layout(size, dict(self.transaction, **updates))

    def test_middle_shifted_id_is_checked_against_actual_prompt(self):
        self.alter(self.full['shifted_input_ids'], 4)
        with self.assertRaisesRegex(ValueError, 'shifted inputs differ'):
            self.qualify()

    def test_complete_tensor_hash_detects_unsampled_corruption(self):
        self.alter(self.full['files']['fusion'], 2048 * 2, update_hash=False)
        with self.assertRaisesRegex(ValueError, 'file identity changed'):
            self.qualify()

    def test_kv_slice_is_compared_with_original_qkv_offset(self):
        self.alter(self.worker['mtp_boundaries']['files']['qkv'], 8192 * 2)
        with self.assertRaisesRegex(ValueError, 'observations differ'):
            self.qualify()

    def test_missing_frontier_cannot_qualify(self):
        del self.full['files']['k-rope']
        with self.assertRaisesRegex(ValueError, 'incomplete original'):
            self.qualify()

    def test_rejected_selected_row_extent(self):
        self.transaction['rows'][0]['row'] = -1
        with self.assertRaisesRegex(ValueError, 'selected MTP identity'):
            self.qualify()

    def test_selected_tensor_size_is_not_inferred_from_metadata_shape(self):
        meta = self.worker['mtp_boundaries']['files']['target-hidden']
        path = self.root / meta['file']
        data = path.read_bytes() + b'\0\0'
        path.write_bytes(data)
        meta.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        with self.assertRaisesRegex(ValueError, 'selected MTP observation layout'):
            self.qualify()


if __name__ == '__main__':
    unittest.main()
