"""Provenance checks for actual MTP normalization calls."""
import copy
import unittest

from capture_gb10_mtp_launchers import norm_call_identity


class MtpNormCallIdentityTests(unittest.TestCase):
    def layout(self):
        return dict(dtype='torch.bfloat16', shape=[2, 2, 256],
                    stride=[9216, 256, 1], storage_offset=8192)

    def test_original_strided_key_view_is_preserved(self):
        layout = self.layout()
        expected = copy.deepcopy(layout)
        result = norm_call_identity(3, 'k-norm', [layout], ['original-key-kernel'])
        self.assertEqual(result['inputs'], [expected])
        self.assertEqual(layout, expected)
        self.assertTrue(result['original_call_returned_unchanged'])

    def test_missing_actual_launcher_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete original'):
            norm_call_identity(3, 'k-norm', [self.layout()], [])

    def test_inconsistent_tensor_rank_is_rejected(self):
        layout = self.layout()
        layout['stride'] = [512, 1]
        with self.assertRaisesRegex(ValueError, 'tensor layout'):
            norm_call_identity(3, 'k-norm', [layout], ['original-key-kernel'])

    def test_invalid_storage_offset_is_rejected(self):
        layout = self.layout()
        layout['storage_offset'] = -1
        with self.assertRaisesRegex(ValueError, 'tensor layout'):
            norm_call_identity(3, 'k-norm', [layout], ['original-key-kernel'])


if __name__ == '__main__':
    unittest.main()
