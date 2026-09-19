"""CPU checks for accepting only correctly aligned original MTP observations."""
import unittest

from capture_gb10_mtp_boundaries import qualify_mtp_capture, selected_draft_rows


class OriginalMtpIdentityTests(unittest.TestCase):
    def trace(self):
        targets = [dict(first_position=0, token_count=3, input_token_ids=[10, 11, 12]),
                   dict(first_position=3, token_count=2, input_token_ids=[13, 14]),
                   dict(first_position=5, token_count=2, input_token_ids=[15, 99]),
                   dict(first_position=6, token_count=2, input_token_ids=[16, 17])]
        proposals = [dict(target_transaction=0, original_max_seq_len=3, first_position=0,
            token_count=3, discarded_target=False, next_token_ids=[13], draft_token_ids=[[14]],
            rows=selected_draft_rows([0, 1, 2], [11, 12, 13], [2])),
            dict(target_transaction=2, original_max_seq_len=7, first_position=5,
            token_count=2, discarded_target=False, next_token_ids=[16], draft_token_ids=[[17]],
            rows=selected_draft_rows([5, 6], [16, 12], [0]))]
        return dict(runtime_boundaries=dict(transactions=targets),
                    mtp_boundaries=dict(transactions=proposals))

    def qualify(self, trace):
        qualify_mtp_capture(trace, [10, 11, 12], [13, 14, 15, 16, 17])

    def test_rejected_row_retained_without_becoming_an_accepted_reference(self):
        trace = self.trace()
        self.qualify(trace)
        capture = trace['mtp_boundaries']
        self.assertEqual(capture['qualified_original_input_rows'], 3)
        rejected = capture['transactions'][1]['rows'][1]
        self.assertFalse(rejected['at_or_before_sample'])
        self.assertFalse(rejected['input_matches_generated_history'])
        self.assertTrue(capture['transactions'][1]['next_target_draft_check']['scheduled_draft_matches'])

    def test_shifted_token_corruption_is_rejected(self):
        trace = self.trace()
        trace['mtp_boundaries']['transactions'][0]['rows'][0]['input_token_id'] = 10
        with self.assertRaisesRegex(ValueError, 'accepted MTP input'):
            self.qualify(trace)

    def test_incorrect_draft_to_target_binding_is_rejected(self):
        trace = self.trace()
        trace['mtp_boundaries']['transactions'][1]['draft_token_ids'] = [[18]]
        with self.assertRaisesRegex(ValueError, 'next original target batch'):
            self.qualify(trace)

    def test_rejected_row_cannot_be_relabeled_as_accepted(self):
        trace = self.trace()
        trace['mtp_boundaries']['transactions'][1]['rows'][1]['at_or_before_sample'] = True
        with self.assertRaisesRegex(ValueError, 'accepted MTP input'):
            self.qualify(trace)

    def test_draft_outside_original_context_bound_is_rejected(self):
        trace = self.trace()
        trace['mtp_boundaries']['transactions'][0]['original_max_seq_len'] = 262144
        with self.assertRaisesRegex(ValueError, 'original target extent'):
            self.qualify(trace)


if __name__ == '__main__':
    unittest.main()
