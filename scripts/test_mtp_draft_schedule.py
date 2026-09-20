"""Boundary regressions for the original scheduled-extent transition rule."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class MtpDraftScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++') or shutil.which('clang++')
        if compiler is None:
            raise unittest.SkipTest('a C++ compiler is required')
        cls.temp = tempfile.TemporaryDirectory(prefix='qrt-mtp-schedule-')
        cls.exe = Path(cls.temp.name) / 'mtp-schedule'
        source = Path(__file__).resolve().parents[1] / 'native/providers/mtp_draft_schedule_probe.cpp'
        subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                        str(source), '-o', str(cls.exe)], check=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_schedule(self, prompt, accepted):
        result = subprocess.run([str(self.exe), str(prompt)],
            input=' '.join(map(str, accepted)), capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines()]

    def test_original_q262143_keeps_accepted_second_row_speculative(self):
        rows = self.run_schedule(262143, [2, 0])
        self.assertEqual([(r['first_position'], r['scheduled_rows']) for r in rows],
                         [(262143, 2), (262145, 1)])

    def test_original_q262142_switches_after_final_pair(self):
        rows = self.run_schedule(262142, [2, 0])
        self.assertEqual([(r['first_position'], r['speculative']) for r in rows],
                         [(262142, True), (262144, False)])

    def test_rejected_last_draft_still_counts_toward_scheduled_limit(self):
        rows = self.run_schedule(262142, [1, 0])
        self.assertEqual([(r['first_position'], r['scheduled_rows']) for r in rows],
                         [(262142, 2), (262143, 1)])

    def test_owner_at_limit_and_later_suffix_have_no_initial_draft(self):
        for prompt in (262144, 263168):
            with self.subTest(prompt=prompt):
                self.assertFalse(self.run_schedule(prompt, [0])[0]['speculative'])

    def test_original_q262140_allows_two_final_pairs(self):
        rows = self.run_schedule(262140, [2, 2, 0])
        self.assertEqual([r['first_position'] for r in rows], [262140, 262142, 262144])
        self.assertEqual([r['speculative'] for r in rows], [True, True, False])

    def test_impossible_acceptance_and_position_overflow_fail(self):
        for prompt, accepted in ((262144, '2'), (2**64 - 1, '1')):
            with self.subTest(prompt=prompt):
                result = subprocess.run([str(self.exe), str(prompt)], input=accepted,
                    capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
