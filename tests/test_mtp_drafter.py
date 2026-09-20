"""Exercise the actual request owner with queued kernels and failed completion."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MtpDrafterTests(unittest.TestCase):
    def test_publication_epoch_and_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('sm121_mtp_prompt_cache.h', 'sm121_mtp_drafter.h', 'sm121_mtp_model_weights.h',
                         'sm121_mtp_target_inputs.h', 'sm121_mtp_cache_snapshot.h', 'sm121_mtp_request.h'):
                source = (ROOT / 'native/providers/gdn' / name).read_text()
                source = '\n'.join(line for line in source.splitlines()
                                   if not line.startswith('#include "sm121_')
                                   and line != '#include <hip/hip_runtime.h>') + '\n'
                source = source.replace('#include "../mtp_target_rows.h"',
                    '#include "native/providers/mtp_target_rows.h"')
                source = source.replace('#include "../mtp_decode_rows.h"',
                    '#include "native/providers/mtp_decode_rows.h"')
                (directory / name).write_text(source)
            probe = (ROOT / 'native/providers/sm121_mtp_prefill_probe.h').read_text()
            probe = probe.replace('#include "sm121_mtp_runtime_tables.h"', '')
            probe = probe.replace('#include "gdn/sm121_mtp_target_inputs.h"', '')
            probe = probe.replace('#include "gdn/sm121_mtp_request.h"', '')
            probe = probe.replace('#include "mtp_target_rows_trace.h"',
                '#include "native/providers/mtp_target_rows_trace.h"')
            (directory / 'sm121_mtp_prefill_probe.h').write_text(probe)
            seed = (ROOT / 'native/providers/sm121_mtp_request_seed.h').read_text()
            seed = '\n'.join(line for line in seed.splitlines() if not line.startswith('#include "')) + '\n'
            (directory / 'sm121_mtp_request_seed.h').write_text(seed)
            exe = directory / 'test'
            build = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(directory),
                '-I', str(ROOT), str(ROOT / 'tests/native/mtp_drafter_host.cpp'), '-o', str(exe)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe), str(directory)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('native MTP plain seed cases=5 completion_fences=6 pass', run.stdout)
            for rows in (7169, 8192):
                prefix = directory / f'prefill-{rows}'
                record = json.loads(prefix.with_suffix('.json').read_text())
                self.assertEqual(record['prompt_tokens'], rows)
                self.assertEqual(record['retained_tokens'], rows)
                self.assertEqual(record['draft_position'], rows-1)
                self.assertEqual(record['target_first_token'], 999)
                self.assertEqual(record['draft_token'], 200)
                self.assertEqual(record['draft_logit'], 12.5)
                self.assertTrue(record['native_drafter_executed'])
                self.assertFalse(record['mtp_acceptance_enabled'])
                self.assertFalse(record['reference_data_used_by_compute'])
                self.assertFalse(record['numerical_acceptance_claimed'])
                self.assertEqual(record['model_pack_bytes'], 8388608)
                self.assertEqual(record['input_owned_bytes'], rows*8200)
                self.assertEqual(len(record['tensors']), 25)
                actual_total = 0
                for tensor in record['tensors']:
                    size = Path(str(prefix)+'.'+tensor['name']+'.bin').stat().st_size
                    self.assertEqual(size, tensor['bytes'])
                    actual_total += size
                self.assertEqual(actual_total, record['capture_bytes'])
                self.assertEqual(Path(str(prefix)+'.target.hidden.bf16.bin').stat().st_size, rows*4096)
                self.assertEqual(Path(str(prefix)+'.target.shifted.u32.bin').stat().st_size, rows*4)
                prefix = directory / f'request-{rows}'
                seed = json.loads(Path(str(prefix)+'.request.json').read_text())
                self.assertEqual(seed['schema'], 'qrt-mtp-native-request-seed-v1')
                self.assertEqual(seed['model_epoch'], 10)
                self.assertEqual(seed['target_generation'], 17)
                self.assertEqual(seed['processed_tokens'], rows)
                self.assertEqual(seed['target_current_token'], 999)
                self.assertEqual(seed['next_draft_token'], 200)
                self.assertEqual(seed['next_draft_logit'], 12.5)
                self.assertEqual(seed['checkpoint_bytes'], rows*2048)
                self.assertTrue(seed['actual_target_frontier_matched'])
                self.assertEqual(seed['accepted_target_blocks'], 0)
                self.assertFalse(seed['mtp_acceptance_enabled'])
                self.assertFalse(seed['reference_data_used_by_compute'])
                self.assertFalse(seed['numerical_acceptance_claimed'])
                capture = json.loads(prefix.with_suffix('.json').read_text())
                self.assertEqual(capture['retained_tokens'], rows)
                self.assertEqual(capture['draft_position'], rows-1)
                self.assertEqual(len(capture['tensors']), 25)


if __name__ == '__main__':
    unittest.main()
