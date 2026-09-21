"""Exercise the actual provider request coordinator and ABI callback ordering."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from tests.test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class NativeMtpDecodeTests(unittest.TestCase):
    def test_actual_request_commit_cancel_and_failures(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        coordinator = function(whole, 'bool run_qwen36_native_mtp_decode(')
        prefix_logit = function(whole, 'bool store_qwen36_native_mtp_prefix_first_logit(')
        runtime = (ROOT / 'native/src/qrt.c').read_text()
        validator = function(runtime, 'static int qrt_qwen36_whole_provider_decode_result_valid(')
        with tempfile.TemporaryDirectory(prefix='qrt-native-mtp-decode-') as temporary:
            directory = Path(temporary)
            (directory / 'native_mtp_decode_actual.h').write_text(coordinator + '\n' + prefix_logit + '\n' + validator)
            exe = directory / 'decode'
            built = subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(ROOT), '-I', str(directory),
                str(ROOT / 'tests/native/native_mtp_decode_host.cpp'), '-o', str(exe)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('native MTP decode request ordering and failure recovery pass', run.stdout)
            self.assertIn('native MTP retirement crossings=60 resumed=8 failures=8 cancellations=10 pass', run.stdout)


if __name__ == '__main__':
    unittest.main()
