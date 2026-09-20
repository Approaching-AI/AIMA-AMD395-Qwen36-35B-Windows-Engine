"""Exercise the actual request owner with queued kernels and failed completion."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MtpDrafterTests(unittest.TestCase):
    def test_publication_epoch_and_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('sm121_mtp_prompt_cache.h', 'sm121_mtp_drafter.h', 'sm121_mtp_model_weights.h'):
                source = (ROOT / 'native/providers/gdn' / name).read_text()
                source = '\n'.join(line for line in source.splitlines()
                                   if not line.startswith('#include "sm121_')
                                   and line != '#include <hip/hip_runtime.h>') + '\n'
                (directory / name).write_text(source)
            exe = directory / 'test'
            build = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(directory),
                '-I', str(ROOT), str(ROOT / 'tests/native/mtp_drafter_host.cpp'), '-o', str(exe)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
