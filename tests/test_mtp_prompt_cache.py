"""Host ownership/failure checks; HIP kernels and numerical outputs are untested here."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MtpPromptCacheTests(unittest.TestCase):
    def test_order_publication_and_quarantine(self):
        source = (ROOT / 'native/providers/gdn/sm121_mtp_prompt_cache.h').read_text()
        # Replace only HIP/kernel declarations with the queued host mocks.
        # Every byte of the production ownership/orchestration body is retained.
        for include in ('#include <hip/hip_runtime.h>\n',
                        '#include "sm121_mtp_frontier.h"\n', '#include "sm121_mtp_kv.h"\n'):
            self.assertEqual(source.count(include), 1)
            source = source.replace(include, '')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'sm121_mtp_prompt_cache_under_test.h').write_text(source)
            exe = directory / 'test'
            build = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(directory),
                str(ROOT / 'tests/native/mtp_prompt_cache_host.cpp'), '-o', str(exe)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
