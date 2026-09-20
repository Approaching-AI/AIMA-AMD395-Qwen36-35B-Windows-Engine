"""Real packing/lease code with queued D2D copies and original weight types."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ROOT / 'native/providers/gdn'


class MtpModelWeightTests(unittest.TestCase):
    def test_model_contract_atomic_pack_and_borrowed_lifetime(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            name = 'sm121_mtp_model_weights.h'
            source = (PROVIDERS / name).read_text().replace('#include <hip/hip_runtime.h>', '')
            (directory / name).write_text(source)
            declarations = []
            for filename, declaration in (('sm121_mtp_prompt_cache.h', 'PromptWeights'),
                    ('sm121_mtp_moe.h', 'MoeWeights'), ('sm121_mtp_drafter.h', 'DrafterWeights')):
                text = (PROVIDERS / filename).read_text()
                first = text.index('struct ' + declaration + ' {')
                last = text.index('\n};', first) + 3
                declarations.append(text[first:last])
            (directory / 'mtp_weight_types.h').write_text('#pragma once\n#include <cstdint>\n'
                'namespace qrt_sm121_mtp {\n' + '\n'.join(declarations) + '\n}\n')
            exe = directory / 'check'
            build = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(directory),
                str(ROOT / 'tests/native/mtp_model_weights_host.cpp'), '-o', str(exe)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
