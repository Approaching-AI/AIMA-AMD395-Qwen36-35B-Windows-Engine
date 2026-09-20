"""Exercise the real target weight binding, pack and lease code under sanitizers."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ROOT / 'native/providers/gdn'


class Q2ModelWeightTests(unittest.TestCase):
    def test_all_original_tensor_contracts_pack_and_failure_lifetimes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('sm121_mtp_model_weights.h', 'sm121_mtp_resident_weights.h',
                         'sm121_q2_model_weights.h', 'sm121_q2_resident_weights.h'):
                (directory / name).write_text((PROVIDERS / name).read_text().replace('#include <hip/hip_runtime.h>', ''))
            declarations = []
            for filename, name, templated in (
                    ('sm121_q2_linear_block_layout.h', 'LinearBlockViews', True),
                    ('sm121_q2_linear_layer.h', 'LinearLayerViews', True),
                    ('sm121_q2_attention_block_layout.h', 'AttentionBlockViews', False),
                    ('sm121_q2_attention_layer.h', 'AttentionLayerViews', False)):
                text = (PROVIDERS / filename).read_text()
                first = text.index(('template<class Element> ' if templated else '') + 'struct ' + name + ' {')
                last = text.index('\n};', first) + 3
                declarations.append(text[first:last])
            text = (PROVIDERS / 'sm121_mtp_moe.h').read_text()
            first = text.index('struct MoeWeights {')
            moe = text[first:text.index('\n};', first) + 3]
            # Real layer/block and MoE declarations instantiate both adapters.
            # Unused cache/state members are inert in this host ownership test.
            (directory / 'target_weight_types.h').write_text('#pragma once\n#include <cstdint>\n#include <cstddef>\n'
                'namespace qrt_sm121_mtp {\n' + moe + '\n}\nnamespace qrt_sm121_q2 {\n'
                'struct CacheView {}; struct RecurrentViews {};\n'
                'template<class Element> struct ConvolutionViews {const uint16_t* weights=nullptr;};\n'
                + '\n'.join(declarations) + '\n}\n')
            exe = directory / 'check'
            build = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(directory),
                str(ROOT / 'tests/native/q2_model_weights_host.cpp'), '-o', str(exe)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
