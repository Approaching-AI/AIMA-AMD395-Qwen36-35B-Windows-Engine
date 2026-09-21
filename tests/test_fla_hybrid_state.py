"""Actual retry kernel must publish complete CTAs or preserve the original state."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HybridStateGdnTests(unittest.TestCase):
    def test_actual_three_stage_ownership(self):
        separate = (ROOT / 'native/providers/gdn/separate_state_replay.h').read_text()
        hybrid = (ROOT / 'native/providers/gdn/hybrid_state_replay.h').read_text()
        with tempfile.TemporaryDirectory(prefix='qrt-hybrid-state-') as directory:
            work = Path(directory)
            (work / 'separate_state_under_test.h').write_text(separate[separate.index('namespace qrt_fla_separate_state {'):])
            (work / 'hybrid_state_under_test.h').write_text(hybrid[hybrid.index('namespace qrt_fla_hybrid_state {'):])
            executable = work / 'check'
            command = [
                os.environ.get('CXX', 'c++'), '-std=c++17', '-O1', '-ffp-contract=off', '-pthread',
                '-DQRT_TEST_HYBRID_STATE', '-fsanitize=address,undefined,float-cast-overflow',
                '-fno-sanitize-recover=all', '-I', str(ROOT), '-I', str(work),
                str(ROOT / 'tests/native/separate_state_gdn_host.cpp'), '-o', str(executable)]
            build = subprocess.run(command,
                capture_output=True, text=True, timeout=40)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('hybrid_state_host=pass', run.stdout)
            print(run.stdout.strip())
            actual = hybrid[hybrid.index('namespace qrt_fla_hybrid_state {'):]
            publish = ('        for (unsigned cell = tid; cell < Columns * 128u; cell += threads)\n'
                       '            state[(head * 128u + first_column + cell / 128u) * 128u + cell % 128u] = current[cell / 128u][cell % 128u];\n')
            controls = {
                'false_completion': actual.replace('if (!accepted) return;',
                    'if (!accepted) { if (!tid) completed[receipt] = 2u; return; }'),
                'partial_state_publish': actual.replace('        if (!accepted) return;\n    }',
                    publish + '        if (!accepted) return;\n    }'),
            }
            for name, source in controls.items():
                self.assertNotEqual(source, actual)
                (work / 'hybrid_state_under_test.h').write_text(source)
                build = subprocess.run(command, capture_output=True, text=True, timeout=40)
                self.assertEqual(build.returncode, 0, build.stderr)
                options = {}
                if os.name == 'posix':
                    import resource
                    options['preexec_fn'] = lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60, **options)
                self.assertNotEqual(run.returncode, 0, name + ' escaped ownership checks')
                self.assertRegex(run.stderr.lower(), r'assertion.*failed')
                print(f'negative_control_detected={name} exit={run.returncode}')


if __name__ == '__main__':
    unittest.main()
