"""Exercise the actual state submission chain and provider scratch handoff."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class StateReplayLaunchTests(unittest.TestCase):
    def test_submission_failure_aliases_and_checkpoint_fallback(self):
        cooperative = (ROOT / 'native/providers/gdn/blackwell_cooperative.cpp').read_text()
        provider = (ROOT / 'native/providers/gdn/qrt_fla_chunk_gdn_q8192_provider.cpp').read_text()
        wrapper = function(cooperative, 'hipError_t state_replay(')
        # The extractor's first brace must be the body, not the default
        # aggregate argument. Every call below supplies that argument.
        caller = function(provider.replace('Segment checkpoints = {}', 'Segment checkpoints'),
                          'bool launch_blackwell_state(')
        with tempfile.TemporaryDirectory(prefix='qrt-state-replay-launch-') as directory:
            work = Path(directory)
            (work / 'state_replay_wrapper_under_test.h').write_text(wrapper)
            (work / 'state_replay_caller_under_test.h').write_text(caller)
            command = [os.environ.get('CXX', 'c++'), '-std=c++17', '-O1',
                       '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                       '-fno-sanitize-recover=all', '-I', str(ROOT), '-I', str(work),
                       str(ROOT / 'tests/native/state_replay_launch_host.cpp'),
                       '-o', str(work / 'check')]
            build = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(work / 'check')], capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('state_replay_launch_host=pass', run.stdout)
            print(run.stdout.strip())
            # A stale receipt from the preceding segment would skip fast
            # initialization; swallowing a failed retry would publish replay
            # work after the caller's failure. Both errors must be observable.
            broken = {
                'missing_fast': wrapper.replace(
                    'qrt_fla_separate_state::fast_kernel<8u>',
                    'qrt_fla_separate_state::replay_kernel<8u>'),
                'swallowed_launch_failure': wrapper.replace(
                    'if (status != hipSuccess) return status;',
                    'if (status != hipSuccess) status = hipSuccess;'),
            }
            for name, source in broken.items():
                self.assertNotEqual(source, wrapper)
                (work / 'state_replay_wrapper_under_test.h').write_text(source)
                build = subprocess.run(command, capture_output=True, text=True, timeout=30)
                self.assertEqual(build.returncode, 0, build.stderr)
                options = {}
                if os.name == 'posix':
                    import resource
                    options['preexec_fn'] = lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                run = subprocess.run([str(work / 'check')], capture_output=True, text=True,
                                     timeout=20, **options)
                self.assertNotEqual(run.returncode, 0, name + ' escaped checks')
                self.assertRegex(run.stderr.lower(), r'assertion.*failed')
                print(f'negative_control_detected={name} exit={run.returncode}')


if __name__ == '__main__':
    unittest.main()
