"""Exercise actual prefix decode spans, immediate callbacks and rollback."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


def decode_span(source):
    start = source.index('    const uint64_t decode_start_ns = qrt_now_ns();',
                         source.index('QRT_PREFILL_DESCRIPTOR_BATCH_HIP_CALL qrt_qwen36_whole_provider_prefix_v1('))
    end = source.index('    if (gb10_continuation_teacher_forced) {\n        uint32_t prediction_prefix_match_count', start)
    return source[start:end]


class PrefixDecodeStreamTests(unittest.TestCase):
    def test_live_span_callbacks_contract_cancel_and_source_restore(self):
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        legacy = (ROOT / 'tests/native/prefix_legacy_decode_span_fixture.inc').read_text()
        start = whole.index('QRT_PREFILL_DESCRIPTOR_BATCH_HIP_CALL qrt_qwen36_whole_provider_prefix_v1(')
        rollback = function(whole[start:], '    auto rollback_failure =') + ';\n'
        with tempfile.TemporaryDirectory(prefix='qrt-prefix-live-stream-') as temporary:
            directory = Path(temporary)
            (directory / 'prefix_live_decode_span.inc').write_text(rollback + decode_span(whole))
            (directory / 'prefix_legacy_decode_span.inc').write_text(rollback + legacy)
            exe = directory / 'prefix-stream'
            compiled = subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(ROOT), '-I', str(directory),
                str(ROOT / 'tests/native/prefix_decode_stream_host.cpp'), '-o', str(exe)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('live prefix decode spans, cancellation and original owner restoration pass', run.stdout)


if __name__ == '__main__':
    unittest.main()
