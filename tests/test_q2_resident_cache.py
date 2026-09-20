"""Bind the real provider session declarations to target publication under ASan/UBSan."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from tests.q2_target_harness import write_target_harness

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ROOT / 'native/providers/gdn'


def declaration(source, marker):
    start = source.index(marker)
    return source[start:source.index('\n};', start) + 3]


class Q2ResidentCacheTests(unittest.TestCase):
    def test_actual_session_accepted_metadata_and_lifetime(self):
        with tempfile.TemporaryDirectory(prefix='qrt-q2-resident-cache-') as temporary:
            directory = Path(temporary)
            write_target_harness(directory)
            (directory / 'sm121_q2_resident_cache.h').write_text((PROVIDERS / 'sm121_q2_resident_cache.h').read_text())
            runtime = (ROOT / 'tests/native/q2_target_host.cpp').read_text().split('int main(){')[0]
            runtime = runtime.replace('static void verify(', '[[maybe_unused]] static void verify(', 1)
            (directory / 'target_test_runtime.h').write_text(runtime)
            whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
            session = '\n'.join(declaration(whole, marker) for marker in (
                'enum class Qwen36ResidentSessionElementKind :',
                'struct Qwen36ResidentSessionLinearLayer {',
                'struct Qwen36ResidentSessionFullAttentionLayer {',
                'struct Qwen36ResidentSessionState {'))
            lifetime = whole[whole.index('std::atomic<bool> g_qwen36_resident_completion_unknown'):
                             whole.index('\nQwen36ResidentSessionState g_qwen36_resident_root_session;')]
            (directory / 'resident_session_types.h').write_text('''#pragma once
#include <atomic>
#include <string>
constexpr unsigned QRT_QWEN36_LAYER_COUNT=40, QRT_QWEN36_HIDDEN_SIZE=2048;
constexpr unsigned QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS=64, kQwen36DflashPrefetchedQueueCapacity=64;
struct qrt_engine_t {};
enum class Qwen36ResidentDecodeActivationWorkspacePhase { kIdle, kBusy };
struct Qwen36ResidentDecodeActivationWorkspace {
 bool in_use=false;
 Qwen36ResidentDecodeActivationWorkspacePhase phase=Qwen36ResidentDecodeActivationWorkspacePhase::kIdle;
};
struct Qwen36ResidentPrefixCheckpointStore { unsigned marker=0; };
namespace qrt_sm121_mtp { struct RequestCheckpoint { std::shared_ptr<unsigned> owner; }; }
''' + session + '\n' + lifetime + '\n')
            runtime_types = ''
            for name in ('sm121_q1_runtime', 'sm121_q1_full_runtime', 'sm121_q1_moe_runtime'):
                source = (ROOT / 'native/providers' / (name + '.h')).read_text()
                runtime_types += 'namespace qrt_' + name + ' {\n' + declaration(source, 'struct Tables {') + '\n}\n'
            (directory / 'resident_table_types.h').write_text(runtime_types)
            first = whole.index('using Qwen36TargetCacheOwner = ')
            last = whole.index('bool preload_qwen36_resident_decode_prebound_layer_plan_locked(', first)
            (directory / 'resident_cache_factory.h').write_text(whole[first:last])
            exe = directory / 'resident-cache'
            built = subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-ffp-contract=off', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                '-I', str(directory), '-I', str(PROVIDERS),
                str(ROOT / 'tests/native/q2_resident_cache_host.cpp'), '-o', str(exe)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('actual resident cache metadata and rollback pins pass', run.stdout)


if __name__ == '__main__':
    unittest.main()
