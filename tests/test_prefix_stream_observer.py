"""Validate the real-model probe's arrival and cancellation observer."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

from tests.test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]

SOURCE = r'''
#include "tools/prefix_stream_observer.h"
#include <assert.h>
#include <string.h>
static uint64_t clock_ns;
static uint64_t now(void){return clock_ns;}
static qrt_prefix_stream_observer_t observer(void){
    qrt_prefix_stream_observer_t s;
    memset(&s,0,sizeof(s));s.now_ns=now;s.kind="fixture";s.capacity=512u;s.cancel_index=SIZE_MAX;return s;
}
static qrt_token_stream_event_v1_t event(unsigned i){
    qrt_token_stream_event_v1_t e;
    memset(&e,0,sizeof(e));e.struct_size=sizeof(e);e.abi_version=QRT_TOKEN_STREAM_EVENT_ABI_VERSION;
    e.phase=i?QRT_TOKEN_STREAM_PHASE_DECODE:QRT_TOKEN_STREAM_PHASE_PREFILL;e.output_index=i;e.token_id=100u+i;
    e.token_step_elapsed_ns=100000000u;e.provider_decode_elapsed_ns=(uint64_t)i*100000000u;
    e.request_elapsed_ns=1000000000u+e.provider_decode_elapsed_ns;return e;
}
int main(void){
    unsigned i,mode;
    qrt_prefix_stream_observer_t s=observer();
    for(i=0;i<512u;++i){qrt_token_stream_event_v1_t e=event(i);clock_ns=e.request_elapsed_ns;assert(qrt_prefix_observe_token(&s,&e));}
    assert(qrt_prefix_observer_live(&s));
    s=observer();
    for(i=0;i<65u;++i){
        qrt_token_stream_event_v1_t e=event(i);
        clock_ns=!i?1000000000u:1000000000u+((i-1u)/63u+1u)*6300000000ull+i;
        assert(qrt_prefix_observe_token(&s,&e));
    }
    assert(!qrt_prefix_observer_live(&s));
    for(mode=0;mode<4u;++mode){
        const unsigned stop[]={0u,1u,64u,511u};s=observer();s.cancel_index=stop[mode];
        for(i=0;i<=stop[mode];++i){
            qrt_token_stream_event_v1_t e=event(i);clock_ns=e.request_elapsed_ns;
            assert(qrt_prefix_observe_token(&s,&e)==(i!=stop[mode]));
        }
        assert(s.cancelled&&!s.failed&&s.count==stop[mode]+1u&&qrt_prefix_observer_live(&s));
        {qrt_token_stream_event_v1_t e=event(0u);assert(!qrt_prefix_observe_token(&s,&e)&&s.count==stop[mode]+1u);}
    }
    for(mode=0;mode<12u;++mode){
        qrt_token_stream_event_v1_t e=event(0u);uint32_t wrong=999u;
        s=observer();clock_ns=e.request_elapsed_ns;assert(qrt_prefix_observe_token(&s,&e));e=event(1u);clock_ns=e.request_elapsed_ns;
        switch(mode){
        case 0:e.struct_size=0u;break;case 1:e.abi_version++;break;case 2:e.output_index++;break;
        case 3:e.token_id=QRT_QWEN36_VOCAB_SIZE;break;case 4:e.phase=QRT_TOKEN_STREAM_PHASE_PREFILL;break;
        case 5:e.token_step_elapsed_ns=0u;break;case 6:e.request_elapsed_ns=1000000000u;break;
        case 7:e.provider_decode_elapsed_ns=0u;break;case 8:e.provider_decode_elapsed_ns++;break;
        case 9:clock_ns=999999999u;break;case 10:s.expected=&wrong;s.expected_count=1u;break;
        case 11:s.capacity=QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS+1u;break;
        }
        assert(!qrt_prefix_observe_token(&s,&e)&&s.failed&&!qrt_prefix_observer_live(&s));
    }
    puts("observer live/buffered controls, four cancellations and12 invalid events pass");
}
'''


class PrefixStreamObserverTests(unittest.TestCase):
    def test_actual_probe_flow_and_failed_restorations(self):
        product = (ROOT / 'native/src/product_cli.c').read_text()
        preload = function(product, 'static int qrt_product_preload_provider(')
        product = product.replace(preload, '''static int qrt_product_preload_provider(
            const char *path, const char *model, qrt_product_preload_t *result) {
            (void)path;(void)model;memset(result,0,sizeof(*result));result->completed=1;return 1;
        }''')
        source = (ROOT / 'tools/prefix_stream_product_probe.c').read_text().replace(
            '#include "../native/src/product_cli.c"', product)
        source += (ROOT / 'tests/native/prefix_stream_probe_backend.c').read_text()
        with tempfile.TemporaryDirectory(prefix='qrt-prefix-product-probe-') as temporary:
            directory = Path(temporary)
            exe = directory / 'probe'
            compiled = subprocess.run(['cc', '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(ROOT / 'tools'),
                '-I', str(ROOT / 'native/src'), '-x', 'c', '-', '-o', str(exe)], input=source,
                capture_output=True, text=True, timeout=40)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            for name, ids in [('prompt', [1] * 17408), ('suffix', list(range(100, 612))),
                              ('owner', [16, *range(201, 232)])]:
                (directory / (name + '.json')).write_text(json.dumps(ids))
            command = [str(exe), 'run', '--model', 'host-fixture-only', '--tokens', str(directory / 'prompt.json'),
                '--prefix-tokens', '16384', '--prefix-hits', '1', '--output-tokens', '512',
                '--expected-output', str(directory / 'suffix.json'), '--expected-owner-output', str(directory / 'owner.json')]
            for mode in ['valid', 'wrong_owner', 'wrong_cancel_token', 'ignore_cancel', 'no_restore', 'wrong_restored_count']:
                with self.subTest(mode=mode):
                    result = subprocess.run(command, env=dict(os.environ, QRT_PREFIX_PROBE_FIXTURE=mode),
                        capture_output=True, text=True, timeout=15)
                    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
                    summary = next(x for x in records if x['type'] == 'prefix_stream_probe_summary')
                    self.assertEqual(result.returncode == 0, mode == 'valid', result.stdout + result.stderr)
                    self.assertEqual(summary['passed'], mode == 'valid')
                    if mode == 'valid':
                        self.assertEqual((summary['calls'], summary['cancel_passes'], summary['owner_after_cancel_passes']), (10, 4, 4))
                        self.assertFalse(summary['performance_acceptance'])
                        calls = [x for x in records if x['type'] == 'prefix_stream_probe_call' and x['kind'] == 'cancel']
                        self.assertEqual([x['callbacks'] for x in calls], [1, 2, 65, 512])
                        self.assertEqual(len([x for x in records if x['type'] == 'prefix_owner_continuation']), 5)

    def test_arrival_contract_and_buffered_negative_control(self):
        with tempfile.TemporaryDirectory(prefix='qrt-prefix-observer-') as temporary:
            directory = Path(temporary)
            source = directory / 'observer.c'
            source.write_text(SOURCE)
            exe = directory / 'observer'
            compile_result = subprocess.run(['cc', '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(ROOT), str(source), '-o', str(exe)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            spans = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            failures = [x for x in spans if x['type'] == 'prefix_stream_span' and not x['live']]
            self.assertEqual(len(failures), 1)
            self.assertEqual((failures[0]['first'], failures[0]['last']), (1, 63))
            self.assertIn('four cancellations and12 invalid events pass', result.stdout)

    def test_actual_probe_host_syntax(self):
        result = subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsyntax-only',
            'tools/prefix_stream_product_probe.c'], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
