"""Exercise cold split ownership, callback clocks and failures in the C path."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class BoundedPrefillSuffixTests(unittest.TestCase):
    def test_cold_seed_stream_cancellation_and_metadata_contract(self):
        source = (ROOT / 'native/src/qrt.c').read_text()
        # Select the definition after prefix_v1_unlocked, not its declaration.
        start = source.index('/* Preserve the reference prefill batch boundary')
        split = function(source[start:], 'static qrt_status_t qrt_qwen36_try_bounded_prefill_suffix(')
        emit = function(source, 'static int qrt_engine_emit_token_stream_event(')
        bridge = function(source, 'static int QRT_CDECL qrt_qwen36_whole_provider_prefix_token_stream_bridge(')
        stream = function(source, 'static qrt_status_t qrt_engine_request_tokens_stream_v1_unlocked(')
        harness = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "native/src/qrt.h"
#include "native/src/qrt_prefix_logit.h"
#define QRT_QWEN36_PRODUCT_Q8192_CONTEXT_TOKENS 8192u
#define QRT_QWEN36_VOCAB_SIZE 248320u
struct qrt_engine {
 int token_stream_active, token_stream_cancelled, token_stream_contract_failed;
 int resident_prefix_cache_seed_capture_active, resident_prefix_cache_session_valid;
 int baseline_output_head_token_emitted;
 uint32_t baseline_output_head_sampled_token_id;
 uint64_t token_stream_prefill_seed_elapsed_ns, token_stream_request_start_ns;
 uint64_t token_stream_last_request_elapsed_ns;
 size_t token_stream_callback_count;
 uint32_t token_stream_emitted_tokens[QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS];
 qrt_token_stream_callback_v1_t token_stream_callback;
 void *token_stream_user_data;
 size_t request_count, token_request_count, last_input_token_count;
 uint32_t last_input_token;
 uint64_t request_elapsed_ns, last_request_elapsed_ns, first_request_elapsed_ns;
 uint64_t last_request_ttft_elapsed_ns, last_request_tpot_elapsed_ns;
 size_t last_request_tpot_sample_count, last_request_output_token_count;
 const char *failure_stage;
};
typedef struct { qrt_engine_t *engine; } qrt_engine_request_serialization_guard_t;
typedef struct { qrt_engine_t *engine; uint64_t expected_session_generation; size_t output_base; }
 qrt_qwen36_token_stream_bridge_t;
static uint64_t clock_ns;
static unsigned seeds, suffixes, locks, unlocks, callback_count;
static int mode, allow_lock=1, eligible=1, cancel_at=-1;
static uint64_t first_wall, first_step;
static uint64_t qrt_now_ns(void) { return ++clock_ns; }
static uint64_t qrt_elapsed_ns(uint64_t start, uint64_t end) { return end-start; }
static void qrt_engine_set_token_request_failure(qrt_engine_t *e, const char *stage, const char *message) {
 (void)message;e->failure_stage=stage;
}
static void qrt_engine_clear_baseline_output_head_report(qrt_engine_t *e) {
 e->baseline_output_head_token_emitted=0;e->baseline_output_head_sampled_token_id=0;
}
static void qrt_engine_clear_resident_prefix_cache_identity(qrt_engine_t *e,int release) {
 assert(!release);e->resident_prefix_cache_session_valid=0;
}
static int qrt_qwen36_whole_provider_direct_entry_eligible(const qrt_engine_t *e,size_t n,size_t out) {
 return eligible && e && n<=10000 && out>0 && out<=QRT_QWEN36_WHOLE_PROVIDER_MAX_OUTPUT_TOKENS;
}
static int qrt_engine_request_serialization_acquire(qrt_engine_t *e,uint32_t op,
 qrt_engine_request_serialization_guard_t *g) {
 assert(op==QRT_QWEN36_REQUEST_SERIALIZATION_OPERATION_ORDINARY);
 if(!allow_lock)return 0;g->engine=e;++locks;return 1;
}
static void qrt_engine_request_serialization_release(qrt_engine_request_serialization_guard_t *g) {
 assert(g->engine);++unlocks;
}
''' + emit + '\n' + bridge + r'''
static qrt_status_t qrt_qwen36_try_bounded_prefill_suffix(qrt_engine_t*,const uint32_t*,size_t,
 uint32_t*,size_t,size_t*,int*);
qrt_status_t qrt_engine_request_tokens(qrt_engine_t *e,const uint32_t *in,size_t n,
 uint32_t *out,size_t capacity,size_t *count) {
 if(n==8192){
  ++seeds;assert(e->resident_prefix_cache_seed_capture_active && !e->token_stream_active);
  assert(capacity==1);clock_ns+=UINT64_C(60000000000);
  ++e->request_count;++e->token_request_count;e->request_elapsed_ns+=60000000000ULL;
  *out=mode==3?QRT_QWEN36_VOCAB_SIZE:42;*count=mode==2?0:1;
  e->baseline_output_head_token_emitted=1;e->baseline_output_head_sampled_token_id=*out;
  e->resident_prefix_cache_session_valid=1;
  assert(qrt_engine_emit_token_stream_event(e,QRT_TOKEN_STREAM_PHASE_PREFILL,0,*out,1,0));
  return mode==1?QRT_STATUS_IO_ERROR:QRT_STATUS_OK;
 }
 int handled=0;qrt_status_t s=qrt_qwen36_try_bounded_prefill_suffix(e,in,n,out,capacity,count,&handled);
 assert(handled);return s;
}
static qrt_status_t qrt_engine_request_tokens_prefix_v1_unlocked(qrt_engine_t *e,
 const uint32_t *in,size_t n,size_t prefix,uint32_t *out,size_t capacity,
 qrt_qwen36_resident_prefix_cache_result_v1_t *r,int stream) {
 ++suffixes;assert(prefix==8192 && n>8192 && in && !e->resident_prefix_cache_seed_capture_active);
 assert(stream==e->token_stream_active && !e->baseline_output_head_token_emitted);
 ++e->request_count;++e->token_request_count;
 if(mode==4)return QRT_STATUS_UNSUPPORTED;
 qrt_qwen36_token_stream_bridge_t b={e,7,0};
 r->ttft_elapsed_ns=mode==6?UINT64_MAX:1000;r->output_token_count=(uint32_t)capacity;
 for(size_t i=0;i<capacity;++i){
  out[i]=100+(uint32_t)i;clock_ns+=1000;
  if(stream && !qrt_qwen36_whole_provider_prefix_token_stream_bridge(&b,7,(uint32_t)i,out[i],1000,i*1000))
   return QRT_STATUS_UNSUPPORTED;
 }
 e->last_request_output_token_count=capacity;
 e->last_request_tpot_sample_count=capacity-1;e->last_request_tpot_elapsed_ns=(capacity-1)*1000;
 if(mode!=5){e->baseline_output_head_token_emitted=1;e->baseline_output_head_sampled_token_id=out[0];}
 return QRT_STATUS_OK;
}
''' + split + '\n' + stream + r'''
static int callback(void *data,const qrt_token_stream_event_v1_t *event) {
 assert(data==(void*)0x1234);assert(event->token_id==100+callback_count);
 assert(event->output_index==callback_count);
 if(!callback_count){first_wall=event->request_elapsed_ns;first_step=event->token_step_elapsed_ns;}
 ++callback_count;return (int)event->output_index!=cancel_at;
}
static qrt_engine_t fresh(void) {
 qrt_engine_t e={0};clock_ns=0;seeds=suffixes=locks=unlocks=callback_count=0;
 mode=0;allow_lock=eligible=1;cancel_at=-1;first_wall=first_step=0;return e;
}
int main(void) {
 uint32_t *in=calloc(10000,sizeof(*in));uint32_t out[512]={0};size_t count;int handled;
 assert(in);in[8192]=63;
 // ABI extension: absent/unknown tags, token mismatch and nonfinite logits
 // cannot turn a seed or unrelated frontier into a valid first logit.
 uint64_t words[2]={0};float logit=-7;
 assert(!qrt_prefix_first_logit_read(words,220,&logit) && logit==-7);
 qrt_prefix_first_logit_store(words,220,9.75f);
 assert(qrt_prefix_first_logit_read(words,220,&logit) && logit==9.75f);
 assert(!qrt_prefix_first_logit_read(words,64,&logit));
 words[0]^=1;assert(!qrt_prefix_first_logit_read(words,220,&logit));
 qrt_prefix_first_logit_store(words,220,INFINITY);assert(words[0]==0 && words[1]==0);
 words[0]=QRT_PREFIX_FIRST_LOGIT_V1_TAG;words[1]=(UINT64_C(220)<<32)|UINT64_C(0x7fc00000);
 assert(!qrt_prefix_first_logit_read(words,220,&logit));
 qrt_engine_t e=fresh();
 assert(qrt_engine_request_tokens_stream_v1_unlocked(&e,in,8193,out,32,&count,callback,(void*)0x1234)==QRT_STATUS_OK);
 assert(count==32 && callback_count==32 && seeds==1 && suffixes==1 && locks==unlocks);
 assert(first_wall>=60000001000ULL && first_step>=60000001000ULL);
 assert(e.last_request_ttft_elapsed_ns==first_step && e.last_request_elapsed_ns>=first_wall);
 assert(e.request_count==1 && e.token_request_count==1 && e.last_input_token_count==8193 && e.last_input_token==63);
 assert(e.first_request_elapsed_ns==e.last_request_elapsed_ns && e.request_elapsed_ns==e.last_request_elapsed_ns);
 assert(e.baseline_output_head_sampled_token_id==100 && !e.token_stream_active && !e.token_stream_callback);
 assert(e.token_stream_prefill_seed_elapsed_ns==0 && !e.resident_prefix_cache_seed_capture_active);
 // A second ordinary call remains cold even when a matching seed exists.
 assert(qrt_engine_request_tokens(&e,in,8193,out,1,&count)==QRT_STATUS_OK);
 assert(seeds==2 && suffixes==2 && count==1 && e.request_count==2);
 // Both ends of the supported suffix extent and the tail/output limit.
 for(size_t n=8193;n<=9216;n+=1023){e=fresh();assert(qrt_engine_request_tokens(&e,in,n,out,512,&count)==QRT_STATUS_OK);assert(count==512);}
 const size_t bypass[]={8191,8192,9217,10000};
 for(size_t i=0;i<4;++i){e=fresh();handled=1;assert(qrt_qwen36_try_bounded_prefill_suffix(&e,in,bypass[i],out,32,&count,&handled)==QRT_STATUS_UNSUPPORTED);assert(!handled && !seeds);}
 for(int fault=1;fault<=6;++fault){
  e=fresh();mode=fault;count=99;
  assert(qrt_engine_request_tokens_stream_v1_unlocked(&e,in,8193,out,32,&count,callback,(void*)0x1234)!=QRT_STATUS_OK);
  assert(count==0 && !e.baseline_output_head_token_emitted && !e.last_request_output_token_count);
  assert(!e.token_stream_active && !e.token_stream_callback && !e.resident_prefix_cache_seed_capture_active);
  assert(!e.token_stream_prefill_seed_elapsed_ns && locks==unlocks && e.request_count==1);
  if(fault<4)assert(!suffixes && !callback_count);
 }
 for(int index=0;index<3;++index){
  e=fresh();cancel_at=index;assert(qrt_engine_request_tokens_stream_v1_unlocked(&e,in,8193,out,32,&count,callback,(void*)0x1234)==QRT_STATUS_UNSUPPORTED);
  assert(callback_count==(unsigned)index+1 && e.token_stream_cancelled && !e.token_stream_active && locks==unlocks && count==0);
 }
 e=fresh();allow_lock=0;assert(qrt_engine_request_tokens(&e,in,8193,out,32,&count)==QRT_STATUS_UNSUPPORTED);assert(!seeds && !locks);
 e=fresh();in[8192]=QRT_QWEN36_VOCAB_SIZE;assert(qrt_engine_request_tokens(&e,in,8193,out,32,&count)==QRT_STATUS_INVALID_ARGUMENT);assert(!seeds && !locks);
 free(in);puts("cold_prefill_suffix: clocks, hidden seed, boundaries, cancellation, ownership and logit ABI pass");return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-prefill-suffix-') as tmp:
            exe = str(Path(tmp) / 'probe')
            subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=undefined', '-I', str(ROOT), '-x', 'c', '-', '-o', exe, '-lm'],
                           input=harness, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
