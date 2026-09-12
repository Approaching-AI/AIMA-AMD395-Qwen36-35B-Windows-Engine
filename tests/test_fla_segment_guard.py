"""Exercise the actual segment wrapper and deferred failure/completion scopes."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class FlaSegmentGuardTests(unittest.TestCase):
    def test_ordered_completion_diagnostics_and_all_failure_paths(self):
        provider = (ROOT/'native/providers/gdn/qrt_fla_chunk_gdn_q8192_provider.cpp').read_text()
        scope = function(provider, 'struct BlackwellSegmentGuard {') + ';'
        timing = 'template<class Operation>\n' + function(provider, 'bool launch_blackwell_math(')
        wrapper = 'int launch_guarded_segment_async(' + provider.split(
            'int launch_guarded_segment_async(', 1)[1].split('int launch_pipeline_async_impl(', 1)[0]
        source = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <initializer_list>
enum hipError_t {hipSuccess,hipErrorUnknown};
using hipEvent_t=void*;using hipStream_t=void*;
constexpr int32_t kSegmentTokens=1024;constexpr unsigned kChunk=64;
unsigned creates,records,waits,drains,destroys,operations,scratch,body_calls;
unsigned fail_create,fail_record,fail_operation;
bool fail_wait=false,fail_scratch=false,batched=true;
float duration=40.0f;int last_tokens=0,last_valid=0;bool last_reset=false;
void* expected_stream=nullptr;
float data[4]{};
struct {char error[768]{};} g_state;
hipError_t hipEventCreate(hipEvent_t* p){if(++creates==fail_create)return hipErrorUnknown;*p=reinterpret_cast<void*>(uintptr_t(creates));return hipSuccess;}
hipError_t hipEventDestroy(hipEvent_t){++destroys;return hipSuccess;}
hipError_t hipEventRecord(hipEvent_t,hipStream_t s){assert(s==expected_stream);return ++records==fail_record?hipErrorUnknown:hipSuccess;}
hipError_t hipEventSynchronize(hipEvent_t){++waits;return fail_wait?hipErrorUnknown:hipSuccess;}
hipError_t hipStreamSynchronize(hipStream_t s){assert(s==expected_stream);++drains;return hipSuccess;}
hipError_t hipEventElapsedTime(float* p,hipEvent_t,hipEvent_t){*p=duration;return hipSuccess;}
void set_error(const char* s,hipError_t){std::snprintf(g_state.error,sizeof(g_state.error),"%s",s);}
void set_error_text(const char* s){set_error(s,hipErrorUnknown);}
bool blackwell_batched_enabled(){return batched;}
bool ensure_scratch(int32_t){++scratch;return !fail_scratch;}
namespace qrt_fla_checkpoint {struct Segment{unsigned count=0;};}
''' + scope + timing + r'''
int launch_segment_async(const float* raw,const float* gates,float* output,float* state,
 void* stream,int32_t tokens,bool reset,int32_t valid,qrt_fla_checkpoint::Segment checkpoints){
 ++body_calls;assert(raw==data&&gates==data+1&&output==data+2&&state==data+3);
 assert(stream==expected_stream&&checkpoints.count==2);last_tokens=tokens;last_valid=valid;last_reset=reset;
 if(tokens<=0||tokens>1024||tokens%64||(valid&&(valid<=tokens-64||valid>tokens)))return 0;
 for(unsigned i=0;i<6;++i){float elapsed=0;
  if(!launch_blackwell_math("original_stage_failure",stream,[&](){return operations++==fail_operation?hipErrorUnknown:hipSuccess;},&elapsed))return 0;
  assert(elapsed==(BlackwellSegmentGuard::active?-1.0f:duration));
 }
 return 1;
}
''' + wrapper + r'''
void reset(){
 assert(!BlackwellSegmentGuard::active);
 creates=records=waits=drains=destroys=operations=scratch=body_calls=0;
 fail_create=fail_record=fail_operation=UINT32_MAX;fail_wait=fail_scratch=false;batched=true;duration=40;
 setenv("QRT_FLA_GDN_SEGMENT_GUARD","1",1);unsetenv("QRT_FLA_GDN_DUMP_Q64_DIR");unsetenv("QRT_FLA_GDN_SYNC_EACH_STAGE");
 g_state.error[0]=0;expected_stream=reinterpret_cast<void*>(uintptr_t(123));
}
int run(int tokens=1024,int valid=0){return launch_guarded_segment_async(data,data+1,data+2,data+3,expected_stream,tokens,true,valid,{2});}
int main(){
 reset();assert(run()&&operations==6&&creates==2&&records==2&&waits==1&&!drains&&destroys==2&&scratch==1&&!BlackwellSegmentGuard::active);
 assert(last_tokens==1024&&!last_valid&&last_reset);
 for(const char* setting:{"QRT_FLA_GDN_DUMP_Q64_DIR","QRT_FLA_GDN_SYNC_EACH_STAGE"}){
  reset();setenv(setting,"1",1);assert(run()&&creates==12&&waits==6&&!scratch);
 }
 reset();batched=false;assert(run()&&creates==12&&waits==6&&!scratch);
 reset();unsetenv("QRT_FLA_GDN_SEGMENT_GUARD");assert(run()&&creates==12&&waits==6&&!scratch);
 for(int invalid:{0,1,63,65,1025}){reset();assert(!run(invalid)&&!creates&&!operations&&!scratch&&body_calls==1);}
 reset();assert(!run(64,65)&&!creates&&!scratch);
 reset();assert(run(64,1)&&last_valid==1&&last_tokens==64&&waits==1);
 reset();fail_scratch=true;assert(!run()&&!creates&&!body_calls);
 for(unsigned i:{0u,2u,5u}){
  reset();fail_operation=i;assert(!run()&&operations==i+1&&creates==2&&records==1&&!waits&&drains==2);
  assert(!BlackwellSegmentGuard::active&&std::strcmp(g_state.error,"original_stage_failure")==0);
 }
 reset();fail_create=2;assert(!run()&&!operations&&!body_calls&&destroys==1);
 reset();fail_record=1;assert(!run()&&!operations&&!body_calls&&!drains&&destroys==2);
 reset();fail_record=2;assert(!run()&&operations==6&&!waits&&drains==1&&!BlackwellSegmentGuard::active);
 reset();fail_wait=true;assert(!run()&&operations==6&&waits==1&&drains==1&&!BlackwellSegmentGuard::active);
 reset();duration=100.01f;assert(!run()&&operations==6&&waits==1&&!drains&&!BlackwellSegmentGuard::active);
 reset();duration=100;assert(run()&&operations==6&&waits==1);
 reset();{BlackwellSegmentGuard scope(expected_stream);assert(!run()&&!creates&&!scratch&&BlackwellSegmentGuard::active==&scope);
  assert(!launch_blackwell_math("foreign",nullptr,[]{++operations;return hipSuccess;}));assert(!operations);
 }
 assert(!BlackwellSegmentGuard::active);
}
'''
        with tempfile.TemporaryDirectory() as temp:
            exe = str(Path(temp)/'segment-guard')
            subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=undefined',
                '-fno-sanitize-recover=all','-x','c++','-','-o',exe],
                input=source,text=True,check=True,timeout=30)
            subprocess.run([exe],check=True,capture_output=True,timeout=10)
