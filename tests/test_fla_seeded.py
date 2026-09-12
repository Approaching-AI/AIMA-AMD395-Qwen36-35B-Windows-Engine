"""Check the real seeded export and segment dispatch; GPU parity is separate."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class FlaSeededTests(unittest.TestCase):
    def test_invalid_surfaces_and_seed_preservation_across_segment_boundaries(self):
        text = (ROOT / "native/providers/gdn/qrt_fla_chunk_gdn_q8192_provider.cpp").read_text()
        implementation = function(text, "int launch_pipeline_async_impl(")
        export = function(text, "QRT_FLA_GDN_EXPORT int qrt_fla_gdn_launch_async_seeded_f32_v1(")
        source = r'''
#include "native/providers/gdn/fla_checkpoint.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>
constexpr int kChunk=64,kSmokeTokens=64,kSegmentTokens=1024,kQkvRows=8192,kGateRows=64,kValueFeatures=4096;
using hipStream_t=void*;using hipError_t=int;
constexpr int hipSuccess=0,hipMemcpyDeviceToDevice=1;
struct {bool prepared=true;float *padded_postconv=nullptr,*padded_gate=nullptr,*padded_output=nullptr;char error[256]{};} g_state;
bool enabled=true;
bool blackwell_state_enabled(){return enabled;}bool blackwell_batched_enabled(){return enabled;}
namespace qrt_fla_blackwell_cooperative {bool enabled(){return ::enabled;}}
bool supported_tokens(int n){return n>0&&n<=65536;}
int padded_tokens(int n){return (n+63)/64*64;}
void set_error_text(const char*){}void set_error(const char*,int){}
bool ensure_scratch(int n){assert(n>0&&n<=1024&&n%64==0);return true;}
int hipMemsetAsync(void*,int,size_t,void*){return 0;}
int hipMemcpyAsync(void*,const void*,size_t,int,void*){return 0;}
unsigned calls=0,resets=0;int processed=0;
int launch_segment_async(const float*,const float*,float*,float* state,void*,int count,bool reset,
                         int valid,qrt_fla_checkpoint::Segment plan){
 assert(count&&count<=1024&&count%64==0&&!plan.count);if(!valid)valid=count;
 assert(valid>count-64&&valid<=count);++calls;processed+=valid;if(reset){++resets;*state=0;}*state+=float(valid);return 1;
}
#define QRT_FLA_GDN_EXPORT
''' + implementation + export + r'''
int main(){
 // Allocations describe real valid device-sized spans; the transport stand-in
 // never reads or initializes their unused host pages.
 constexpr size_t capacity=8193;
 auto raw=(float*)std::malloc(capacity*8192*4),gate=(float*)std::malloc(capacity*64*4);
 auto out=(float*)std::malloc(capacity*4096*4),state=(float*)std::malloc(2097152);
 assert(raw&&gate&&out&&state);
 g_state.padded_postconv=(float*)std::malloc(64*8192*4);
 g_state.padded_gate=(float*)std::malloc(64*64*4);
 g_state.padded_output=(float*)std::malloc(64*4096*4);
 for(int n:{1,63,64,65,127,128,1023,1024,1025,7169,8192,8193}){
  calls=resets=processed=0;*state=123;
  assert(qrt_fla_gdn_launch_async_seeded_f32_v1(raw,gate,out,state,0,nullptr,n));
  assert(calls&&resets==0&&processed==n&&*state==123+n);
  calls=resets=processed=0;*state=123;
  assert(launch_pipeline_async_impl(raw,gate,out,state,0,nullptr,n));
  assert(resets==1&&processed==n&&*state==n);
 }
 auto reject=[&](const float* r,const float* g,float* o,float* s,int n=65,int decay=0){
  unsigned before=calls;*state=321;
  assert(!qrt_fla_gdn_launch_async_seeded_f32_v1(r,g,o,s,decay,nullptr,n));
  assert(calls==before&&*state==321);
 };
 reject(nullptr,gate,out,state);reject(raw,nullptr,out,state);reject(raw,gate,nullptr,state);reject(raw,gate,out,nullptr);
 reject(raw,gate,out,raw);reject(raw,gate,const_cast<float*>(gate),state);
 reject(raw,raw,out,state);reject(raw,gate,out,state,-1);reject(raw,gate,out,state,0);
 reject(raw,gate,out,state,65537);reject(raw,gate,out,state,65,1);
 reject(raw,gate,out,reinterpret_cast<float*>(UINTPTR_MAX-3u));
 reject(raw,gate,out,reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(state)+1u));
 enabled=false;reject(raw,gate,out,state);enabled=true;
 setenv("QRT_FLA_GDN_CAPTURE_FIRST_DIR","capture",1);reject(raw,gate,out,state);
 unsetenv("QRT_FLA_GDN_CAPTURE_FIRST_DIR");setenv("QRT_FLA_GDN_DUMP_Q64_DIR","dump",1);
 reject(raw,gate,out,state);unsetenv("QRT_FLA_GDN_DUMP_Q64_DIR");
 std::free(raw);std::free(gate);std::free(out);std::free(state);
 std::free(g_state.padded_postconv);std::free(g_state.padded_gate);std::free(g_state.padded_output);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / "seeded")
            subprocess.run([
                "c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=undefined", "-fno-sanitize-recover=all",
                "-I", str(ROOT), "-x", "c++", "-", "-o", exe,
            ], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=15, capture_output=True)
