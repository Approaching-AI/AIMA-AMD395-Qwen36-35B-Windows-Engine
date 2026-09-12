"""Exercise the actual scope and key-major entry point with host transport."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


def compile_run(source):
    with tempfile.TemporaryDirectory() as tmp:
        exe = str(Path(tmp) / "suffix")
        subprocess.run([
            "c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-fsanitize=undefined", "-fno-sanitize-recover=all",
            "-I", str(ROOT), "-x", "c++", "-", "-o", exe,
        ], input=source, text=True, check=True, timeout=30)
        subprocess.run([exe], check=True, timeout=15, capture_output=True)


class PrefixFlaSuffixTests(unittest.TestCase):
    def test_scope_requires_all_layers_at_one_aligned_position_and_restores(self):
        whole = (ROOT / "native/providers/whole_provider.cpp").read_text()
        scope = function(whole, "struct ScopedQwen36PrefixFlaSingleSuffix {") + ";"
        compile_run("#include <cstdint>\n#include <cstddef>\n#include <cassert>\n" + scope + r'''
int main() {
 using Scope=ScopedQwen36PrefixFlaSingleSuffix;
 assert(!Scope::active);
 for (size_t prefix : {64u,1024u,7168u}) {
  Scope owner(true,prefix,1); assert(owner.enabled&&!owner.complete());
  assert(!owner.claim(0,prefix+1)&&!owner.claim(3,prefix)&&!owner.claim(40,prefix));
  for (unsigned layer=0;layer<40;++layer) if(layer%4!=3) assert(owner.claim(layer,prefix));
  assert(owner.complete()&&!owner.claim(0,prefix));
  {Scope decode(false,prefix+1,1);assert(!decode.enabled&&decode.complete());assert(Scope::active==&decode);}
  assert(Scope::active==&owner);
  try {Scope failing(true,prefix,1);assert(!failing.complete());throw 1;} catch(int){}
  assert(Scope::active==&owner);
 }
 assert(!Scope::active);
 for(size_t prefix : {0u,1u,63u,65u,7169u,8192u,16384u}) {Scope invalid(true,prefix,1);assert(!invalid.enabled);}
 for(size_t suffix : {0u,2u,64u,1024u}) {Scope invalid(true,7168,suffix);assert(!invalid.enabled);}
}
'''.replace("int main()", "#include <initializer_list>\nint main()"))

    def test_key_major_bits_seed_preservation_and_failure_drain(self):
        provider = (ROOT / "native/providers/gdn/qrt_fla_chunk_gdn_q8192_provider.cpp").read_text()
        transpose = function(provider, "__global__ void transpose_seeded_state_bits(")
        export = function(provider, "QRT_FLA_GDN_EXPORT int qrt_fla_gdn_launch_async_seeded_key_major_f32_v1(")
        compile_run(r'''
#include "native/providers/gdn/fla_checkpoint.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>
constexpr unsigned kStateElements=524288;
using hipStream_t=void*;using hipError_t=int;constexpr int hipSuccess=0;
struct {bool prepared=true;float* seeded_row_state=nullptr;} g_state;
bool enabled=true;unsigned calls=0,drains=0,launches=0;int fault=0;
bool blackwell_state_enabled(){return enabled;}bool blackwell_batched_enabled(){return enabled;}
namespace qrt_fla_blackwell_cooperative {bool enabled(){return ::enabled;}}
void set_error_text(const char*){}void set_error(const char*,int){}
int hipMemGetInfo(size_t* a,size_t* t){*a=fault==1?0u:1024u*1024u*1024u;*t=*a;return 0;}
int hipMalloc(void** p,size_t n){if(fault==2)return 1;*p=std::malloc(n);return *p?0:1;}
int hipStreamSynchronize(void*){++drains;return 0;}
int hipGetLastError(){return (fault==3&&launches==1)||(fault==5&&launches==2);}
struct {unsigned x;} blockIdx,blockDim,threadIdx;
struct dim3{unsigned x;explicit dim3(unsigned n):x(n){}};
#define __global__
#define QRT_FLA_GDN_EXPORT
''' + transpose + r'''
void transport(dim3 grid,dim3 block,const uint32_t* in,uint32_t* out){
 ++launches;blockDim.x=block.x;
 for(blockIdx.x=0;blockIdx.x<grid.x;++blockIdx.x)
  for(threadIdx.x=0;threadIdx.x<block.x;++threadIdx.x)transpose_seeded_state_bits(in,out);
}
#define hipLaunchKernelGGL(kernel,grid,block,shared,stream,in,out) transport(grid,block,in,out)
std::vector<uint32_t> row;
int launch_pipeline_async_impl(const float*,const float*,float*,float* state,int,void*,int,
                               const qrt_fla_checkpoint::Plan* plan,bool reset){
 ++calls;assert(!reset&&!plan&&std::memcmp(state,row.data(),qrt_fla_checkpoint::kStateBytes)==0);
 if(fault==4)return 0;
 // A distinguishable row-major result makes a missing/wrong reverse transpose observable.
 for(unsigned i=0;i<kStateElements;++i)std::memcpy(state+i,&row[kStateElements-1-i],4);
 return 1;
}
''' + export + r'''
int main(){
 std::vector<float> raw(8192),gate(64),out(4096),state(kStateElements);
 std::vector<uint32_t> key(kStateElements);row.resize(kStateElements);
 // Includes signed zero, infinities, NaN payloads and subnormals: no float math is permitted.
 for(unsigned h=0;h<32;++h)for(unsigned v=0;v<128;++v)for(unsigned k=0;k<128;++k){
  unsigned i=h*16384+v*128+k;row[i]=i*0x9e3779b9u;key[h*16384+k*128+v]=row[i];
 }
 auto run=[&]{launches=0;std::memcpy(state.data(),key.data(),qrt_fla_checkpoint::kStateBytes);
  return qrt_fla_gdn_launch_async_seeded_key_major_f32_v1(raw.data(),gate.data(),out.data(),state.data(),0,nullptr,1);};
 fault=1;assert(!run()&&calls==0&&launches==0&&!g_state.seeded_row_state);
 fault=2;assert(!run()&&calls==0&&launches==0&&!g_state.seeded_row_state);
 fault=0;assert(run()&&calls==1&&launches==2&&drains==0);
 for(unsigned h=0;h<32;++h)for(unsigned v=0;v<128;++v)for(unsigned k=0;k<128;++k)
  assert(std::memcmp(&state[h*16384+k*128+v],&row[kStateElements-1-(h*16384+v*128+k)],4)==0);
 auto* allocation=g_state.seeded_row_state;assert(run()&&g_state.seeded_row_state==allocation);
 unsigned before=calls;
 enabled=false;assert(!run()&&calls==before&&launches==0);enabled=true;
 g_state.prepared=false;assert(!run()&&calls==before&&launches==0);g_state.prepared=true;
 setenv("QRT_FLA_GDN_CAPTURE_FIRST_DIR","capture",1);assert(!run()&&launches==0);
 unsetenv("QRT_FLA_GDN_CAPTURE_FIRST_DIR");setenv("QRT_FLA_GDN_DUMP_Q64_DIR","dump",1);
 assert(!run()&&launches==0);unsetenv("QRT_FLA_GDN_DUMP_Q64_DIR");
 assert(!qrt_fla_gdn_launch_async_seeded_key_major_f32_v1(raw.data(),gate.data(),raw.data(),state.data(),0,nullptr,1));
 assert(!qrt_fla_gdn_launch_async_seeded_key_major_f32_v1(raw.data(),gate.data(),out.data(),state.data(),0,nullptr,65537));
 fault=3;assert(!run()&&drains==1&&calls==before);
 fault=4;assert(!run()&&drains==2&&calls==before+1);
 fault=5;assert(!run()&&drains==3&&calls==before+2);
 std::free(g_state.seeded_row_state);
}
''')
