"""Exercise optional fused selection, alias ownership and submission failures."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_attention_workspace import function
ROOT = Path(__file__).resolve().parents[1]


class FusedStateOutputTests(unittest.TestCase):
    def test_actual_policy_and_launch_failures(self):
        source = (ROOT / 'native/providers/gdn/blackwell_cooperative.cpp').read_text()
        code = r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include "native/providers/gdn/fused_state_output_policy.h"
enum hipError_t{hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3{unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
namespace qrt_fla_fused_state_output{template<unsigned Columns,bool Capture>void kernel(){}}
unsigned score_calls=0,launches=0,queries=0;bool score_failure=false,launch_failure=false;
dim3 grid_seen,block_seen;void(*kernel_seen)()=nullptr;hipStream_t stream_seen=nullptr;
template<class... Args>void record(void(*kernel)(),dim3 grid,dim3 block,unsigned shared,hipStream_t stream,Args...){++launches;assert(!shared);grid_seen=grid;block_seen=block;kernel_seen=kernel;stream_seen=stream;}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
hipError_t hipGetLastError(){++queries;return launch_failure?hipErrorUnknown:hipSuccess;}
namespace qrt_fla_blackwell_cooperative{
hipError_t scores(const uint16_t*,const uint16_t*,const float*,uint16_t*,unsigned count,const unsigned char*,hipStream_t){++score_calls;assert(count&&count<=1024);return score_failure?hipErrorUnknown:hipSuccess;}
''' + function(source, 'hipError_t state_output(') + r'''
}
int main(){
 using qrt_fla_blackwell_cooperative::state_output;
 unsetenv("QRT_FLA_GDN_FUSED_STATE_OUTPUT");assert(qrt_fla_fused_policy::columns()==0);
 for(const char* setting:{"","0","4","8","1","04","8 ","-1","true"}){setenv("QRT_FLA_GDN_FUSED_STATE_OUTPUT",setting,1);const int expected=!setting[0]||!std::strcmp(setting,"0")?0:!std::strcmp(setting,"4")?4:!std::strcmp(setting,"8")?8:-1;assert(qrt_fla_fused_policy::columns()==expected);}
 for(int columns:{-1,0,1,4,8,16})for(unsigned flags=0;flags<32;++flags){const bool batched=flags&1,state=flags&2,cooperative=flags&4;const unsigned lanes=flags&8?1:4,checkpoints=flags&16?1:0;assert(qrt_fla_fused_policy::selected(columns,batched,state,cooperative,lanes,checkpoints)==((columns==4||columns==8)&&batched&&state&&cooperative&&lanes==1&&!checkpoints));}
 uint16_t words[32]{};float floats[32]{};unsigned char table[1]{};auto stream=reinterpret_cast<void*>(uintptr_t(123));
 auto run=[&](unsigned count,unsigned columns,bool capture){return state_output(words,words+1,words+2,words+3,floats,words+4,floats+1,capture?words+5:nullptr,capture?words+6:nullptr,floats+2,count,columns,table,stream);};
 for(unsigned count:{1u,63u,64u,65u,1024u})for(unsigned columns:{4u,8u})for(bool capture:{false,true})for(unsigned failure=0;failure<3;++failure){
  score_failure=failure==1;launch_failure=failure==2;score_calls=launches=queries=0;assert(run(count,columns,capture)==(failure?hipErrorUnknown:hipSuccess));assert(score_calls==1);
  if(score_failure){assert(!launches&&!queries);continue;}
  assert(launches==1&&queries==1&&grid_seen.x==128/columns&&grid_seen.y==32&&grid_seen.z==1&&block_seen.x==256&&stream_seen==stream);
  auto expected=columns==4?(capture?qrt_fla_fused_state_output::kernel<4,true>:qrt_fla_fused_state_output::kernel<4,false>):(capture?qrt_fla_fused_state_output::kernel<8,true>:qrt_fla_fused_state_output::kernel<8,false>);assert(kernel_seen==expected);
 }
 score_failure=launch_failure=false;score_calls=launches=queries=0;
 for(unsigned count:{0u,1025u,UINT32_MAX})assert(run(count,8,false)==hipErrorInvalidValue);
 for(unsigned columns:{0u,1u,16u,UINT32_MAX})assert(run(64,columns,false)==hipErrorInvalidValue);
 assert(state_output(nullptr,words+1,words+2,words+3,floats,words+4,floats+1,nullptr,nullptr,floats+2,64,8,table,stream)==hipErrorInvalidValue);
 assert(state_output(words,words+1,words+2,words+3,floats,words+4,floats+1,words+5,nullptr,floats+2,64,8,table,stream)==hipErrorInvalidValue);
 assert(state_output(words,words+1,words+2,words+3,floats,words+4,floats+1,nullptr,words+6,floats+2,64,8,table,stream)==hipErrorInvalidValue);
 assert(state_output(words,words+1,words+2,words+3,floats,words,floats+1,nullptr,nullptr,floats+2,64,8,table,stream)==hipErrorInvalidValue);
 assert(state_output(words,words+1,words+2,words+3,floats,words+4,floats+2,nullptr,nullptr,floats+2,64,8,table,stream)==hipErrorInvalidValue);
 assert(state_output(words,words+1,words+2,words+3,floats,words+4,floats+1,words+5,words+5,floats+2,64,8,table,stream)==hipErrorInvalidValue);
 assert(state_output(words,words+1,words+2,words+3,floats,words+4,floats+1,nullptr,nullptr,floats+2,64,8,nullptr,stream)==hipErrorInvalidValue);
 assert(!score_calls&&!launches&&!queries);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'launch')
            subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-I', str(ROOT), '-x', 'c++', '-', '-o', exe],
                           input=code, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=10)
