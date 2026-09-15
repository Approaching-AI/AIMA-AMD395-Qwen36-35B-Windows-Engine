"""Exercise the real bounded dense integer replay dispatcher without a GPU."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MatrixProjectionLaunchTests(unittest.TestCase):
    def test_dimensions_capacities_tails_and_submission_failure(self):
        header = (ROOT / 'native/providers/moe_accumulator/sm121_matrix_projection.h').read_text()
        code = r'''
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <initializer_list>
enum hipError_t{hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3{unsigned x;explicit dim3(unsigned a):x(a){}};
constexpr unsigned threads=256u;
template<unsigned Groups,bool Trace>void replay_kernel(){}
unsigned calls=0;bool fail=false;unsigned seen_blocks=0;void* seen_stream=nullptr;void(*seen_kernel)()=nullptr;
template<class... Args>void record(void(*kernel)(),dim3 grid,dim3 block,unsigned shared,void* stream,Args...){++calls;assert(block.x==256&&!shared);seen_blocks=grid.x;seen_stream=stream;seen_kernel=kernel;}
hipError_t hipGetLastError(){return fail?hipErrorUnknown:hipSuccess;}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
''' + 'template<unsigned Groups>\n' + function(header, 'inline hipError_t dispatch(') + function(header, 'inline hipError_t launch(') + r'''
int main(){
 uint16_t data[1];unsigned mask[1],trace[1];float output[1];void* stream=reinterpret_cast<void*>(uintptr_t(123));
 auto run=[&](unsigned variant,size_t ww=8976,size_t iw=4624,size_t mw=18,size_t oc=561,uint32_t* tr=nullptr,size_t tc=0){return launch(data,ww,data,iw,mask,mw,output,oc,33,17,272,variant,stream,tr,tc);};
 for(unsigned variant:{0u,1u})for(bool traced:{false,true})for(bool error:{false,true}){
  fail=error;calls=0;assert(run(variant,8976,4624,18,561,traced?trace:nullptr,traced?28611:0)==(error?hipErrorUnknown:hipSuccess));
  assert(calls==1&&seen_blocks==6&&seen_stream==stream);
  auto expected=variant?(traced?replay_kernel<8,true>:replay_kernel<8,false>):(traced?replay_kernel<4,true>:replay_kernel<4,false>);assert(seen_kernel==expected);
 }
 calls=0;fail=false;
 assert(run(2)==hipErrorInvalidValue);assert(run(0,8975)==hipErrorInvalidValue);assert(run(0,8976,4623)==hipErrorInvalidValue);
 assert(run(0,8976,4624,17)==hipErrorInvalidValue);assert(run(0,8976,4624,18,560)==hipErrorInvalidValue);
 assert(run(0,8976,4624,18,561,trace,28610)==hipErrorInvalidValue);assert(run(0,8976,4624,18,561,nullptr,1)==hipErrorInvalidValue);
 for(unsigned width:{0u,15u,17u,8193u,UINT32_MAX})assert(launch(data,SIZE_MAX,data,SIZE_MAX,mask,SIZE_MAX,output,SIZE_MAX,33,17,width,0,stream)==hipErrorInvalidValue);
 for(unsigned rows:{0u,16385u,UINT32_MAX})assert(launch(data,SIZE_MAX,data,SIZE_MAX,mask,SIZE_MAX,output,SIZE_MAX,rows,17,272,0,stream)==hipErrorInvalidValue);
 for(unsigned tokens:{0u,8193u,UINT32_MAX})assert(launch(data,SIZE_MAX,data,SIZE_MAX,mask,SIZE_MAX,output,SIZE_MAX,33,tokens,272,0,stream)==hipErrorInvalidValue);
 assert(launch(nullptr,SIZE_MAX,data,SIZE_MAX,mask,SIZE_MAX,output,SIZE_MAX,33,17,272,0,stream)==hipErrorInvalidValue);
 assert(launch(data,SIZE_MAX,nullptr,SIZE_MAX,mask,SIZE_MAX,output,SIZE_MAX,33,17,272,0,stream)==hipErrorInvalidValue);
 assert(launch(data,SIZE_MAX,data,SIZE_MAX,nullptr,SIZE_MAX,output,SIZE_MAX,33,17,272,0,stream)==hipErrorInvalidValue);
 assert(launch(data,SIZE_MAX,data,SIZE_MAX,mask,SIZE_MAX,nullptr,SIZE_MAX,33,17,272,0,stream)==hipErrorInvalidValue);assert(!calls);
 assert(launch(data,SIZE_MAX,data,SIZE_MAX,mask,SIZE_MAX,output,SIZE_MAX,16384,8192,8192,1,stream)==hipSuccess&&seen_blocks==524288);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'launch')
            subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-x', 'c++', '-', '-o', exe], input=code, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=10)
