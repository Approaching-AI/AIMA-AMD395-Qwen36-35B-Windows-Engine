"""Check actual bitmap extraction and bounded cooperative launch contracts."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_attention_workspace import function
ROOT=Path(__file__).resolve().parents[1]
class CooperativeHalfProjectionTests(unittest.TestCase):
    def test_bitmap_tails_capacities_and_submission_failures(self):
        h=(ROOT/'native/providers/moe_accumulator/sm121_cooperative_half_projection.h').read_text()
        code=r'''
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <initializer_list>
#define __host__
#define __device__
struct Row{unsigned words[9];};
enum hipError_t{hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3{unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
constexpr unsigned threads=256u,outputs=64u;
template<unsigned R,unsigned K,bool T>void replay_kernel(){}
unsigned calls=0;bool fail=false;dim3 grid_seen;void* stream_seen;void(*kernel_seen)()=nullptr;
template<class... Args>void record(void(*kernel)(),dim3 grid,dim3 block,unsigned shared,void* stream,Args...){++calls;assert(block.x==256u&&!shared);grid_seen=grid;stream_seen=stream;kernel_seen=kernel;}
hipError_t hipGetLastError(){return fail?hipErrorUnknown:hipSuccess;}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
'''+function(h,'__host__ __device__ inline unsigned row_mask(')+'\ntemplate<unsigned Rows,unsigned KGroups>\n'+function(h,'inline hipError_t dispatch(')+function(h,'inline hipError_t launch(')+r'''
int main(){
 for(unsigned rows:{1u,2u,15u,16u,17u,31u,32u,33u,65u})for(unsigned tokens:{1u,3u,17u,33u}){
  const size_t cells=size_t(rows)*tokens,words=(cells+31u)/32u;
  std::vector<unsigned> storage(words+6u,0xa5a5a5a5u);auto* mask=storage.data()+3u;
  for(size_t i=0;i<words;++i)mask[i]=0u;
  for(size_t i=0;i<words*32u;++i)if(i>=cells || i%11u==7u)mask[i/32u]|=1u<<(i&31u);
  for(unsigned r=0;r<rows+2u;++r)for(unsigned t=0;t<tokens+2u;++t)for(unsigned width:{16u,32u}){
   unsigned expected=0u;for(unsigned bit=0;bit<width;++bit)if(t<tokens&&r+bit<rows&&(size_t(t)*rows+r+bit)%11u==7u)expected|=1u<<bit;
   assert(row_mask(mask,rows,tokens,r,t,width)==expected);
  }
  for(unsigned i=0;i<3u;++i)assert(storage[i]==0xa5a5a5a5u&&storage[words+3u+i]==0xa5a5a5a5u);
 }
 Row operands[1];unsigned bits[1],trace[1];float values[1];void* stream=reinterpret_cast<void*>(uintptr_t(123));
 auto run=[&](unsigned variant,size_t wc=561u,size_t ic=289u,size_t mc=18u,size_t oc=561u,uint32_t* tr=nullptr,size_t tc=0u){return launch(operands,wc,operands,ic,bits,mc,values,oc,33u,17u,272u,variant,stream,tr,tc);};
 for(unsigned variant:{0u,1u,2u})for(bool traced:{false,true})for(bool bad:{false,true}){
  calls=0;fail=bad;assert(run(variant,561u,289u,18u,561u,traced?trace:nullptr,traced?28611u:0u)==(bad?hipErrorUnknown:hipSuccess));
  assert(calls==1u&&stream_seen==stream&&grid_seen.x==(variant?32u:24u));
 }
 calls=0;fail=false;
 assert(run(3u)==hipErrorInvalidValue);assert(run(0u,560u)==hipErrorInvalidValue);assert(run(0u,561u,288u)==hipErrorInvalidValue);
 assert(run(0u,561u,289u,17u)==hipErrorInvalidValue);assert(run(0u,561u,289u,18u,560u)==hipErrorInvalidValue);
 assert(run(0u,561u,289u,18u,561u,trace,28610u)==hipErrorInvalidValue);assert(run(0u,561u,289u,18u,561u,nullptr,1u)==hipErrorInvalidValue);
 for(unsigned width:{0u,15u,17u,8193u,UINT32_MAX})assert(launch(operands,SIZE_MAX,operands,SIZE_MAX,bits,SIZE_MAX,values,SIZE_MAX,33u,17u,width,0u,stream)==hipErrorInvalidValue);
 for(unsigned rows:{0u,16385u,UINT32_MAX})assert(launch(operands,SIZE_MAX,operands,SIZE_MAX,bits,SIZE_MAX,values,SIZE_MAX,rows,17u,272u,0u,stream)==hipErrorInvalidValue);
 for(unsigned tokens:{0u,8193u,UINT32_MAX})assert(launch(operands,SIZE_MAX,operands,SIZE_MAX,bits,SIZE_MAX,values,SIZE_MAX,33u,tokens,272u,0u,stream)==hipErrorInvalidValue);
 assert(launch(nullptr,SIZE_MAX,operands,SIZE_MAX,bits,SIZE_MAX,values,SIZE_MAX,33u,17u,272u,0u,stream)==hipErrorInvalidValue);
 assert(launch(operands,SIZE_MAX,nullptr,SIZE_MAX,bits,SIZE_MAX,values,SIZE_MAX,33u,17u,272u,0u,stream)==hipErrorInvalidValue);
 assert(launch(operands,SIZE_MAX,operands,SIZE_MAX,nullptr,SIZE_MAX,values,SIZE_MAX,33u,17u,272u,0u,stream)==hipErrorInvalidValue);
 assert(launch(operands,SIZE_MAX,operands,SIZE_MAX,bits,SIZE_MAX,nullptr,SIZE_MAX,33u,17u,272u,0u,stream)==hipErrorInvalidValue);assert(!calls);
 assert(launch(operands,SIZE_MAX,operands,SIZE_MAX,bits,SIZE_MAX,values,SIZE_MAX,16384u,8192u,8192u,2u,stream)==hipSuccess&&grid_seen.x==2097152u);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            exe=str(Path(tmp)/'launch')
            subprocess.run(['c++','-std=c++17','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all','-x','c++','-','-o',exe],input=code,text=True,check=True,timeout=30)
            subprocess.run([exe],capture_output=True,check=True,timeout=10)
