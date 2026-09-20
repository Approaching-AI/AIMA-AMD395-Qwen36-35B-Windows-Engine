"""Exercise original target head launch slices and stop every partial failure."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT=Path(__file__).resolve().parents[1]


class Q2TargetHeadTests(unittest.TestCase):
    def test_private_k16_logits_and_real_argmax(self):
        actual=function((ROOT/'native/providers/gdn/sm121_q2_head.h').read_text(),'inline hipError_t launch_target_head(')
        code=r'''
#include "native/providers/gdn/sm121_q2_recurrent_layout.h"
#include <cassert>
#include <tuple>
#include <vector>
#include <array>
using namespace qrt_sm121_q2;
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorUnknown};using hipStream_t=void*;
struct dim3 {unsigned x,y,z;explicit dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
namespace qrt_sm121_mtp {constexpr unsigned head_vocabulary=248320u;void head_argmax(){}}
namespace qrt_sm121_q1_moe {template<unsigned K>void projection(){}}
template<class T>T* fake(unsigned slot){return reinterpret_cast<T*>((uintptr_t(1)<<36u)+(uintptr_t(slot)<<32u));}
const uint16_t* weight=fake<uint16_t>(1);const uint16_t* input=fake<uint16_t>(2);uint16_t* logits=fake<uint16_t>(3);
uint32_t* tokens=fake<uint32_t>(4);float* values=fake<float>(5);uint32_t* invalid=fake<uint32_t>(6);
hipStream_t wanted_stream=reinterpret_cast<void*>(0x1234);
unsigned calls=0,fail_at=0,blocks=0;std::array<unsigned,2> processed{};bool argmax=false;
hipError_t hipMemsetAsync(void* p,int byte,size_t bytes,hipStream_t stream){
 assert(p==invalid && !byte && bytes==4u && stream==wanted_stream && calls==0u);++calls;
 return fail_at==calls?hipErrorUnknown:hipSuccess;
}
template<class A,class B,class C,class D>void record(void(*kernel)(),dim3 grid,dim3 block,size_t shared,hipStream_t stream,A a,B b,C c,D d){
 assert(!shared && stream==wanted_stream && grid.y==1u && grid.z==1u && block.x==256u);
 if constexpr(std::is_same<A,const uint16_t*>::value){
  assert(kernel==qrt_sm121_q1_moe::projection<2048u> && !argmax);
  const unsigned row=a==input?0u:1u;assert(a==input+row*2048u);if(row)assert(processed[0]==248320u);
  const unsigned first=processed[row];const unsigned remaining=248320u-first;
  assert(b==weight+size_t(first)*2048u && c==logits+row*248320u+first);
  assert(d==(remaining<blocks*16u?remaining:blocks*16u) && grid.x==(d+15u)/16u);
  processed[row]+=d;
 }else{
  assert(kernel==qrt_sm121_mtp::head_argmax && processed[0]==248320u && processed[1]==248320u && grid.x==2u && !argmax);
  assert(a==logits && b==tokens && c==values && d==invalid);argmax=true;
 }
 ++calls;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
hipError_t hipGetLastError(){return calls==fail_at?hipErrorUnknown:hipSuccess;}
''' + actual + r'''
int main(){
 const auto run=[&](unsigned cap){return launch_target_head(weight,input,logits,tokens,values,invalid,cap,wanted_stream);};
 for(unsigned cap:{1u,257u,1024u,4096u}){
  blocks=cap;const unsigned total=2u*((248320u+cap*16u-1u)/(cap*16u))+2u;
  const std::vector<unsigned> failures=cap==1u?std::vector<unsigned>{0u,1u,2u,total-1u,total}:std::vector<unsigned>{};
  for(unsigned fault=0;fault<=total;++fault){
   if(cap==1u && std::find(failures.begin(),failures.end(),fault)==failures.end())continue;
   calls=0;fail_at=fault;processed={};argmax=false;
   assert(run(cap)==(fault?hipErrorUnknown:hipSuccess));assert(calls==(fault?fault:total));
   assert(argmax==(!fault||fault==total));
  }
 }
 calls=0;fail_at=0;
 for(unsigned cap:{0u,4097u,~0u})assert(run(cap)==hipErrorInvalidValue);
 const auto reject=[&](const uint16_t* w,const uint16_t* x,uint16_t* l,uint32_t* t,float* f,uint32_t* i){
  assert(launch_target_head(w,x,l,t,f,i,1024u,wanted_stream)==hipErrorInvalidValue && !calls);
 };
 reject(nullptr,input,logits,tokens,values,invalid);reject(weight,nullptr,logits,tokens,values,invalid);
 reject(weight,input,nullptr,tokens,values,invalid);reject(weight,input,logits,nullptr,values,invalid);
 reject(weight,input,logits,tokens,nullptr,invalid);reject(weight,input,logits,tokens,values,nullptr);
 reject(weight,input,const_cast<uint16_t*>(weight)+1u,tokens,values,invalid);
 reject(weight,input,const_cast<uint16_t*>(input)+1u,tokens,values,invalid);
 reject(weight,input,logits,reinterpret_cast<uint32_t*>(logits)+1u,values,invalid);
 reject(weight,input,logits,tokens,reinterpret_cast<float*>(tokens),invalid);
 reject(weight,input,logits,tokens,values,reinterpret_cast<uint32_t*>(values)+1u);
 reject(weight,input,logits,tokens,values,reinterpret_cast<uint32_t*>(~uintptr_t(3)));
}
'''
        code=code.replace('#include <array>','#include <array>\n#include <algorithm>\n#include <type_traits>')
        with tempfile.TemporaryDirectory(prefix='qrt-q2-head-') as temporary:
            src=Path(temporary)/'head.cpp';src.write_text(code);exe=Path(temporary)/'head'
            subprocess.run([os.getenv('CXX','c++'),'-std=c++17','-O1','-Wall','-Wextra','-Werror',
                '-ffp-contract=off','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I',str(ROOT),str(src),'-o',str(exe)],check=True,timeout=60)
            subprocess.run([str(exe)],check=True,timeout=15)


if __name__=='__main__':
    unittest.main()
