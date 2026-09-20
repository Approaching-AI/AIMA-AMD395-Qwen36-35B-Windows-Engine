"""Exercise actual private block submission, aliases and every partial failure."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Q2LinearBlockTests(unittest.TestCase):
    def test_private_graph_and_submission_failures(self):
        hip = r'''
#pragma once
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <vector>
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3{unsigned x,y,z;explicit dim3(unsigned a=0,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
inline dim3 blockIdx,blockDim,threadIdx;
inline unsigned launches=0,fail_at=0;
inline hipStream_t expected_stream=reinterpret_cast<void*>(0x1234);
struct Event{unsigned k,columns,grid;const uint16_t* input;const uint16_t* weights;uint16_t* output;};
inline std::vector<Event> projections;
#define __global__
#define __shared__ static
inline void __syncthreads(){}
inline float __shfl_down(float x,unsigned,unsigned){return x;}
inline float __shfl(float x,unsigned,unsigned){return x;}
#define QRT_SM121_Q1_MOE_H
namespace qrt_sm121_q1_moe {
template<unsigned K>void projection(const uint16_t*,const uint16_t*,uint16_t*,unsigned){}
}
inline void record(void(*kernel)(const uint16_t*,const uint16_t*,uint16_t*,unsigned),
 dim3 grid,dim3 block,size_t shared,hipStream_t stream,const uint16_t* x,const uint16_t* w,uint16_t* y,unsigned n){
 assert(stream==expected_stream && !shared && grid.y==1 && grid.z==1 && block.x==256);
 unsigned k=kernel==qrt_sm121_q1_moe::projection<2048u>?2048u:4096u;
 assert(kernel==qrt_sm121_q1_moe::projection<2048u> || kernel==qrt_sm121_q1_moe::projection<4096u>);
 projections.push_back({k,n,grid.x,x,w,y});++launches;
}
template<class Kernel,class... Args>void record(Kernel,dim3 grid,dim3 block,size_t shared,hipStream_t stream,Args...){
 assert(stream==expected_stream && !shared && grid.y==1 && grid.z==1);
 assert((launches==8u && grid.x==32u && block.x==256u) ||
        (launches==9u && grid.x==32u && block.x==128u) ||
        (launches==10u && grid.x==64u && block.x==32u));++launches;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
inline hipError_t hipGetLastError(){return launches==fail_at?hipErrorUnknown:hipSuccess;}
'''
        code = r'''
#include "native/providers/gdn/sm121_q2_linear_block.h"
#include <array>
using namespace qrt_sm121_q2;
template<class T>T* fake(unsigned slot){return reinterpret_cast<T*>((uintptr_t(1)<<36u)+(uintptr_t(slot)<<30u));}
template<class Element>void exercise(){
 LinearBlockViews<Element> v;
 v.normalized_input=fake<uint16_t>(1);v.qkv_weights=fake<uint16_t>(2);v.z_weights=fake<uint16_t>(3);
 v.a_weights=fake<uint16_t>(4);v.b_weights=fake<uint16_t>(5);v.output_weights=fake<uint16_t>(6);v.norm_weights=fake<uint16_t>(7);
 v.qkv=fake<uint16_t>(8);v.z=fake<uint16_t>(9);v.a=fake<uint16_t>(10);v.b=fake<uint16_t>(11);
 v.gated=fake<uint16_t>(12);v.output=fake<uint16_t>(13);
 v.convolution={v.qkv,fake<Element>(14),fake<uint16_t>(15),fake<unsigned char>(16),fake<Element>(17),fake<uint16_t>(18),8192u};
 v.recurrent={v.convolution.staged_convolution,v.a,v.b,fake<float>(19),fake<float>(20),fake<uint16_t>(21),false};
 LinearBlockTables tables{{fake<float>(22),fake<float>(23),fake<unsigned char>(24),fake<unsigned char>(25)},fake<float>(26)};
 for(bool key_major:{false,true})for(size_t position:{size_t(0),size_t(7169),size_t(262143)}){
  v.recurrent.key_major=key_major;v.convolution.first_position=position;
  for(unsigned fault=0;fault<=13u;++fault){
   launches=0;fail_at=fault;projections.clear();
   assert(launch_linear_block(v,tables,expected_stream)==(fault?hipErrorUnknown:hipSuccess));
   assert(launches==(fault?fault:13u));
   const std::array<const uint16_t*,5> weights={v.qkv_weights,v.z_weights,v.a_weights,v.b_weights,v.output_weights};
   const std::array<uint16_t*,5> output={v.qkv,v.z,v.a,v.b,v.output};
   const std::array<unsigned,5> columns={8192u,4096u,32u,32u,2048u};
   for(size_t i=0;i<projections.size();++i){const auto& p=projections[i];const auto part=i/2u,row=i%2u;
    const auto* input=part==4u?v.gated:v.normalized_input;const unsigned k=part==4u?4096u:2048u;
    assert(p.k==k && p.columns==columns[part] && p.grid==(p.columns+15u)/16u);
    assert(p.input==input+row*k && p.weights==weights[part] && p.output==output[part]+row*columns[part]);
   }
  }
  for(unsigned accepted:{1u,2u}){
   const auto selected=accepted_linear(v.convolution,v.recurrent,accepted);
   assert(selected.state==v.recurrent.staged_states+(accepted-1u)*state_elements);
   assert(selected.ring==v.convolution.staged_rings+(accepted-1u)*ring_elements);
  }
 }
 const auto reject=[&](LinearBlockViews<Element> bad,LinearBlockTables t){
  launches=0;fail_at=0;assert(launch_linear_block(bad,t,expected_stream)==hipErrorInvalidValue && !launches);
 };
 auto bad=v;bad.convolution.qkv=fake<uint16_t>(27);reject(bad,tables);
 bad=v;bad.recurrent.a=fake<uint16_t>(27);reject(bad,tables);
 bad=v;bad.recurrent.b=fake<uint16_t>(27);reject(bad,tables);
 bad=v;bad.recurrent.convolution=fake<uint16_t>(27);reject(bad,tables);
 const uint16_t* LinearBlockViews<Element>::* reads[]={&LinearBlockViews<Element>::normalized_input,
  &LinearBlockViews<Element>::qkv_weights,&LinearBlockViews<Element>::z_weights,&LinearBlockViews<Element>::a_weights,
  &LinearBlockViews<Element>::b_weights,&LinearBlockViews<Element>::output_weights,&LinearBlockViews<Element>::norm_weights};
 for(auto member:reads){bad=v;bad.*member=nullptr;reject(bad,tables);bad=v;bad.*member=v.gated+1u;reject(bad,tables);}
 uint16_t* LinearBlockViews<Element>::* writes[]={&LinearBlockViews<Element>::qkv,&LinearBlockViews<Element>::z,
  &LinearBlockViews<Element>::a,&LinearBlockViews<Element>::b,&LinearBlockViews<Element>::gated,&LinearBlockViews<Element>::output};
 for(auto member:writes){
  for(auto read:reads){bad=v;bad.*member=const_cast<uint16_t*>(v.*read)+1u;
   bad.convolution.qkv=bad.qkv;bad.recurrent.a=bad.a;bad.recurrent.b=bad.b;reject(bad,tables);}
  for(auto other:writes)if(member!=other){bad=v;bad.*member=(v.*other)+1u;
   bad.convolution.qkv=bad.qkv;bad.recurrent.a=bad.a;bad.recurrent.b=bad.b;reject(bad,tables);}
  bad=v;bad.*member=nullptr;bad.convolution.qkv=bad.qkv;bad.recurrent.a=bad.a;bad.recurrent.b=bad.b;reject(bad,tables);
  bad=v;bad.*member=reinterpret_cast<uint16_t*>(~uintptr_t(1));reject(bad,tables);
 }
 auto t=tables;t.gated_silu=nullptr;reject(v,t);t=tables;t.gated_silu=reinterpret_cast<const float*>(v.gated);reject(v,t);
 bad=v;bad.norm_weights=reinterpret_cast<const uint16_t*>(v.recurrent.staged_states)+1u;reject(bad,tables);
 bad=v;bad.convolution.staged_rings=reinterpret_cast<Element*>(v.gated);reject(bad,tables);
 bad=v;bad.recurrent.staged_core=v.output;reject(bad,tables);
 bad=v;bad.output=reinterpret_cast<uint16_t*>(const_cast<float*>(tables.recurrent.g))+1u;reject(bad,tables);
}
int main(){exercise<float>();exercise<uint16_t>();}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-q2-block-') as temporary:
            root=Path(temporary);(root/'hip').mkdir();(root/'hip/hip_runtime.h').write_text(hip)
            src=root/'block.cpp';src.write_text(code);exe=root/'block'
            subprocess.run([os.getenv('CXX','c++'),'-std=c++17','-O1','-Wall','-Wextra','-Werror',
                '-ffp-contract=off','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                '-I',str(root),'-I',str(ROOT),str(src),'-o',str(exe)],check=True,timeout=60)
            subprocess.run([str(exe)],check=True,timeout=15)


if __name__ == '__main__':
    unittest.main()
