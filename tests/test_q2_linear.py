"""The two producers share one private convolution and reject all cache aliases."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class Q2LinearTests(unittest.TestCase):
    def test_connected_staging_and_partial_launch_failure(self):
        hip=r'''
#pragma once
#include <cstddef>
#include <cassert>
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t=void*;
struct dim3 {unsigned x,y,z;explicit dim3(unsigned a=0,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
inline dim3 blockIdx,blockDim,threadIdx;
inline unsigned launches=0,fail_at=0;
#define __global__
#define __shared__ static
inline void __syncthreads(){}
template<class Kernel,class... Args>void record(Kernel,dim3 grid,dim3 block,size_t shared,hipStream_t stream,Args...){
 assert(grid.x==32u && grid.y==1u && grid.z==1u && !shared && !stream);
 assert(block.x==(launches?128u:256u) && block.y==1u && block.z==1u);++launches;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
inline hipError_t hipGetLastError(){return launches==fail_at?hipErrorUnknown:hipSuccess;}
'''
        code=r'''
#include "native/providers/gdn/sm121_q2_linear.h"
#include <vector>
#include <cassert>
using namespace qrt_sm121_q2;
template<class T>const T* fake(unsigned slot){return reinterpret_cast<const T*>((uintptr_t(1)<<36u)+(uintptr_t(slot)<<30u));}
template<class Element>void exercise(){
 std::vector<Element> rings(2u*ring_elements);
 std::vector<uint16_t> convolution(2u*8192u),cores(2u*core_elements);
 std::vector<float> states(2u*state_elements);
 ConvolutionViews<Element> cv{fake<uint16_t>(0),fake<Element>(1),fake<uint16_t>(2),fake<unsigned char>(3),rings.data(),convolution.data(),7169u};
 RecurrentViews rv{convolution.data(),fake<uint16_t>(4),fake<uint16_t>(5),fake<float>(6),states.data(),cores.data(),false};
 RecurrentTables tables{fake<float>(7),fake<float>(8),fake<unsigned char>(9),fake<unsigned char>(10)};
 const auto reject=[&](ConvolutionViews<Element> conv,RecurrentViews recurrent){
  launches=0;fail_at=0;assert(launch_linear(conv,recurrent,tables)==hipErrorInvalidValue && !launches);
 };
 for(size_t position:{size_t(0),size_t(1),size_t(2),size_t(3),size_t(7169),size_t(262143),size_t(263678)})for(bool key_major:{false,true}){
  cv.first_position=position;rv.key_major=key_major;
  for(unsigned failure:{0u,1u,2u}){
   launches=0;fail_at=failure;assert(launch_linear(cv,rv,tables)==(failure?hipErrorUnknown:hipSuccess));
   assert(launches==(failure?failure:2u));
  }
  for(unsigned rows:{1u,2u}){
   const auto selected=accepted_linear(cv,rv,rows);assert(selected.rows==rows);
   assert(selected.ring==rings.data()+(rows-1u)*ring_elements && selected.state==states.data()+(rows-1u)*state_elements);
  }
 }
 for(unsigned rows:{0u,3u,~0u}){const auto selected=accepted_linear(cv,rv,rows);assert(!selected.rows && !selected.state && !selected.ring);}
 auto c=cv;auto r=rv;c.first_position=(std::numeric_limits<size_t>::max)();reject(c,r);
 c=cv;c.initial_ring=nullptr;reject(c,r);
 c=cv;c.weights=nullptr;reject(c,r);
 c=cv;c.silu=nullptr;reject(c,r);
 c=cv;c.staged_rings=nullptr;reject(c,r);
 c=cv;c.staged_convolution=nullptr;reject(c,r);
 c=cv;r=rv;r.convolution=fake<uint16_t>(0);reject(c,r);assert(!accepted_linear(c,r,1u).state);
 c=cv;r=rv;c.qkv=reinterpret_cast<const uint16_t*>(states.data())+2u;reject(c,r);
 c=cv;r=rv;c.weights=reinterpret_cast<const uint16_t*>(states.data())+3u;reject(c,r);
 c=cv;r=rv;c.initial_ring=reinterpret_cast<const Element*>(cores.data());reject(c,r);
 c=cv;r=rv;r.initial_state=reinterpret_cast<const float*>(rings.data());reject(c,r);
 c=cv;r=rv;r.a=reinterpret_cast<const uint16_t*>(rings.data())+1u;reject(c,r);
 c=cv;r=rv;r.b=reinterpret_cast<const uint16_t*>(rings.data())+2u;reject(c,r);
 c=cv;r=rv;c.staged_rings=reinterpret_cast<Element*>(states.data())+1u;reject(c,r);
 c=cv;r=rv;c.staged_rings=reinterpret_cast<Element*>(const_cast<float*>(tables.g))+1u;reject(c,r);
 c=cv;r=rv;c.staged_convolution=reinterpret_cast<uint16_t*>(rings.data())+1u;r.convolution=c.staged_convolution;reject(c,r);
}
int main(){exercise<float>();exercise<uint16_t>();}
'''
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'hip').mkdir();(root/'hip/hip_runtime.h').write_text(hip)
            src=root/'q2.cpp';src.write_text(code);exe=root/'q2'
            subprocess.run([os.getenv('CXX','c++'),'-std=c++17','-O1','-Wall','-Wextra','-Werror',
                '-ffp-contract=off','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                '-I',str(root),'-I',str(ROOT),str(src),'-o',str(exe)],check=True,timeout=60)
            subprocess.run([str(exe)],check=True,timeout=15)


if __name__=='__main__':unittest.main()
