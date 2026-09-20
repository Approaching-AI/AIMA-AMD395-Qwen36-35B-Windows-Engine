"""Reject aliased q2 writes before launch; retain both selectable outcomes."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Q2RecurrentTests(unittest.TestCase):
    def test_readonly_frontier_and_failed_launch(self):
        hip = r'''
#pragma once
#include <cstddef>
#include <cassert>
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t = void*;
struct dim3 { unsigned x,y,z; explicit dim3(unsigned a=0,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
inline dim3 blockIdx,threadIdx;
inline unsigned launches=0;
inline hipError_t launch_status=hipSuccess;
#define __global__
#define __shared__ static
inline void __syncthreads() {}
template<class Kernel,class... Args>
void record(Kernel,dim3 grid,dim3 block,size_t shared,hipStream_t stream,Args...) {
    assert(grid.x==32u && grid.y==1u && grid.z==1u);
    assert(block.x==128u && block.y==1u && block.z==1u && !shared && !stream);
    ++launches;
}
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
inline hipError_t hipGetLastError(){return launch_status;}
'''
        source = r'''
#include "native/providers/gdn/sm121_q2_recurrent.h"
#include <vector>
#include <cassert>
using namespace qrt_sm121_q2;
template<class T> const T* fake(unsigned slot) {
    return reinterpret_cast<const T*>((uintptr_t(1)<<36u)+uintptr_t(slot)*(uintptr_t(1)<<30u));
}
int main() {
    std::vector<float> states(2u*state_elements,7.0f);
    std::vector<uint16_t> core(2u*core_elements,0xa5a5u);
    RecurrentViews view{fake<uint16_t>(0),fake<uint16_t>(1),fake<uint16_t>(2),fake<float>(3),
        states.data(),core.data(),false};
    RecurrentTables tables{fake<float>(4),fake<float>(5),fake<unsigned char>(6),fake<unsigned char>(7)};
    const auto reject=[&](RecurrentViews v,RecurrentTables t) {
        const auto before=launches;
        assert(launch_recurrent(v,t)==hipErrorInvalidValue && launches==before);
    };
    for(bool key_major:{false,true}) {
        view.key_major=key_major;
        for(hipError_t status:{hipSuccess,hipErrorUnknown}) {
            launch_status=status;const auto before=launches;
            assert(launch_recurrent(view,tables)==status && launches==before+1u);
        }
        assert(accepted_state(view,1)==states.data());
        assert(accepted_state(view,2)==states.data()+state_elements);
        for(unsigned extent:{0u,3u,~0u})assert(!accepted_state(view,extent));
    }
    auto bad=view;bad.staged_states=nullptr;reject(bad,tables);assert(!accepted_state(bad,1));
    bad=view;bad.staged_core=nullptr;reject(bad,tables);
    bad=view;bad.staged_core=reinterpret_cast<uint16_t*>(states.data())+1u;reject(bad,tables);
    bad=view;bad.staged_states=reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(states.data())+2u);reject(bad,tables);
    bad=view;bad.staged_states=reinterpret_cast<float*>((std::numeric_limits<uintptr_t>::max)()-3u);reject(bad,tables);
    for(unsigned field=0;field<8u;++field)for(bool null:{false,true}) {
        auto v=view;auto t=tables;
        // The start is outside the write span, but its tail aliases it.
        const uintptr_t address=null?0u:reinterpret_cast<uintptr_t>(states.data())-4u;
        switch(field) {
        case 0:v.convolution=reinterpret_cast<const uint16_t*>(address);break;
        case 1:v.a=reinterpret_cast<const uint16_t*>(address);break;
        case 2:v.b=reinterpret_cast<const uint16_t*>(address);break;
        case 3:v.initial_state=reinterpret_cast<const float*>(address);break;
        case 4:t.g=reinterpret_cast<const float*>(address);break;
        case 5:t.beta=reinterpret_cast<const float*>(address);break;
        case 6:t.exp2=reinterpret_cast<const unsigned char*>(address);break;
        case 7:t.rsqrt=reinterpret_cast<const unsigned char*>(address);break;
        }
        reject(v,t);
    }
    for(float value:states)assert(value==7.0f);
    for(uint16_t value:core)assert(value==0xa5a5u);
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'hip').mkdir()
            (root / 'hip/hip_runtime.h').write_text(hip)
            path = root / 'q2.cpp'
            path.write_text(source)
            exe = root / 'q2'
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O1', '-Wall', '-Wextra',
                            '-Werror', '-ffp-contract=off', '-fsanitize=address,undefined',
                            '-fno-sanitize-recover=all', '-I', str(root), '-I', str(ROOT),
                            str(path), '-o', str(exe)], check=True, timeout=60)
            subprocess.run([str(exe)], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
