"""Check the real expert-order launcher ABI and every submission failure.

GPU permutation and arithmetic checks live in moe_expert_order_suite.h.
This host harness records submissions; it does not emulate GPU atomics.
"""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MoeExpertOrderTests(unittest.TestCase):
    def test_actual_launch_arguments_and_failures(self):
        source = r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3 {unsigned x; explicit dim3(unsigned v=1):x(v){}};
dim3 blockIdx,blockDim,threadIdx,gridDim;
#define __global__
unsigned atomicAdd(unsigned* p,unsigned value){unsigned old=*p;*p+=value;return old;}
struct Call {std::string name;unsigned grid,block;std::vector<uintptr_t> args;};
std::vector<Call> calls;
unsigned fail=0,status_calls=0,expected_blocks=0;
hipStream_t expected_stream=reinterpret_cast<void*>(uintptr_t(123));
template<class T> uintptr_t word(T v){if constexpr(std::is_pointer_v<T>)return reinterpret_cast<uintptr_t>(v);else return uintptr_t(v);}
template<class... Args> void submit(const char* name,dim3 grid,dim3 block,unsigned shared,hipStream_t stream,Args... args){
    assert(stream==expected_stream&&shared==0&&block.x==256);
    assert(grid.x==(std::string(name)=="prefix"?1u:expected_blocks));
    calls.push_back({name,grid.x,block.x,{word(args)...}});
}
#define hipLaunchKernelGGL(kernel,...) submit(#kernel,__VA_ARGS__)
hipError_t hipMemsetAsync(void* p,int value,size_t bytes,hipStream_t stream){
    assert(stream==expected_stream&&value==0&&bytes==1024);
    calls.push_back({"memset",0,0,{word(p),bytes}});
    return ++status_calls==fail?hipErrorUnknown:hipSuccess;
}
hipError_t hipGetLastError(){return ++status_calls==fail?hipErrorUnknown:hipSuccess;}
#include "native/providers/triton_moe/expert_candidate_order.h"
int main(){
    namespace order=qrt_moe_expert_order;
    assert(order::bytes(0)==0&&order::bytes(4194305)==0&&order::bytes(4194304)==16780292);
    uint32_t indices=0,count=0;int32_t ids=0;
    for(size_t capacity:{size_t(1),size_t(256),size_t(257),size_t(4194304)}){
        std::vector<uint32_t> storage(capacity+769u);
        const auto view=order::views({storage.data(),capacity});
        expected_blocks=unsigned((capacity+255)/256);if(expected_blocks>1024)expected_blocks=1024;
        for(unsigned columns:{512u,2048u})for(fail=0;fail<=4;++fail){
            calls.clear();status_calls=0;
            const auto result=order::launch(&indices,&count,&ids,columns,{storage.data(),capacity},expected_stream);
            assert(result==(fail?hipErrorUnknown:hipSuccess));
            assert(calls.size()==(fail?fail:4u)&&status_calls==calls.size());
            assert(calls[0].name=="memset"&&calls[0].args[0]==word(view.counts));
            if(calls.size()>1)assert(calls[1].name=="histogram"&&calls[1].args==std::vector<uintptr_t>({word(&indices),word(&count),word(&ids),columns,word(view.counts)}));
            if(calls.size()>2)assert(calls[2].name=="prefix"&&calls[2].args==std::vector<uintptr_t>({word(view.counts),word(view.offsets),word(view.cursors)}));
            if(calls.size()>3)assert(calls[3].name=="scatter"&&calls[3].args==std::vector<uintptr_t>({word(&indices),word(&count),word(&ids),columns,word(view.cursors),word(storage.data())}));
        }
        auto invalid=[&](const uint32_t* input,const uint32_t* selected,const int32_t* experts,unsigned columns,order::Workspace workspace){
            calls.clear();status_calls=0;
            assert(order::launch(input,selected,experts,columns,workspace,expected_stream)==hipErrorInvalidValue);
            assert(calls.empty()&&status_calls==0);
        };
        invalid(nullptr,&count,&ids,512,{storage.data(),capacity});
        invalid(&indices,nullptr,&ids,512,{storage.data(),capacity});
        invalid(&indices,&count,nullptr,512,{storage.data(),capacity});
        invalid(&indices,&count,&ids,512,{nullptr,capacity});
        invalid(&indices,&count,&ids,512,{storage.data(),0});
        invalid(&indices,&count,&ids,512,{storage.data(),4194305});
        invalid(&indices,&count,&ids,1024,{storage.data(),capacity});
        invalid(storage.data(),&count,&ids,512,{storage.data(),capacity});
    }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'hip').mkdir()
            (folder / 'hip/hip_runtime.h').write_text('#pragma once\n')
            (folder / 'test.cpp').write_text(source)
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                            '-Wall', '-Wextra', '-Werror', '-I', str(folder),
                            '-I', str(ROOT), str(folder / 'test.cpp'), '-o', str(folder / 'test')],
                           check=True, timeout=60, capture_output=True, text=True)
            subprocess.run([str(folder / 'test')], check=True, timeout=20,
                           capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
