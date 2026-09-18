"""Check the real class/expert launcher, ranges, and submission failures."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MoeClassExpertOrderTests(unittest.TestCase):
    def test_actual_launcher_arguments_and_failures(self):
        source = r'''
#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3 {unsigned x;explicit dim3(unsigned v=1):x(v){}};
dim3 blockIdx,blockDim,threadIdx,gridDim;
#define __global__
#define __device__
#define __forceinline__ inline
unsigned atomicAdd(unsigned* p,unsigned v){unsigned old=*p;*p+=v;return old;}
struct Call {std::string name;unsigned grid,block;std::vector<uintptr_t> args;};
std::vector<Call> calls;
unsigned fail=0,status_calls=0,expected_blocks=0;
hipStream_t expected_stream=reinterpret_cast<void*>(uintptr_t(123));
template<class T> uintptr_t word(T value){if constexpr(std::is_pointer_v<T>)return reinterpret_cast<uintptr_t>(value);else return uintptr_t(value);}
template<class T> void append(std::vector<uintptr_t>& args,T value){
    if constexpr(std::is_pointer_v<T>||std::is_integral_v<T>||std::is_enum_v<T>)args.push_back(word(value));
    else {args.push_back(word(value.input));args.push_back(word(value.weights));args.push_back(word(value.projection));}
}
template<class... Args> void submit(const char* name,dim3 grid,dim3 block,unsigned shared,hipStream_t stream,Args... args){
    assert(stream==expected_stream&&shared==0&&block.x==256);
    assert(grid.x==(std::string(name)=="prefix"?1u:expected_blocks));
    std::vector<uintptr_t> values;(append(values,args),...);
    calls.push_back({name,grid.x,block.x,values});
}
#define hipLaunchKernelGGL(kernel,...) submit(#kernel,__VA_ARGS__)
hipError_t hipMemsetAsync(void* p,int value,size_t bytes,hipStream_t stream){
    assert(stream==expected_stream&&value==0&&(bytes==1024||bytes==3072));
    calls.push_back({"memset",0,0,{word(p),bytes}});
    return ++status_calls==fail?hipErrorUnknown:hipSuccess;
}
hipError_t hipGetLastError(){return ++status_calls==fail?hipErrorUnknown:hipSuccess;}
#include "native/providers/triton_moe/class_expert_candidate_order.h"
int main(){
    namespace order=qrt_moe_class_expert_order;
    assert(order::bytes(0)==0&&order::bytes(4194305)==0&&order::bytes(4194304)==16786436);
    uint32_t indices=0,count=0,input_flags=0,weight_flags=0;int32_t ids=0;
    for(size_t capacity:{size_t(1),size_t(256),size_t(257),size_t(4194304)}){
        std::vector<uint32_t> storage(capacity+order::metadata_words);
        const auto view=order::views({storage.data(),capacity});
        assert(view.indices==storage.data()&&view.counts==storage.data()+capacity);
        assert(view.offsets==view.counts+768&&view.cursors==view.offsets+769);
        expected_blocks=unsigned((capacity+255)/256);if(expected_blocks>1024)expected_blocks=1024;
        for(auto projection:{order::Projection::Gate,order::Projection::Up,order::Projection::Down})for(fail=0;fail<=4;++fail){
            calls.clear();status_calls=0;
            const auto result=order::launch(&indices,&count,&ids,{&input_flags,&weight_flags,projection},{storage.data(),capacity},expected_stream);
            assert(result==(fail?hipErrorUnknown:hipSuccess));
            assert(calls.size()==(fail?fail:4u)&&status_calls==calls.size());
            assert(calls[0].name=="memset"&&calls[0].args==std::vector<uintptr_t>({word(view.counts),3072}));
            if(calls.size()>1)assert(calls[1].name=="histogram"&&calls[1].args==std::vector<uintptr_t>({word(&indices),word(&count),word(&ids),word(&input_flags),word(&weight_flags),word(projection),word(view.counts)}));
            if(calls.size()>2)assert(calls[2].name=="prefix"&&calls[2].args==std::vector<uintptr_t>({word(view.counts),word(view.offsets),word(view.cursors)}));
            if(calls.size()>3)assert(calls[3].name=="scatter"&&calls[3].args==std::vector<uintptr_t>({word(&indices),word(&count),word(&ids),word(&input_flags),word(&weight_flags),word(projection),word(view.cursors),word(storage.data())}));
        }
        auto invalid=[&](const uint32_t* in,const uint32_t* selected,const int32_t* experts,order::Rows rows,order::Workspace workspace){
            calls.clear();status_calls=0;
            assert(order::launch(in,selected,experts,rows,workspace,expected_stream)==hipErrorInvalidValue);
            assert(calls.empty()&&status_calls==0);
        };
        const order::Rows rows{&input_flags,&weight_flags,order::Projection::Gate};
        invalid(nullptr,&count,&ids,rows,{storage.data(),capacity});
        invalid(&indices,nullptr,&ids,rows,{storage.data(),capacity});
        invalid(&indices,&count,nullptr,rows,{storage.data(),capacity});
        invalid(&indices,&count,&ids,{nullptr,&weight_flags,rows.projection},{storage.data(),capacity});
        invalid(&indices,&count,&ids,{&input_flags,nullptr,rows.projection},{storage.data(),capacity});
        invalid(&indices,&count,&ids,{&input_flags,&weight_flags,order::Projection(3)},{storage.data(),capacity});
        invalid(&indices,&count,&ids,rows,{nullptr,capacity});
        invalid(&indices,&count,&ids,rows,{storage.data(),0});
        invalid(&indices,&count,&ids,rows,{storage.data(),4194305});
        invalid(storage.data(),&count,&ids,rows,{storage.data(),capacity});
    }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'hip').mkdir()
            (folder / 'hip/hip_runtime.h').write_text('#pragma once\n')
            (folder / 'test.cpp').write_text(source)
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                            '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                            '-I', str(folder), '-I', str(ROOT), str(folder / 'test.cpp'),
                            '-o', str(folder / 'test')], check=True, timeout=60)
            subprocess.run([str(folder / 'test')], check=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
