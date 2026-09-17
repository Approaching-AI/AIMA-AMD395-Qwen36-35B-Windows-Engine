"""Check the actual combined QK callback, including its complete fallback scan."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MicrotileQkLaunchTests(unittest.TestCase):
    def test_fixed_owner_bounds_and_both_submission_failures(self):
        header = (ROOT / 'native/providers/ck_fmha/microtile_exact_qk.h').read_text()
        actual = function(header, 'inline int launch_workspace(')
        source = r'''
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <tuple>
#include <initializer_list>
#include "prepared_decoded_qk_workspace.h"
using hipStream_t=void*;
enum hipError_t { hipSuccess,hipErrorInvalidValue,hipErrorUnknown };
struct dim3 { unsigned x,y,z; dim3(unsigned a,unsigned b=1u,unsigned c=1u):x(a),y(b),z(c){} };
namespace qrt_blackwell_attention { constexpr unsigned kQueryHeads=16u; }
namespace qrt_deferred_qk_fallback { void replay_scan(){} }
unsigned launches=0u,fail=0u,queries=0u,keys=0u,scan_cells=0u;
uintptr_t pointers[5]{},raw[3]{};
template<class K,class... A> void launch(K,dim3 grid,dim3 threads,unsigned,hipStream_t,A... args){
    if(threads.x!=256u)std::abort();
    const auto values=std::make_tuple(args...);
    if constexpr(sizeof...(A)==9u){
        if(launches||grid.y!=16u)std::abort();
        pointers[0]=reinterpret_cast<uintptr_t>(std::get<0>(values));
        pointers[1]=reinterpret_cast<uintptr_t>(std::get<1>(values));
        pointers[2]=reinterpret_cast<uintptr_t>(std::get<2>(values));
        pointers[3]=reinterpret_cast<uintptr_t>(std::get<3>(values));
        pointers[4]=reinterpret_cast<uintptr_t>(std::get<4>(values));
        keys=grid.x;queries=grid.z;
    }else{
        static_assert(sizeof...(A)==7u);
        if(launches!=1u||grid.y!=1u||grid.z!=1u)std::abort();
        raw[0]=reinterpret_cast<uintptr_t>(std::get<0>(values));
        raw[1]=reinterpret_cast<uintptr_t>(std::get<1>(values));
        raw[2]=reinterpret_cast<uintptr_t>(std::get<2>(values));
        scan_cells=grid.x*256u;
    }
    ++launches;
}
hipError_t hipGetLastError(){return launches==fail?hipErrorUnknown:hipSuccess;}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) launch(kernel,__VA_ARGS__)
namespace qrt_microtile_exact_qk {
template<unsigned Q,unsigned K> void scores(){}
''' + actual + r'''
}
int main(){
    using namespace qrt_prepared_decoded_qk;
    using qrt_microtile_exact_qk::launch_workspace;
    std::unique_ptr<uint32_t[]> storage(new uint32_t[workspace_words]);
    Workspace workspace{storage.get(),8192u};uint16_t query=0u,key=0u;float output=0.0f;
    const uintptr_t base=reinterpret_cast<uintptr_t>(storage.get());
    auto score=[&](unsigned start,unsigned count,unsigned stride,unsigned key_stride){
        return launch_workspace(&workspace,&query,&key,&output,nullptr,start,count,stride,key_stride);
    };
    if(score(0,0,0,8192)!=hipErrorInvalidValue||score(0,129,129,8192)!=hipErrorInvalidValue||
       score(8192,1,8193,8192)!=hipErrorInvalidValue||score(8191,2,8193,8192)!=hipErrorInvalidValue||
       score(0,32,31,8192)!=hipErrorInvalidValue||score(0,32,32,8191)!=hipErrorInvalidValue||
       score(UINT32_MAX,1,0,8192)!=hipErrorInvalidValue||launches)return 1;
    if(launch_workspace(nullptr,&query,&key,&output,nullptr,0,32,32,8192)!=hipErrorInvalidValue||
       launch_workspace(&workspace,nullptr,&key,&output,nullptr,0,32,32,8192)!=hipErrorInvalidValue||
       launch_workspace(&workspace,&query,nullptr,&output,nullptr,0,32,32,8192)!=hipErrorInvalidValue||
       launch_workspace(&workspace,&query,&key,nullptr,nullptr,0,32,32,8192)!=hipErrorInvalidValue||launches)return 2;
    for(const Workspace bad:{Workspace{nullptr,8192u},Workspace{storage.get(),0u},Workspace{storage.get(),8193u}})
        if(launch_workspace(&bad,&query,&key,&output,nullptr,0,32,32,8192)!=hipErrorInvalidValue||launches)return 3;
    for(unsigned tokens:{1u,17u,7169u,8192u})for(unsigned count:{1u,17u,128u}){
        if(count>tokens)continue;
        workspace.tokens=tokens;launches=fail=0u;
        if(score(tokens-count,count,tokens,tokens)!=hipSuccess||launches!=2u||
           keys!=(tokens+31u)/32u||queries!=(count+31u)/32u||
           scan_cells!=((size_t(count)*16u*tokens+255u)/256u)*256u||
           pointers[0]!=base||pointers[1]!=base+query_words*4u||
           pointers[2]!=base+(query_words+key_words)*4u||
           pointers[3]!=base+(query_words+key_words+query_flag_words)*4u||
           pointers[4]!=reinterpret_cast<uintptr_t>(&output)||
           raw[0]!=reinterpret_cast<uintptr_t>(&query)||raw[1]!=reinterpret_cast<uintptr_t>(&key)||
           raw[2]!=reinterpret_cast<uintptr_t>(&output))return 4;
        for(unsigned failure:{1u,2u}){
            launches=0u;fail=failure;
            if(score(tokens-count,count,tokens,tokens)!=hipErrorUnknown||launches!=failure)return 5;
        }
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-microtile-qk-') as tmp:
            exe = str(Path(tmp) / 'launch')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-I', str(ROOT / 'native/providers/ck_fmha'),
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
