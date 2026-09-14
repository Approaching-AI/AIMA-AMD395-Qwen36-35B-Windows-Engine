"""Actual prepared-QK arena views, shape rejection and submission failures."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]

class PreparedDecodedQkLaunchTests(unittest.TestCase):
    def test_arena_views_and_failed_launches(self):
        header=(ROOT/'native/providers/ck_fmha/prepared_decoded_qk.h').read_text()
        actual='\n'.join(function(header,name) for name in ('inline int prepare_workspace(', 'inline int launch_workspace('))
        source=r'''
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <initializer_list>
#include "''' + str(ROOT/'native/providers/ck_fmha/prepared_decoded_qk_workspace.h') + r'''"
using hipStream_t=void*;
enum hipError_t { hipSuccess,hipErrorInvalidValue,hipErrorUnknown };
struct dim3 { unsigned x,y,z; dim3(unsigned a,unsigned b=1u,unsigned c=1u):x(a),y(b),z(c){} };
namespace qrt_blackwell_attention { constexpr unsigned kQueryHeads=16u,kKvHeads=2u,kHeadDim=256u,kThreads=256u; }
unsigned launches=0u,fail=0u;
uintptr_t prepared_pointer[2]{},flag_pointer[2]{},transpose_pointer[2]{},score_pointers[5]{};
unsigned grids[2]{},score_grid=0u,query_grid=0u;
template<class K,class... A> void launch(K,dim3 grid,dim3 threads,unsigned,hipStream_t,A... args) {
    if(threads.x!=256u) std::abort();
    const auto values=std::make_tuple(args...);
    if constexpr(sizeof...(A)==5u) {
        if(launches>=2u)std::abort();
        prepared_pointer[launches]=reinterpret_cast<uintptr_t>(std::get<1>(values));
        flag_pointer[launches]=reinterpret_cast<uintptr_t>(std::get<2>(values));
        transpose_pointer[launches]=reinterpret_cast<uintptr_t>(std::get<3>(values));
        grids[launches]=grid.x;
    } else {
        static_assert(sizeof...(A)==11u);
        score_pointers[0]=reinterpret_cast<uintptr_t>(std::get<2>(values));
        score_pointers[1]=reinterpret_cast<uintptr_t>(std::get<3>(values));
        score_pointers[2]=reinterpret_cast<uintptr_t>(std::get<4>(values));
        score_pointers[3]=reinterpret_cast<uintptr_t>(std::get<5>(values));
        score_pointers[4]=reinterpret_cast<uintptr_t>(std::get<6>(values));
        score_grid=grid.x;query_grid=grid.z;if(grid.y!=16u)std::abort();
    }
    ++launches;
}
hipError_t hipGetLastError(){return launches==fail?hipErrorUnknown:hipSuccess;}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) launch(kernel,__VA_ARGS__)
namespace qrt_prepared_decoded_qk {
using namespace qrt_blackwell_attention;
template<bool Key> void prepare(){}
template<unsigned Window,bool FloatCarry,unsigned Rows=16u,unsigned Keys=16u> void scores(){}
''' + actual + r'''
}
int main() {
    using namespace qrt_prepared_decoded_qk;
    static_assert(workspace_words*4u==151584768u);
    std::unique_ptr<uint32_t[]> storage(new uint32_t[workspace_words]);
    Workspace workspace{storage.get(),8192u};uint16_t input=0u,transposed=0u;float output=0.0f;
    for(const Workspace bad : {Workspace{nullptr,8192u},Workspace{storage.get(),0u},Workspace{storage.get(),8193u}})
        if(prepare_workspace(&input,&input,&transposed,bad,nullptr)!=hipErrorInvalidValue || launches) return 1;
    if(prepare_workspace(nullptr,&input,&transposed,workspace,nullptr)!=hipErrorInvalidValue ||
       prepare_workspace(&input,nullptr,&transposed,workspace,nullptr)!=hipErrorInvalidValue ||
       prepare_workspace(&input,&input,nullptr,workspace,nullptr)!=hipErrorInvalidValue || launches) return 2;
    const uintptr_t base=reinterpret_cast<uintptr_t>(storage.get());
    for(unsigned tokens:{1u,17u,7169u,8192u}) {
        workspace.tokens=tokens;launches=fail=0u;
        if(prepare_workspace(&input,&input,&transposed,workspace,nullptr)!=hipSuccess || launches!=2u ||
           prepared_pointer[0]!=base || prepared_pointer[1]!=base+query_words*4u ||
           flag_pointer[0]!=base+(query_words+key_words)*4u ||
           flag_pointer[1]!=base+(query_words+key_words+query_flag_words)*4u ||
           grids[0]!=tokens*16u || grids[1]!=tokens*2u || transpose_pointer[0] ||
           transpose_pointer[1]!=reinterpret_cast<uintptr_t>(&transposed)) return 3;
        for(unsigned failure:{1u,2u}) {
            launches=0u;fail=failure;
            if(prepare_workspace(&input,&input,&transposed,workspace,nullptr)!=hipErrorUnknown || launches!=failure) return 4;
        }
    }
    launches=fail=0u;workspace.tokens=8192u;
    auto score=[&](unsigned start,unsigned count,unsigned stride,unsigned key_stride) {
        return launch_workspace(&workspace,&input,&transposed,&output,nullptr,start,count,stride,key_stride);
    };
    if(score(0,0,0,8192)!=hipErrorInvalidValue || score(0,129,129,8192)!=hipErrorInvalidValue ||
       score(8192,1,8193,8192)!=hipErrorInvalidValue || score(8191,2,8193,8192)!=hipErrorInvalidValue ||
       score(0,32,31,8192)!=hipErrorInvalidValue || score(0,32,32,8191)!=hipErrorInvalidValue || launches) return 5;
    if(launch_workspace(nullptr,&input,&transposed,&output,nullptr,0,32,32,8192)!=hipErrorInvalidValue ||
       launch_workspace(&workspace,nullptr,&transposed,&output,nullptr,0,32,32,8192)!=hipErrorInvalidValue ||
       launch_workspace(&workspace,&input,nullptr,&output,nullptr,0,32,32,8192)!=hipErrorInvalidValue ||
       launch_workspace(&workspace,&input,&transposed,nullptr,nullptr,0,32,32,8192)!=hipErrorInvalidValue || launches) return 6;
    for(unsigned count:{1u,17u,128u}) {
        launches=fail=0u;
        if(score(8192u-count,count,8192,8192)!=hipSuccess || launches!=1u || score_grid!=512u ||
           query_grid!=(count+15u)/16u || score_pointers[0]!=base || score_pointers[1]!=base+query_words*4u ||
           score_pointers[2]!=base+(query_words+key_words)*4u ||
           score_pointers[3]!=base+(query_words+key_words+query_flag_words)*4u ||
           score_pointers[4]!=reinterpret_cast<uintptr_t>(&output)) return 7;
        launches=0u;fail=1u;
        if(score(8192u-count,count,8192,8192)!=hipErrorUnknown || launches!=1u) return 8;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-prepared-qk-launch-') as tmp:
            exe=str(Path(tmp)/'launch')
            subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-Wall','-Wextra','-Werror','-x','c++','-','-o',exe],input=source,text=True,check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=10)

if __name__=='__main__':unittest.main()
