"""Actual range encoding/launch bounds and compact Q view ownership."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class PreparedDecodedQkRangeTests(unittest.TestCase):
    def test_query_origin_views_and_failed_submissions(self):
        header = (ROOT / 'native/providers/ck_fmha/prepared_decoded_qk_range.h').read_text()
        actual = '\n'.join(function(header, name) for name in
                           ('inline int prepare_workspace_from_query_origin(',
                            'inline int prepare_workspace(', 'inline int launch_workspace('))
        source = r'''
#include <cstdlib>
#include <memory>
#include <tuple>
#include <initializer_list>
#include "''' + str(ROOT / 'native/providers/ck_fmha/prepared_decoded_qk_range_workspace.h') + r'''"
using hipStream_t=void*;
enum hipError_t { hipSuccess,hipErrorInvalidValue,hipErrorUnknown };
struct dim3 { unsigned x,y,z; dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
namespace qrt_blackwell_attention { constexpr unsigned kQueryHeads=16,kKvHeads=2,kHeadDim=256,kThreads=256; }
namespace qrt_prepared_decoded_qk { template<bool Key> void prepare() {} }
unsigned launches=0,fail=0,grids[2]{},prepared_tokens[2]{},score_shape[5]{},score_grid=0,query_grid=0;
uintptr_t inputs[2]{},packed[2]{},flags[2]{},transpose[2]{},score_views[5]{};
template<class K,class... A> void launch(K,dim3 grid,dim3 threads,unsigned,hipStream_t,A... args) {
    if(threads.x!=256u)std::abort();const auto values=std::make_tuple(args...);
    if constexpr(sizeof...(A)==5u) {
        if(launches>=2u)std::abort();
        inputs[launches]=reinterpret_cast<uintptr_t>(std::get<0>(values));
        packed[launches]=reinterpret_cast<uintptr_t>(std::get<1>(values));
        flags[launches]=reinterpret_cast<uintptr_t>(std::get<2>(values));
        transpose[launches]=reinterpret_cast<uintptr_t>(std::get<3>(values));
        grids[launches]=grid.x;prepared_tokens[launches]=std::get<4>(values);
    } else {
        static_assert(sizeof...(A)==12u);
        score_views[0]=reinterpret_cast<uintptr_t>(std::get<2>(values));
        score_views[1]=reinterpret_cast<uintptr_t>(std::get<3>(values));
        score_views[2]=reinterpret_cast<uintptr_t>(std::get<4>(values));
        score_views[3]=reinterpret_cast<uintptr_t>(std::get<5>(values));
        score_views[4]=reinterpret_cast<uintptr_t>(std::get<6>(values));
        score_shape[0]=std::get<7>(values);score_shape[1]=std::get<8>(values);
        score_shape[2]=std::get<9>(values);score_shape[3]=std::get<10>(values);score_shape[4]=std::get<11>(values);
        score_grid=grid.x;query_grid=grid.z;if(grid.y!=16u)std::abort();
    }
    ++launches;
}
hipError_t hipGetLastError() { return launches==fail?hipErrorUnknown:hipSuccess; }
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) launch(kernel,__VA_ARGS__)
namespace qrt_prepared_decoded_qk_range {
void scores() {}
''' + actual + r'''
}
int main() {
    using namespace qrt_prepared_decoded_qk_range;
    static_assert(workspace_words(maximum_keys)*4u==679039232u);
    static_assert(!workspace_words(0u) && !workspace_words(maximum_keys+1u));
    const size_t words=workspace_words(256u);
    std::unique_ptr<uint32_t[]> storage(new uint32_t[words]);
    std::unique_ptr<uint16_t[]> query(new uint16_t[146u*16u*256u]);
    uint16_t key=0,transposed=0;float output=0;
    Workspace w{storage.get(),words,256u,17u,129u,146u};
    for(unsigned mutation=0;mutation<9u;++mutation) {
        auto bad=w;
        switch(mutation) {
        case 0:bad.words=nullptr;break;case 1:--bad.word_count;break;
        case 2:bad.key_capacity=0;break;case 3:bad.key_capacity=maximum_keys+1u;break;
        case 4:bad.query_count=0;break;case 5:bad.query_count=maximum_queries+1u;break;
        case 6:bad.query_start=UINT32_MAX;break;case 7:bad.key_tokens=0;break;case 8:bad.key_tokens=257u;break;
        }
        if(valid(bad) || prepare_workspace(query.get(),&key,&transposed,bad,nullptr)!=hipErrorInvalidValue || launches)return 1;
    }
    auto boundary=w;boundary.key_capacity=boundary.key_tokens=maximum_keys;
    boundary.word_count=workspace_words(maximum_keys);boundary.query_start=maximum_keys-1u;boundary.query_count=1u;
    if(!valid(boundary))return 2;++boundary.query_count;if(valid(boundary))return 3;
    const uintptr_t base=reinterpret_cast<uintptr_t>(storage.get());
    if(prepare_workspace(query.get(),&key,&transposed,w,nullptr)!=hipSuccess || launches!=2u ||
       inputs[0]!=reinterpret_cast<uintptr_t>(query.get()+17u*16u*256u) ||
       packed[0]!=base || packed[1]!=base+query_words*4u ||
       flags[0]!=base+(query_words+key_words(256u))*4u || flags[1]!=flags[0]+query_flag_words*4u ||
       grids[0]!=129u*16u || grids[1]!=146u*2u || prepared_tokens[0]!=129u || prepared_tokens[1]!=146u ||
       transpose[0] || transpose[1]!=reinterpret_cast<uintptr_t>(&transposed))return 4;
    for(unsigned failure:{1u,2u}) {
        launches=0;fail=failure;
        if(prepare_workspace(query.get(),&key,&transposed,w,nullptr)!=hipErrorUnknown || launches!=failure)return 5;
    }
    launches=fail=0;
    auto score=[&](unsigned start,unsigned count,unsigned stride,unsigned keys) {
        return launch_workspace(&w,query.get(),&transposed,&output,nullptr,start,count,stride,keys);
    };
    if(score(16,1,17,146)!=hipErrorInvalidValue || score(146,1,147,146)!=hipErrorInvalidValue ||
       score(17,129,146,146)!=hipErrorInvalidValue || score(145,2,147,146)!=hipErrorInvalidValue ||
       score(17,128,144,146)!=hipErrorInvalidValue || score(17,128,145,145)!=hipErrorInvalidValue || launches)return 6;
    if(score(17,128,145,146)!=hipSuccess || launches!=1u || score_grid!=10u || query_grid!=8u ||
       score_shape[0]!=17u || score_shape[1]!=128u || score_shape[2]!=145u || score_shape[3]!=146u || score_shape[4]!=17u ||
       score_views[0]!=base || score_views[1]!=packed[1] || score_views[2]!=flags[0] ||
       score_views[3]!=flags[1] || score_views[4]!=reinterpret_cast<uintptr_t>(&output))return 7;
    launches=0;if(score(145,1,146,146)!=hipSuccess || launches!=1u || score_shape[4]!=17u)return 8;
    launches=0;fail=1;if(score(17,128,145,146)!=hipErrorUnknown || launches!=1u)return 9;
    // The same absolute range can start at the first cell of a compact Q
    // allocation. Intermediate origins and both failed submissions retain
    // the original logical metadata and complete K preparation.
    std::unique_ptr<uint16_t[]> compact(new uint16_t[129u*16u*256u]);
    for(unsigned origin:{0u,5u,17u}){
        launches=fail=0;
        const auto* q=origin==17u?compact.get():query.get();
        if(prepare_workspace_from_query_origin(q,&key,&transposed,w,nullptr,origin)!=hipSuccess ||
           launches!=2u || inputs[0]!=reinterpret_cast<uintptr_t>(q+(17u-origin)*4096u) ||
           prepared_tokens[0]!=129u || prepared_tokens[1]!=146u)return 10;
    }
    for(unsigned failure:{1u,2u}){
        launches=0;fail=failure;
        if(prepare_workspace_from_query_origin(compact.get(),&key,&transposed,w,nullptr,17u)!=hipErrorUnknown ||
           launches!=failure)return 11;
    }
    launches=fail=0;
    for(unsigned origin:{18u,UINT32_MAX})
        if(prepare_workspace_from_query_origin(compact.get(),&key,&transposed,w,nullptr,origin)!=hipErrorInvalidValue || launches)return 12;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-prepared-range-') as tmp:
            cpp = Path(tmp) / 'range.cpp'; cpp.write_text(source)
            exe = Path(tmp) / 'range'
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-fno-omit-frame-pointer', str(cpp), '-o', str(exe)],
                           check=True, timeout=30)
            subprocess.run([str(exe)], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
