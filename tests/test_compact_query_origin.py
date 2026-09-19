"""Check compact original-Q views in classification, dispatch and exact replay."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class CompactQueryOriginTests(unittest.TestCase):
    def test_actual_fallback_and_long_dispatch(self):
        providers = ROOT/'native/providers'
        fallback = (providers/'ck_fmha/deferred_qk_fallback.h').read_text()
        actual_replay = '\n'.join(function(fallback, signature) for signature in (
            '__device__ __forceinline__ void replay_cell(',
            '__global__ void replay_scan(', '__global__ void replay_scan_from_query_origin('))
        raw_dot = function((providers/'ck_fmha/decoded_window_qk.h').read_text(),
                           '__device__ __attribute__((noinline)) float raw_dot(')
        normalize = function((providers/'moe_accumulator/sm121_wave16.h').read_text(),
                             '__device__ __forceinline__ qrt_q1_moe_hawkeye::Value\nnormalize(')
        long_header = (providers/'ck_fmha/long_narrow_qk.h').read_text()
        actual_long = function(long_header, 'struct Workspace')+';\n'+ '\n'.join(
            function(long_header, signature) for signature in (
                'inline bool valid(', 'inline int prepare_domain(', 'inline int launch_workspace('))
        code = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <tuple>
#include <vector>
#include "native/providers/moe_accumulator/sm121_group16_modulo.h"
#include "native/providers/moe_accumulator/sm121_canonical_normalize.h"
#include "native/providers/ck_fmha/prepared_decoded_qk_range_workspace.h"
#define __device__
#define __global__
#define __forceinline__ inline
unsigned __clz(unsigned v){return unsigned(__builtin_clz(v));}
struct dim3{unsigned x,y,z;dim3(unsigned a=0,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
dim3 threadIdx,blockIdx,blockDim;
using hipStream_t=void*;
enum hipError_t{hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
namespace qrt_blackwell_attention{
constexpr unsigned kQueryHeads=16u,kKvHeads=2u,kHeadDim=256u;
constexpr int kBlackwellZeroExponent=-133;
constexpr float kExactScale=0.0625f;
}
namespace qrt_sm121_f32_carry{uint32_t bits(float f){uint32_t x;std::memcpy(&x,&f,4);return x;}}
namespace qrt_sm121_wave16{
constexpr int16_t kZeroExponent=-133;
''' + normalize + r'''
}
namespace qrt_decoded_window_qk{
''' + raw_dot + r'''
}
namespace qrt_deferred_qk_fallback{
constexpr uint32_t deferred_bits=0x7ffffffeu;
''' + actual_replay + r'''
}
namespace qrt_narrow_domain_qk{void classify_rows(){}}
unsigned launches=0,fail_launch=0,zeroes=0,fail_zero=0;
unsigned observed_origin=0,observed_start=0,observed_count=0;
const uint16_t* observed_queries[3]{};
hipError_t hipMemsetAsync(void*,int,size_t bytes,hipStream_t){assert(bytes==8u);++zeroes;return fail_zero?hipErrorUnknown:hipSuccess;}
hipError_t hipGetLastError(){return launches==fail_launch?hipErrorUnknown:hipSuccess;}
template<class Kernel,class... Args>void launch(Kernel,dim3,dim3,unsigned,hipStream_t,Args... args){
    const auto tuple=std::make_tuple(args...);
    if constexpr(sizeof...(Args)==3u){assert(launches<2u);observed_queries[launches]=std::get<0>(tuple);}
    else if constexpr(sizeof...(Args)==13u){observed_origin=std::get<12>(tuple);}
    else{static_assert(sizeof...(Args)==8u);observed_queries[2]=std::get<0>(tuple);
        observed_start=std::get<3>(tuple);observed_count=std::get<4>(tuple);observed_origin=std::get<7>(tuple);}
    ++launches;
}
#define hipLaunchKernelGGL(kernel,...) launch(kernel,__VA_ARGS__)
namespace qrt_long_narrow_qk{
template<bool,unsigned,unsigned,unsigned>void scores(){}
''' + actual_long + r'''
}
float marker(){const uint32_t x=qrt_deferred_qk_fallback::deferred_bits;float f;std::memcpy(&f,&x,4);return f;}
uint16_t value(unsigned row,unsigned feature){
    constexpr unsigned exponents[]={0u,83u,84u,95u,117u,127u,141u,159u,174u};
    const unsigned cell=row*17u+feature*11u;
    return uint16_t(((cell%3u)==0u?0x8000u:0u)|(exponents[cell%9u]<<7u)|(cell%127u));
}
int main(){
    unsigned checked=0;
    for(unsigned origin:{17u,8192u,131072u,262144u}){
        constexpr unsigned input_count=5u,skip=2u,count=2u,heads=16u,width=256u;
        const unsigned start=origin+skip,stride=start+count,keys=origin+input_count;
        std::vector<uint16_t> query(input_count*heads*width);
        for(unsigned row=0;row<input_count*heads;++row)for(unsigned f=0;f<width;++f)
            query[row*width+f]=value(row,f);
        const auto saved=query;
        // Allocate only actual query rows. The large historical positions are
        // logical labels; reading them as compact offsets trips ASan.
        std::unique_ptr<uint16_t[]> transposed(new uint16_t[size_t(keys)*512u]);
        std::unique_ptr<float[]> output(new float[size_t(count)*heads*stride+1u]);
        for(unsigned kv=0;kv<2u;++kv)for(unsigned key:{0u,13u,start})
            for(unsigned f=0;f<width;++f)transposed[(size_t(kv)*width+f)*keys+key]=value(kv+key+1u,f+3u);
        for(unsigned row=0;row<count;++row)for(unsigned head=0;head<heads;++head)
            for(unsigned key:{0u,13u,start}){
                const size_t cell=(size_t(row)*heads+head)*stride+key;
                uint16_t right[width];
                for(unsigned f=0;f<width;++f)right[f]=transposed[(size_t(head/8u)*width+f)*keys+key];
                const float want=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(
                    0.0f,query.data()+((row+skip)*heads+head)*width,right,width)*0.0625f;
                output[cell]=marker();blockDim.x=256u;blockIdx.x=unsigned(cell/256u);threadIdx.x=unsigned(cell%256u);
                qrt_deferred_qk_fallback::replay_scan_from_query_origin(query.data(),transposed.get(),output.get(),
                    start,count,stride,keys,origin);
                assert(qrt_sm121_f32_carry::bits(output[cell])==qrt_sm121_f32_carry::bits(want));++checked;
                // A successful score must not be touched even with null
                // original inputs; replay acts only on its reserved marker.
                qrt_deferred_qk_fallback::replay_cell(nullptr,nullptr,output.get(),cell,start,stride,keys,origin);
                assert(qrt_sm121_f32_carry::bits(output[cell])==qrt_sm121_f32_carry::bits(want));
            }
        const size_t end=size_t(count)*heads*stride;output[end]=123.0f;
        blockIdx.x=unsigned(end/256u);threadIdx.x=unsigned(end%256u);
        qrt_deferred_qk_fallback::replay_scan_from_query_origin(nullptr,nullptr,output.get(),start,count,stride,keys,origin);
        assert(output[end]==123.0f&&query==saved);
    }
    // Exercise original origin-zero callers with the unchanged entry point.
    std::vector<uint16_t> q(3u*4096u),key(3u*512u),right(256u);
    for(unsigned i=0;i<q.size();++i)q[i]=value(i/256u,i%256u);
    for(unsigned i=0;i<key.size();++i)key[i]=value(i/3u,i%3u);
    std::vector<float> out(16u*3u,marker());blockDim.x=256u;blockIdx.x=0u;
    for(unsigned head=0;head<16u;++head){
        threadIdx.x=head*3u;
        qrt_deferred_qk_fallback::replay_scan(q.data(),key.data(),out.data(),2u,1u,3u,3u);
        for(unsigned f=0;f<256u;++f)right[f]=key[(head/8u*256u+f)*3u];
        const float want=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,q.data()+(2u*16u+head)*256u,right.data(),256u)*0.0625f;
        assert(qrt_sm121_f32_carry::bits(out[head*3u])==qrt_sm121_f32_carry::bits(want));++checked;
    }
    namespace range=qrt_prepared_decoded_qk_range;
    namespace longq=qrt_long_narrow_qk;
    uint32_t fake=0;unsigned domain[2]{};float result=0;
    range::Workspace d{&fake,range::workspace_words(16384u),16384u,8192u,129u,8321u};
    longq::Workspace w{d,domain,domain,domain,8192u};
    assert(longq::valid(w));
    for(unsigned origin:{0u,8192u}){
        w.query_origin=origin;launches=zeroes=0;
        std::unique_ptr<uint16_t[]> original(new uint16_t[size_t(8321u-origin)*4096u]);
        assert(longq::prepare_domain(original.get(),key.data(),w,nullptr)==hipSuccess&&launches==2u&&zeroes==1u);
        assert(observed_queries[0]==original.get()+size_t(8192u-origin)*4096u&&observed_queries[1]==key.data());
        for(unsigned failure:{1u,2u}){
            launches=zeroes=0;fail_launch=failure;
            assert(longq::prepare_domain(original.get(),key.data(),w,nullptr)==hipErrorUnknown&&launches==failure);
        }
        fail_launch=0;fail_zero=1;launches=zeroes=0;
        assert(longq::prepare_domain(original.get(),key.data(),w,nullptr)==hipErrorUnknown&&!launches);fail_zero=0;
        for(unsigned failure:{0u,1u,2u,3u}){
            launches=0;fail_launch=failure;
            assert(longq::launch_workspace(&w,original.get(),key.data(),&result,nullptr,8320u,1u,8321u,8321u)==
                (failure?hipErrorUnknown:hipSuccess));
            assert(launches==(failure?failure:3u));
            if(!failure)assert(observed_origin==origin&&observed_start==8320u&&observed_count==1u&&observed_queries[2]==original.get());
        }
    }
    launches=zeroes=fail_launch=0;w.query_origin=8193u;
    assert(!longq::valid(w));
    assert(longq::prepare_domain(q.data(),key.data(),w,nullptr)==hipErrorInvalidValue);
    assert(longq::launch_workspace(&w,q.data(),key.data(),&result,nullptr,8192u,1u,8193u,8321u)==hipErrorInvalidValue);
    assert(!launches&&!zeroes);
    std::printf("compact_query_original_dot_checks=%u dispatch_origins=2 invalid_origin_rejected=1\n",checked);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-compact-query-') as temporary:
            cpp=Path(temporary)/'check.cpp';cpp.write_text(code)
            exe=Path(temporary)/'check'
            subprocess.run(['c++','-std=c++17','-O1','-Wall','-Wextra','-Werror',
                            '-fsanitize=address,undefined','-fno-sanitize-recover=all',
                            '-DQRT_SM121_COMPACT_NORMALIZE=1','-I',str(ROOT),str(cpp),'-o',str(exe)],
                           check=True,timeout=30)
            subprocess.run([str(exe)],check=True,timeout=30)


if __name__=='__main__':
    unittest.main()
