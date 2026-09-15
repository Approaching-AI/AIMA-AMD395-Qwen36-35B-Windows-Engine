"""Exercise actual slab ordering, reducer offsets and failed submissions."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class ParallelPvLaunchTests(unittest.TestCase):
    def test_deferred_workspace_reuse_bounds_and_failure_order(self):
        text = (ROOT / 'native/providers/ck_fmha/parallel_pv_partials.h').read_text()
        source = r'''
#include "parallel_pv_plan.h"
#include "../moe_accumulator/sm121_pv_final_bound.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>
#define __global__
#define hipLaunchKernelGGL(...) submit(__VA_ARGS__)
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
dim3 blockIdx,blockDim,threadIdx;
using hipStream_t=void*;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorUnknown=2;
namespace qrt_sm121_attention_rcp { float evaluate(const unsigned char* t,float denominator){assert(t&&denominator==1.0f);return 1.0f;} }
namespace qrt_parallel_pv {
namespace bound=qrt_sm121_pv_bound;
namespace deferred=qrt_sm121_pv_final_bound;
constexpr int produce=17;
unsigned calls=0,fail_at=0,completed_producers=0,completed_reducers=0,epoch=0;
std::vector<std::function<void()>> queued;
int hipGetLastError(){return calls==fail_at?hipErrorUnknown:hipSuccess;}
void submit(int kind,dim3 grid,dim3 block,unsigned shared,hipStream_t stream,
    const uint16_t* value,const uint16_t* p,Partial* workspace,unsigned start,
    unsigned offset,unsigned count,unsigned stride) {
    assert(kind==17&&grid.x==16&&grid.y==16&&grid.z==(groups(start+offset+count)+3)/4);
    assert(block.x==128&&!shared&&stream==reinterpret_cast<void*>(17)&&value&&p&&start+offset+count<=stride);
    if(++calls==fail_at)return;
    queued.push_back([=]{
        assert(completed_producers==completed_reducers);epoch=offset;
        for(unsigned g=0;g<groups(start+offset+count);++g)for(unsigned q=0;q<count;++q)
            for(unsigned feature=0;feature<4096;++feature) {
                const float term=float(start+offset+q+g+1)/4096.0f;
                workspace[(size_t(g)*16+q)*4096+feature]={term,term};
            }
        ++completed_producers;
    });
}
template<class Kernel>void submit(Kernel kernel,dim3 grid,dim3 block,unsigned shared,hipStream_t stream,
    const Partial* workspace,const float* scales,float* output,float* errors,
    unsigned start,unsigned offset,unsigned count,unsigned output_start,unsigned stride,
    const unsigned char* table,float* accumulator,float* denominator) {
    assert(grid.x==count*16&&!shared&&block.x==256&&stream==reinterpret_cast<void*>(17));
    if(++calls==fail_at)return;
    queued.push_back([=]{
        assert(completed_producers==completed_reducers+1&&epoch==offset);blockDim=block;
        for(unsigned b=0;b<grid.x;++b)for(unsigned t=0;t<block.x;++t){blockIdx.x=b;threadIdx.x=t;
            kernel(workspace,scales,output,errors,start,offset,count,output_start,stride,table,accumulator,denominator);}
        ++completed_reducers;
    });
}
'''
        source += function(text, '__global__ void reduce(')
        source += '\n' + function(text, 'inline int launch(')
        source += r'''
constexpr int shared_kernel=18;
void submit(int kind,dim3 grid,dim3 block,unsigned shared,hipStream_t stream,
    const uint16_t* v,const uint16_t* p,const float* scales,float* output,float* errors,
    unsigned start,unsigned count,unsigned offset,unsigned stride,const unsigned char* table,float*,float*) {
    assert(kind==18&&grid.x==16&&grid.y==16&&grid.z==(count+15)/16&&block.x==128&&!shared);
    assert(stream==reinterpret_cast<void*>(17)&&v&&p&&scales&&output&&errors&&table);
    assert(valid(start,count,offset,stride));++calls;
}
'''
        shared = (ROOT / 'native/providers/ck_fmha/shared_parallel_pv.h').read_text()
        source += '\n' + function(shared, 'inline int launch_shared(')
        source += r'''
void run(unsigned start,unsigned count,unsigned failure) {
    constexpr unsigned guard=19,output_start=3;
    const unsigned stride=start+count,tiles=(stride+31)/32,cells=count*4096;
    const size_t capacity=partial_count(start,count);
    std::vector<Partial> partials(capacity+2*guard,Partial{-12345.0f,-12345.0f});
    const size_t output_words=size_t(output_start+count)*4096+2*guard,den_words=size_t(output_start+count)*16+2*guard;
    std::vector<float> scales(size_t(count)*16*(tiles+1)+2*guard,1.0f),out(output_words,-12345),acc(out),den(den_words,-12345),error(cells+2*guard,-12345);
    uint16_t input=0;unsigned char table=0;
    calls=completed_producers=completed_reducers=0;fail_at=failure;queued.clear();
    const int status=launch(&input,&input,scales.data()+guard,out.data()+guard,error.data()+guard,
        start,count,output_start,stride,&table,partials.data()+guard,capacity,reinterpret_cast<void*>(17),acc.data()+guard,den.data()+guard);
    const unsigned total_calls=2*((count+15)/16);
    assert(status==(failure?hipErrorUnknown:hipSuccess)&&calls==(failure?failure:total_calls));
    // Submission is asynchronous: actual output remains untouched until the
    // owner drains the stream, even when a later submission failed.
    assert(std::all_of(out.begin(),out.end(),[](float x){return x==-12345;}));
    for(auto& operation:queued)operation();queued.clear();
    const unsigned done=failure?(failure-1)/2:total_calls/2,done_queries=std::min(count,done*16);
    assert(completed_reducers==done);
    for(size_t i=0;i<out.size();++i){
        const size_t begin=guard+output_start*4096;
        const bool live=i>=begin&&i<begin+size_t(done_queries)*4096;
        if(!live){assert(out[i]==-12345&&acc[i]==-12345);continue;}
        const unsigned query=unsigned((i-begin)/4096),g=groups(start+query+1);
        // Closed-form sum of the exact integer test terms; independent of
        // the reducer's loops, workspace indexing and launch partition.
        const double expected=(double(g)*(start+query+1)+double(g)*(g-1)/2)/4096.0;
        assert(double(out[i])==expected&&double(acc[i])==expected);
    }
    for(size_t i=0;i<den.size();++i){const size_t begin=guard+output_start*16;
        assert(den[i]==(i>=begin&&i<begin+done_queries*16?1.0f:-12345.0f));}
    for(size_t i=0;i<error.size();++i){
        if(i>=guard&&i<guard+size_t(done_queries)*4096)assert(std::isfinite(error[i])&&error[i]>0.0f);
        else assert(error[i]==-12345);}
    for(size_t i=0;i<guard;++i)assert(partials[i].value==-12345&&partials[i].absolute==-12345&&
        partials[guard+capacity+i].value==-12345&&partials[guard+capacity+i].absolute==-12345);
    assert(std::all_of(scales.begin(),scales.end(),[](float x){return x==1.0f;}));
}
} // namespace qrt_parallel_pv
int main(){using namespace qrt_parallel_pv;
    run(0,33,0);run(63,128,0);run(8175,17,0);run(8191,1,0);
    for(unsigned failure=1;failure<=16;++failure)run(63,128,failure);
    uint16_t p=0;float s=1,out=0,error=0;unsigned char t=0;Partial part{};
    auto invoke=[&](unsigned start,unsigned count,unsigned output,unsigned stride,size_t capacity){
        calls=0;return launch(&p,&p,&s,&out,&error,start,count,output,stride,&t,&part,capacity,reinterpret_cast<void*>(17));};
    assert(invoke(0,0,0,1,size_t(-1))==1&&calls==0);assert(invoke(0,129,0,256,size_t(-1))==1&&calls==0);
    assert(invoke(8192,1,0,8192,size_t(-1))==1&&calls==0);assert(invoke(8191,2,0,8192,size_t(-1))==1&&calls==0);
    assert(invoke(0,1,8192,1,size_t(-1))==1&&calls==0);assert(invoke(0,1,0,8193,size_t(-1))==1&&calls==0);
    assert(invoke(~0u,1,0,8192,size_t(-1))==1&&calls==0);assert(invoke(0,1,0,1,partial_count(0,1)-1)==1&&calls==0);
    for(unsigned missing=0;missing<7;++missing){calls=0;
        assert(launch(missing==0?nullptr:&p,missing==1?nullptr:&p,missing==2?nullptr:&s,
            missing==3?nullptr:&out,missing==4?nullptr:&error,0,1,0,1,missing==5?nullptr:&t,
            missing==6?nullptr:&part,size_t(-1),reinterpret_cast<void*>(17))==1&&calls==0);}
    fail_at=0;calls=0;
    assert(launch_shared(&p,&p,&s,&out,&error,8064,128,0,8192,&t,reinterpret_cast<void*>(17))==0&&calls==1);
    calls=0;fail_at=1;
    assert(launch_shared(&p,&p,&s,&out,&error,16,17,3,33,&t,reinterpret_cast<void*>(17))==2&&calls==1);
    for(unsigned count:{0u,129u,~0u}){calls=0;
        assert(launch_shared(&p,&p,&s,&out,&error,0,count,0,8192,&t,reinterpret_cast<void*>(17))==1&&calls==0);}
    calls=0;assert(launch_shared(&p,&p,&s,&out,&error,8176,17,0,8192,&t,reinterpret_cast<void*>(17))==1&&calls==0);
    assert(launch_shared(&p,&p,&s,&out,&error,0,1,8192,8192,&t,reinterpret_cast<void*>(17))==1&&calls==0);
    assert(launch_shared(&p,&p,&s,&out,&error,0,1,0,8193,&t,reinterpret_cast<void*>(17))==1&&calls==0);
    for(unsigned missing=0;missing<6;++missing){calls=0;
        assert(launch_shared(missing==0?nullptr:&p,missing==1?nullptr:&p,missing==2?nullptr:&s,
            missing==3?nullptr:&out,missing==4?nullptr:&error,0,1,0,1,missing==5?nullptr:&t,
            reinterpret_cast<void*>(17))==1&&calls==0);}
    std::puts("parallel_pv_deferred_api=pass actual_reducer=pass failed_submissions=16 invalid_arguments=15 native_wmma_checked=0");
    std::puts("parallel_pv_shared_launch=pass injected_launch_failure=1 invalid_arguments=12 native_wmma_checked=0");
}
'''
        compiler = shutil.which('clang++') or shutil.which('c++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory(prefix='qrt-parallel-pv-launch-') as temp:
            path = Path(temp)
            (path / 'check.cpp').write_text(source)
            subprocess.run([compiler, '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-Wno-unknown-pragmas', '-fsanitize=address,undefined', '-ffp-contract=off',
                            '-I', str(ROOT / 'native/providers/ck_fmha'), str(path / 'check.cpp'),
                            '-o', str(path / 'check')], check=True, timeout=60)
            subprocess.run([str(path / 'check')], check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
