"""Actual isolated orchestration: owned offsets, stage order and failed work."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def declaration(text, name):
    first = text.rfind("template<", 0, text.index(name))
    return text[first:text.index("{", first)]


class InplaceProbabilityPipelineTests(unittest.TestCase):
    def test_original_functions_stop_on_each_failed_stage(self):
        ck = ROOT / "native/providers/ck_fmha"
        producer = declaration((ck / "streamed_exact_attention.h").read_text(), "__global__ void produce(")
        replay = declaration((ck / "blackwell_attention.h").read_text(), "__global__ void blackwell_compacted_pv_replay_kernel(")
        owner = (ck / "inplace_probability_pipeline.h").read_text()
        owner = owner[owner.index("namespace qrt_inplace_probability_pipeline"):]
        old_replay = (ck / "long_fused_probability_pv.h").read_text()
        old_replay = old_replay[old_replay.index("template<bool InplaceProbability"):old_replay.index("inline int launch(")]
        code = r'''
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>
#include "native/providers/ck_fmha/inplace_probability_storage.h"
#define __global__
using hipStream_t=void*;
enum hipError_t:int{hipSuccess=0,hipErrorInvalidValue=1};
struct dim3{unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
constexpr unsigned start0=8192u,count0=33u,stride0=start0+count0,output0=3u;
const auto layout0=qrt_inplace_probability_storage::layout(count0,stride0);
float* memory=nullptr;float* output_memory=nullptr;bool expected_final=false;
const uint16_t operand[1]{};const unsigned char table[1]{},packed_table[1]{};
unsigned kernels=0,observed=0,memsets=0,qks=0;
unsigned fail_kernel=0,fail_observer=99u;bool fail_qk=false,fail_memset=false;
hipError_t hipGetLastError(){return hipError_t(kernels==fail_kernel?71:0);}
hipError_t hipMemsetAsync(void* p,int value,size_t bytes,hipStream_t){
    assert(p==reinterpret_cast<unsigned*>(memory+layout0.count)&&value==0&&bytes==4u);
    ++memsets;return hipError_t(fail_memset?72:0);
}
template<class F,class...A>void invoke(F f,dim3 grid,dim3 block,unsigned shared,hipStream_t s,A...a){
    assert(block.x==256u&&!shared&&!s);assert(grid.x);++kernels;f(a...);
}
#define hipLaunchKernelGGL(kernel,...) invoke(kernel,__VA_ARGS__)
namespace qrt_native_exp2_workspace{struct Workspace{unsigned char* packed;const unsigned char* original;};}
namespace qrt_sm121_pv_long_final_bound{struct Finalizer{};}
namespace qrt_streamed_exact_attention{struct NativeExp{};struct ShortFinalizer{};
''' + producer + r'''{
    static_assert(!FuseQk&&InplaceProbability);
    assert(FinalBound==expected_final&&query==nullptr&&transposed_key==nullptr&&value==operand);
    assert(source_scores==memory&&reinterpret_cast<float*>(probabilities)==memory);
    assert(scales==memory+layout0.scales&&errors==memory+layout0.errors);
    assert(output==output_memory+output0*4096u&&start==start0&&count==count0);
    assert(stride==stride0&&key_stride==stride0&&exp2_table==table&&packed_exp==packed_table&&rcp_table==table&&vllm_sum);
    assert(!raw_accumulator&&!raw_denominator&&!diagnostic_scores);
}
}
namespace qrt_blackwell_attention{
constexpr unsigned kSplitMaxTokens=264736u;
struct SplitCompletionObserver{void* state;int(*callback)(void*,unsigned,hipStream_t);};
int observe_split_stage(SplitCompletionObserver* p,unsigned stage,hipStream_t s){return p?p->callback(p->state,stage,s):0;}
void blackwell_collect_pv_replay_kernel(const float* output,const float* errors,unsigned start,unsigned cells,unsigned* indices,unsigned* selected){
    assert(output==output_memory&&errors==memory+layout0.errors&&start==output0&&cells==count0*4096u);
    assert(indices==reinterpret_cast<unsigned*>(memory+layout0.indices)&&selected==reinterpret_cast<unsigned*>(memory+layout0.count));
}
''' + replay + r'''{
    static_assert(TransposedValue&&!AllCells&&InplaceProbability);
    assert(RegisterRescale&&value==operand&&transposed_value==operand&&value_stride==stride0);
    assert(reinterpret_cast<const float*>(probabilities)==memory&&scales==memory+layout0.scales);
    assert(output==output_memory&&query_start==start0&&output_start==output0&&score_stride==stride0&&rcp_table==table);
    assert(indices==reinterpret_cast<unsigned*>(memory+layout0.indices)&&count==reinterpret_cast<unsigned*>(memory+layout0.count));
    assert(!raw_accumulator&&!raw_denominator&&!all_cells);
}
}
namespace qrt_long_narrow_qk{struct Workspace{};
int launch_workspace(const void* w,const uint16_t* q,const uint16_t* k,float* out,hipStream_t s,unsigned start,unsigned count,unsigned stride,unsigned keys){
    assert(w&&q==operand&&k==operand&&out==memory&&!s&&start==start0&&count==count0&&stride==stride0&&keys==stride0);
    ++qks;return fail_qk?73:0;
}}
namespace qrt_long_fused_probability_pv{
''' + old_replay + '\n}\n' + owner
        code += r'''
int main(){
    std::vector<float> arena(layout0.elements+16u,-37.25f),outputs((count0+output0+2u)*4096u,-19.5f);
    memory=arena.data()+8u;output_memory=outputs.data();
    qrt_long_narrow_qk::Workspace qk;
    qrt_native_exp2_workspace::Workspace exp{const_cast<unsigned char*>(packed_table),table};
    qrt_blackwell_attention::SplitCompletionObserver observer{nullptr,[](void*,unsigned stage,hipStream_t s){
        assert(!s&&stage==observed);++observed;return stage==fail_observer?74:0;
    }};
    auto reset=[](){kernels=observed=memsets=qks=0;fail_kernel=0;fail_observer=99u;fail_qk=fail_memset=false;};
    auto call=[&](size_t extent){return qrt_inplace_probability_pipeline::launch(qk,exp,
        operand,operand,operand,operand,output_memory,start0,count0,output0,stride0,table,table,
        memory,extent,nullptr,&observer,expected_final);};
    unsigned cases=0;
    for(bool final:{false,true}){
        expected_final=final;reset();assert(call(layout0.elements)==0);
        assert(qks==1&&kernels==3&&memsets==1&&observed==5);++cases;
        reset();assert(call(layout0.elements-1u)==1&&!qks&&!kernels&&!memsets&&!observed);++cases;
        reset();fail_qk=true;assert(call(layout0.elements)==73&&qks==1&&!kernels&&!memsets&&!observed);++cases;
        for(unsigned stage=0;stage<5u;++stage){
            reset();fail_observer=stage;assert(call(layout0.elements)==74&&observed==stage+1u);
            const unsigned expected_kernels=stage==0u?0u:stage<=2u?1u:stage==3u?2u:3u;
            assert(kernels==expected_kernels&&memsets==(stage>=3u?1u:0u));++cases;
        }
        for(unsigned step=1;step<=3u;++step){
            reset();fail_kernel=step;assert(call(layout0.elements)==71&&kernels==step);
            assert(observed==(step==1u?1u:step==2u?3u:4u));++cases;
        }
        reset();fail_memset=true;assert(call(layout0.elements)==72&&kernels==1&&observed==3&&memsets==1);++cases;
        for(unsigned invalid=0;invalid<12u;++invalid){
            reset();auto owner=exp;
            if(invalid==8u)owner.packed=nullptr;if(invalid==9u)owner.original=nullptr;
            const int status=qrt_inplace_probability_pipeline::launch(qk,owner,
                invalid==0u?nullptr:operand,invalid==1u?nullptr:operand,invalid==2u?nullptr:operand,
                invalid==3u?nullptr:operand,invalid==4u?nullptr:output_memory,
                start0,invalid==5u?129u:count0,output0,invalid==6u?stride0-1u:stride0,
                invalid==10u?nullptr:table,invalid==11u?nullptr:table,
                invalid==7u?nullptr:memory,layout0.elements,nullptr,&observer,final);
            assert(status==1&&!qks&&!kernels&&!memsets&&!observed);++cases;
        }
    }
    for(float x:arena)assert(x==-37.25f);
    for(float x:outputs)assert(x==-19.5f);
    std::printf("{\"pipeline_cases\":%u,\"actual_layout_offsets\":true,\"failed_steps_stop_submission\":true,\"gpu_execution\":false}\n",cases);
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-inplace-pipeline-") as tmp:
            cpp = Path(tmp) / "pipeline.cpp"
            cpp.write_text(code)
            exe = Path(tmp) / "pipeline"
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O1",
                            "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
                            "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-I", str(ROOT),
                            str(cpp), "-o", str(exe)], check=True, timeout=40)
            subprocess.run([str(exe)], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
