"""Check the actual experimental QK launcher span and partial-submit ordering."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class SelectiveQkLaunchTests(unittest.TestCase):
    def test_bounds_and_failed_dependency_do_not_submit_consumers(self):
        header = (ROOT / 'native/providers/ck_fmha/selective_qk.h').read_text()
        blackwell = (ROOT / 'native/providers/ck_fmha/blackwell_attention.h').read_text()
        actual = '\n'.join(function(header, name) for name in (
            'inline size_t probability_scratch_elements(', 'inline int launch_probability_attention('))
        old_span = '\n'.join(function(blackwell, name) for name in (
            'constexpr bool split_separate_probability(', 'constexpr unsigned split_query_limit(',
            'inline size_t split_scratch_elements('))
        source = r'''
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
using hipStream_t = void*;
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
constexpr unsigned kQueryHeads=16u, kHeadDim=256u, kThreads=256u, kIntegerMatrixColumns=128u;
constexpr unsigned kSplitMaxTokens=65536u, kExactTileTokens=32u;
struct dim3 { unsigned x,y,z; explicit dim3(unsigned a,unsigned b=1u,unsigned c=1u):x(a),y(b),z(c){} };
void native_scores(){} void collect_maxima(){} void collect_probabilities(){}
void repair_scores(){} void probabilities_and_denominators(){}
template<bool A,bool B,bool C,bool D=false,bool E=false> void blackwell_mantissa_value_kernel(){}
unsigned launches=0u,memsets=0u,replays=0u,fail_kernel=0u,fail_memset=0u;
bool fail_replay=false;
const char* names[8]{};
hipError_t hipGetLastError(){return fail_kernel && launches==fail_kernel ? hipErrorUnknown : hipSuccess;}
hipError_t hipMemsetAsync(void*,int,size_t bytes,hipStream_t){
    ++memsets;return bytes!=4u || memsets==fail_memset ? hipErrorUnknown : hipSuccess;
}
template<class Kernel,class... Args> void record(const char* name,Kernel,dim3,dim3,Args...){names[launches++]=name;}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(#kernel,kernel,__VA_ARGS__)
template<class... Args> int launch_compacted_pv_replay(Args...){++replays;return fail_replay ? hipErrorUnknown : hipSuccess;}
''' + old_span + actual + r'''
void reset(){launches=memsets=replays=fail_kernel=fail_memset=0u;fail_replay=false;}
int main(){
    uint16_t input=0;unsigned char table=0;
    // Backing arrays are large enough for every checked pointer offset; mock
    // kernels do not dereference model/output pointers.
    const size_t base=split_scratch_elements(128u,8192u,22u), extra=probability_scratch_elements(128u,8192u);
    auto* scratch=new float[base];auto* work=new float[extra];float output=0;
    auto launch=[&](unsigned start=8064u,unsigned queries=128u,unsigned layout=22u,size_t b=0u,size_t e=0u,
        const uint16_t* key=nullptr,unsigned key_stride=8192u,const uint16_t* value=nullptr,unsigned value_stride=0u,
        bool final=true,bool direct=true,const unsigned char* exp=nullptr,unsigned output_start=3u){
        return launch_probability_attention(&input,&input,&input,&output,nullptr,start,queries,output_start,
            exp ? exp : &table,&table,layout,scratch,b ? b : base,work,e ? e : extra,
            key ? key : &input,key_stride,value,value_stride,final,direct);
    };
    if(extra*4u!=136314884u || probability_scratch_elements(0u,1u) || probability_scratch_elements(129u,8192u) ||
       probability_scratch_elements(2u,1u) || probability_scratch_elements(1u,8193u) ||
       probability_scratch_elements(1u,0xffffffffu))return 1;
    for(unsigned layout:{22u,24u})for(bool final:{false,true})for(bool direct:{false,true}){
        reset();if(launch(8064u,128u,layout,base,extra,&input,8192u,&input,8192u,final,direct)!=hipSuccess ||
            launches!=7u || memsets!=2u || replays!=1u || std::strcmp(names[0],"native_scores") ||
            std::strcmp(names[5],"probabilities_and_denominators"))return 2;
    }
    for(unsigned failure=1u;failure<=7u;++failure){
        reset();fail_kernel=failure;
        if(launch()!=hipErrorUnknown || launches!=failure || replays)return 3;
    }
    for(unsigned failure:{1u,2u}){
        reset();fail_memset=failure;
        if(launch()!=hipErrorUnknown || memsets!=failure || launches!=(failure==1u?1u:3u) || replays)return 4;
    }
    reset();fail_replay=true;if(launch()!=hipErrorUnknown || replays!=1u)return 5;
    reset();
    if(launch(8064u,128u,22u,base-1u)!=hipErrorInvalidValue || launches || memsets || replays)return 6;
    if(launch(8064u,128u,22u,base,extra-1u)!=hipErrorInvalidValue || launches || memsets || replays)return 7;
    for(unsigned bad:{0u,1u,13u,23u,25u,0xffffffffu}){
        if(launch(8064u,128u,bad)!=hipErrorInvalidValue || launches)return 8;
    }
    if(launch(8192u,1u)!=hipErrorInvalidValue || launch(8064u,129u)!=hipErrorInvalidValue ||
       launch(0u,0u)!=hipErrorInvalidValue || launch(0xffffffffu,1u)!=hipErrorInvalidValue || launches)return 9;
    if(launch(8064u,128u,22u,base,extra,&input,8191u)!=hipErrorInvalidValue ||
       launch(8064u,128u,22u,base,extra,&input,8193u)!=hipErrorInvalidValue ||
       launch(8064u,128u,22u,base,extra,&input,8192u,&input,8191u)!=hipErrorInvalidValue || launches)return 10;
    if(launch(8064u,128u,22u,base,extra,&input,8192u,nullptr,8192u)!=hipErrorInvalidValue ||
       launch(8064u,128u,22u,base,extra,&input,8192u,nullptr,0u,true,true,&table,262144u)!=hipErrorInvalidValue || launches)return 11;
    if(launch_probability_attention(&input,&input,&input,&output,nullptr,0u,1u,0u,
        nullptr,&table,22u,scratch,base,work,extra,&input,1u,nullptr,0u,true,true)!=hipErrorInvalidValue || launches)return 12;
    delete[] work;delete[] scratch;return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-selective-qk-policy-') as tmp:
            executable = str(Path(tmp) / 'policy')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-x', 'c++', '-', '-o', executable], input=source, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
