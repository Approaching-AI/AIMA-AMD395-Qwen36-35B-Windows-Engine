"""Execute the actual optional GDN launchers and checkpoint fallback."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class CoarseIntervalLaunchTests(unittest.TestCase):
    def test_selection_submissions_failures_and_checkpoint_ownership(self):
        source = (ROOT / "native/providers/gdn/blackwell_cooperative.cpp").read_text()
        wrappers = "\n".join(function(source, "hipError_t " + name + "(")
                             for name in ("wu", "output", "state", "state_checkpoints"))
        code = r'''
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include "native/providers/gdn/coarse_interval_policy.h"
#include "native/providers/gdn/paired_score_policy.h"
namespace qrt_fla_lifetime { void output_kernel(){} template<unsigned C>void state_kernel(){} }
#include "native/providers/gdn/fla_checkpoint.h"
enum hipError_t{hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3{unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
int scalar_mode=1,scalar_columns=8;
namespace qrt_fla_blackwell_scalar{
int mode(){return scalar_mode;}int state_columns(){return scalar_columns;}
void wu_kernel(){}void output_kernel(){}template<unsigned C>void state_kernel(){}
}
namespace qrt_fla_interval{
template<bool C,bool A>void wu_kernel(){}template<bool C,bool A>void output_kernel(){}template<bool C,bool A>void state_kernel(){}
}
constexpr unsigned threads=256u,tile_columns=8u,state_columns=4u;
void wu_kernel(){}void output_kernel(){}template<bool C>void state_kernel(){}
unsigned launches=0,queries=0,count_seen=0;bool fail=false,null_statistics=false;
dim3 grid_seen,block_seen;hipStream_t stream_seen=nullptr;void(*kernel_seen)()=nullptr;
template<class... Args>void record(void(*kernel)(),dim3 grid,dim3 block,unsigned shared,hipStream_t stream,Args... args){
    ++launches;assert(!shared);kernel_seen=kernel;grid_seen=grid;block_seen=block;stream_seen=stream;
    const auto values=std::make_tuple(args...);
    if constexpr(std::is_integral_v<std::tuple_element_t<6,decltype(values)>>)count_seen=std::get<6>(values);
    else count_seen=std::get<7>(values);
    null_statistics=false;
    if constexpr(std::is_same_v<std::remove_cv_t<std::tuple_element_t<sizeof...(Args)-1,decltype(values)>>,unsigned*>)
        null_statistics=std::get<sizeof...(Args)-1>(values)==nullptr;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
hipError_t hipGetLastError(){++queries;return fail?hipErrorUnknown:hipSuccess;}
''' + wrappers + r'''
int main(){
    unsetenv("QRT_FLA_GDN_PAIRED_SCORE_ARENAS");
    unsetenv("QRT_FLA_GDN_COARSE_INTERVAL");assert(qrt_fla_coarse_policy::mode()==0);
    unsigned checked=0u;
    for(const char* setting:{"","0","1","2","-1","01","1 ","true"}){
        setenv("QRT_FLA_GDN_COARSE_INTERVAL",setting,1);
        const int mode=!setting[0]||!std::strcmp(setting,"0")?0:!std::strcmp(setting,"1")?1:-1;
        assert(qrt_fla_coarse_policy::mode()==mode);
        for(bool scalar:{false,true})for(unsigned count:{0u,1u,1024u,1025u,UINT32_MAX})
            assert(qrt_fla_coarse_policy::selected(mode,scalar,count)==(mode==1&&scalar&&count&&count<=1024u));
        for(int selection:{-1,0,1,4,8})for(unsigned operation=0u;operation<3u;++operation)
        for(unsigned count:{1u,63u,64u,65u,1024u})for(bool failure:{false,true}){
            scalar_mode=selection==1?1:selection<0?-1:0;scalar_columns=selection==4||selection==8?selection:selection<0?-1:0;
            const bool active=mode==1&&(operation==2u?scalar_columns==8:scalar_mode==1);
            const bool invalid=mode<0||selection<0;fail=failure;launches=queries=0u;
            uint16_t words[16]{};float floats[16]{};unsigned char table[1]{};auto stream=reinterpret_cast<void*>(uintptr_t(0x395u));
            const hipError_t status=operation==0u?wu(words,words+1,words+2,words+3,floats,words+4,words+1,count,table,stream):
                operation==1u?output(words,words+1,words+2,floats,words+3,floats+1,count,table,stream):
                state(words,words+1,words+2,floats,words+3,words+4,floats+1,count,table,stream);
            ++checked;
            if(invalid){assert(status==hipErrorInvalidValue&&!launches&&!queries);continue;}
            assert(status==(failure?hipErrorUnknown:hipSuccess)&&launches==1u&&queries==1u);
            assert(count_seen==count&&stream_seen==stream&&block_seen.x==256u&&block_seen.y==1u&&block_seen.z==1u);
            void(*expected)()=nullptr;
            if(operation==0u)expected=active?qrt_fla_interval::wu_kernel<true,false>:scalar_mode?qrt_fla_blackwell_scalar::wu_kernel:wu_kernel;
            else if(operation==1u)expected=active?qrt_fla_interval::output_kernel<true,false>:scalar_mode?qrt_fla_blackwell_scalar::output_kernel:output_kernel;
            else expected=active?qrt_fla_interval::state_kernel<true,false>:scalar_columns==8?qrt_fla_blackwell_scalar::state_kernel<8>:scalar_columns==4?qrt_fla_blackwell_scalar::state_kernel<4>:state_kernel<false>;
            assert(kernel_seen==expected&&null_statistics==active&&grid_seen.y==32u);
            assert(grid_seen.x==(active?(operation==1u?4u:8u):operation==2u?(scalar_columns==8?16u:32u):16u));
            assert(grid_seen.z==(operation==2u?1u:(count+63u)/64u));
        }
    }
    setenv("QRT_FLA_GDN_COARSE_INTERVAL","1",1);scalar_mode=1;scalar_columns=8;
    qrt_fla_checkpoint::Segment checkpoint{};checkpoint.count=1u;checkpoint.prefix_tokens[0]=64u;
    checkpoint.states[0]=reinterpret_cast<float*>(uintptr_t(0x70000000u));
    auto words=reinterpret_cast<uint16_t*>(uintptr_t(0x10000000u));auto floats=reinterpret_cast<float*>(uintptr_t(0x20000000u));
    for(bool failure:{false,true}){
        launches=queries=0;fail=failure;
        const auto status=state_checkpoints(words,words+1,words+2,floats,words+3,words+4,floats+1,65u,reinterpret_cast<unsigned char*>(uintptr_t(0x30000000u)),nullptr,checkpoint);
        assert(status==(failure?hipErrorUnknown:hipSuccess)&&launches==1u&&queries==1u);
        assert(kernel_seen==state_kernel<true>&&grid_seen.x==32u&&!null_statistics);
    }
    std::printf("coarse_gdn_launch_cases=%u original_checkpoint_route=1 null_statistics=1 failures_propagated=1\n",checked);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            executable = str(Path(directory) / "launch")
            subprocess.run(["c++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                            "-I", str(ROOT), "-x", "c++", "-", "-o", executable],
                           input=code, text=True, check=True, timeout=30)
            result = subprocess.run([executable], capture_output=True, text=True,
                                    timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("coarse_gdn_launch_cases=1200", result.stdout)
        print(result.stdout.strip(), flush=True)
