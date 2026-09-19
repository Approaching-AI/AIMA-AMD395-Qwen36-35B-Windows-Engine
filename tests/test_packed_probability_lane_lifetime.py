"""Execute the actual producer's softmax/store loop with 32 host lanes."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from tests.test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class PackedProbabilityLaneLifetimeTests(unittest.TestCase):
    def test_original_loop_preserves_scores_until_collective_consumption(self):
        source = (ROOT / "native/providers/ck_fmha/streamed_exact_attention.h").read_text()
        loop = function(source, "for (unsigned r = 0u; r < 4u; ++r)")
        code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>
#include "native/providers/ck_fmha/inplace_probability_storage.h"
#include "native/providers/ck_fmha/packed_probability_storage.h"
class Barrier {
    std::mutex mutex; std::condition_variable condition;
    unsigned arrived=0,generation=0;
public:
    void wait(){
        std::unique_lock<std::mutex> lock(mutex);const unsigned old=generation;
        if(++arrived==32u){arrived=0;++generation;condition.notify_all();}
        else condition.wait(lock,[&]{return old!=generation;});
    }
};
struct Transport{Barrier barrier;std::array<uint32_t,32> values{};};
thread_local unsigned host_lane=0;
thread_local Transport* transport=nullptr;
template<class T>T shuffle(T value,unsigned from,unsigned width){
    static_assert(sizeof(T)==4u);assert(width==32u&&from<32u);
    __builtin_memcpy(&transport->values[host_lane],&value,4u);
    transport->barrier.wait();T result;
    __builtin_memcpy(&result,&transport->values[from],4u);
    transport->barrier.wait();return result;
}
template<class T>T __shfl_xor(T value,unsigned mask,unsigned width){return shuffle(value,host_lane^mask,width);}
template<class T>T __shfl_down(T value,unsigned delta,unsigned width){
    return shuffle(value,host_lane+delta<width?host_lane+delta:host_lane,width);
}
uint32_t bits(float value){uint32_t result;__builtin_memcpy(&result,&value,4u);return result;}
namespace attention {
constexpr float kExactLog2e=1.4426950408889634f;
uint16_t f32_to_bf16(float value){const uint32_t u=bits(value);return uint16_t((u+0x7fffu+((u>>16u)&1u))>>16u);}
}
struct HostExp{static float evaluate(const unsigned char*,const unsigned char*,float value){return std::exp2(value);}};
struct State{
    unsigned start,count,stride,wave,head;bool vllm;
    Transport transport;
    std::vector<float> source,scales;
    std::vector<uint16_t> separate;
    std::array<float,32> alpha,denominator;
    uint16_t probability[32][32]{};
    State(unsigned first,unsigned rows,unsigned group,unsigned h,bool order)
        :start(first),count(rows),stride(first+rows),wave(group),head(h),vllm(order),
        source(size_t(rows)*16u*stride+16u,-37.25f),
        scales(size_t(rows)*16u*((stride+31u)/32u+1u),-19.5f),
        separate(size_t(rows)*16u*stride+16u,0xa5a5u){
        alpha.fill(-19.5f);denominator.fill(-19.5f);
        for(size_t i=0;i<size_t(rows)*16u*stride;++i)
            source[i+8u]=float(int((i*37u+i/13u)%61u)-30)*0.125f;
    }
};
template<unsigned Mode>void run_lane(State& state,unsigned lane){
    host_lane=lane;transport=&state.transport;
    constexpr bool FuseQk=false,InplaceProbability=Mode!=0u,PackedProbability=Mode==2u;
    using Exp=HostExp;
    const unsigned wave=state.wave,head=state.head,row_tile=0u,start=state.start;
    const unsigned count=state.count,stride=state.stride,tile_stride=(stride+31u)/32u;
    const bool vllm_sum=state.vllm;
    float scores[1][1]{};
    const float* source_scores=state.source.data()+8u;
    float* diagnostic_scores=nullptr;
    uint16_t* probabilities=Mode?reinterpret_cast<uint16_t*>(state.source.data()+8u):state.separate.data()+8u;
    auto& probability=state.probability;
    float* alpha=state.alpha.data();float* denominator=state.denominator.data();float* scales=state.scales.data();
    const unsigned char* exp2_table=nullptr;const unsigned char* packed_exp=nullptr;
    float running_max[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    float running_sum[4]={1.0f,1.0f,1.0f,1.0f};
    const unsigned tiles=(start+std::min(32u,count)+31u)/32u;
    for(unsigned tile=0u;tile<tiles;++tile){const unsigned key_base=tile*32u;
''' + loop + r'''
    }
}
template<unsigned Mode>void run(State& state){
    std::vector<std::thread> threads;
    for(unsigned lane=0u;lane<32u;++lane)threads.emplace_back([&,lane]{run_lane<Mode>(state,lane);});
    for(auto& thread:threads)thread.join();
}
void compare(const State& expected,const State& actual,bool packed){
    const unsigned stride=actual.stride,count=actual.count;
    for(size_t i=0;i<expected.scales.size();++i)assert(bits(expected.scales[i])==bits(actual.scales[i]));
    for(unsigned row=0;row<32u;++row){
        assert(bits(expected.alpha[row])==bits(actual.alpha[row]));
        assert(bits(expected.denominator[row])==bits(actual.denominator[row]));
        for(unsigned lane=0;lane<32u;++lane)assert(expected.probability[row][lane]==actual.probability[row][lane]);
    }
    for(unsigned row=0;row<count;++row)for(unsigned head=0;head<16u;++head){
        const bool visited=row%8u==actual.wave&&head==actual.head;
        const unsigned tokens=actual.start+row+1u;
        const unsigned end=visited?std::min(stride,((tokens+31u)/32u)*32u):0u;
        const size_t flat_row=size_t(row)*16u+head;
        for(unsigned key=0;key<end;++key){
            const auto value=packed?qrt_packed_probability_storage::load(actual.source.data()+8u,flat_row,key,stride)
                :qrt_inplace_probability_storage::load<true>(reinterpret_cast<const uint16_t*>(actual.source.data()+8u),flat_row*stride+key);
            assert(value==expected.separate[8u+flat_row*stride+key]);
        }
        const unsigned written_words=packed?(end+1u)/2u:end;
        for(unsigned key=written_words;key<stride;++key)
            assert(bits(expected.source[8u+flat_row*stride+key])==bits(actual.source[8u+flat_row*stride+key]));
    }
    for(unsigned i=0;i<8u;++i){
        assert(actual.source[i]==-37.25f&&actual.source[actual.source.size()-1u-i]==-37.25f);
        assert(actual.separate[i]==0xa5a5u&&actual.separate[actual.separate.size()-1u-i]==0xa5a5u);
    }
}
int main(){
    struct Shape{unsigned start,count,wave,head;};
    const Shape shapes[]={{0,1,0,0},{1,1,0,15},{31,1,0,3},{32,1,0,15},
        {63,1,0,0},{0,32,0,15},{0,32,7,0},{1,32,7,15},{33,31,6,7}};
    unsigned cases=0;
    for(auto shape:shapes)for(bool order:{false,true}){
        State original(shape.start,shape.count,shape.wave,shape.head,order);
        State tagged(shape.start,shape.count,shape.wave,shape.head,order);
        State packed(shape.start,shape.count,shape.wave,shape.head,order);
        run<0>(original);run<1>(tagged);run<2>(packed);
        compare(original,tagged,false);compare(original,packed,true);++cases;
    }
    std::printf("{\"actual_probability_loop_cases\":%u,\"host_lanes\":32,\"storage_formats\":3,"
        "\"both_sum_orders\":true,\"payload_scale_denominator_and_tail_match\":true,"
        "\"host_exp_and_transport\":true,\"gpu_execution\":false}\n",cases);
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-packed-probability-lanes-") as tmp:
            cpp = Path(tmp) / "lanes.cpp"
            cpp.write_text(code)
            exe = Path(tmp) / "lanes"
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-pthread",
                            "-fstrict-aliasing", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-I", str(ROOT),
                            str(cpp), "-o", str(exe)], check=True, timeout=45)
            subprocess.run([str(exe)], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
