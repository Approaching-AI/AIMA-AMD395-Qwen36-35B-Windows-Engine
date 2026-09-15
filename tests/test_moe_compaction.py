"""Execute real routed selectors and scheduling with a small threaded HIP model.

The dot stand-in validates transport; native tests supply arithmetic evidence.
"""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MoeCompactionTests(unittest.TestCase):
    def test_real_phases_windows_sparse_dense_debug_and_submission_faults(self):
        s = (ROOT / 'native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp').read_text()
        definitions = function(s, 'struct MoeCorrectionBounds {') + ';\n'
        definitions += function(s, 'enum class MoeCorrectionPhase {') + ';\n'
        partition = (ROOT / 'native/providers/moe_accumulator/sm121_replay_partition.h').read_text()
        definitions += 'namespace qrt_sm121_replay_partition {\n' + function(partition, 'void append_block(') + '\n}\n'
        definitions += 'template<uint32_t ProjectionRows, uint32_t InputDivisor, uint32_t WeightRows, uint32_t WeightOffset>\n' + function(s, 'void moe_collect_partitioned(') + '\n'
        definitions += function(s, 'uint32_t moe_replay_count(') + '\n'
        definitions += function(s, 'uint32_t moe_replay_cell(') + '\n'
        definitions += 'template<unsigned Lanes>\n' + function(s, 'float moe_routed_replay_dot(') + '\n'
        helpers = '\n'.join(function(s, signature) for signature in (
            'float routed_silu_from_gate_bf16(',
            'bool\nrouted_gate_projection_needs_hawkeye_replay(',
            'bool\nrouted_up_projection_needs_hawkeye_replay(',
            'bool moe_l2_candidate(',
        ))
        names = ['routed_gate_batched_hawkeye_correction_kernel',
                 'routed_up_batched_hawkeye_correction_activation_kernel',
                 'routed_down_batched_hawkeye_correction_kernel']
        kernels = '\n'.join('template<MoeCorrectionPhase Phase>\n' + function(s, 'void ' + name + '(') for name in names)
        launchers = function(s, 'template <uint32_t MaximumBlocks = kMaximumMoeCorrectionBlocks,')
        launchers += '\n' + function(s, 'template<bool NeedsFinalize, typename Kernel, typename... Args>')
        source = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include "moe_accumulator/bf16_midpoint_selector.h"
constexpr uint32_t kNativeThreads=32, kHidden=32, kIntermediate=16, kTopK=2;
constexpr uint32_t kRoutes=64, kActivatedElements=kRoutes*kIntermediate;
constexpr uint32_t kMaximumMoeCorrectionBlocks=4, kMoeCompactionBlocks=4;
constexpr uint32_t kMaximumMoeCompactionBlocks=64;
constexpr uint32_t kMoeCompactionCapacity=kMaximumMoeCompactionBlocks*kNativeThreads;
#define QRT_TRITON_MOE_ROUTED_PROJECTION_DEBUG 1
#define QRT_MOE_ROUTED_REPLAY_LANES 4
#define QRT_SM121_DOT_STAGING_GROUPS 4
#define __shared__ static
struct dim3 { unsigned x; explicit dim3(unsigned v=1):x(v){} };
thread_local dim3 blockIdx, threadIdx, blockDim, gridDim;
uint32_t __float_as_uint(float f) { uint32_t u; std::memcpy(&u,&f,4); return u; }
float __uint_as_float(uint32_t u) { float f; std::memcpy(&f,&u,4); return f; }
float __fmul_rn(float a,float b) { return a*b; }
float __builtin_amdgcn_exp2f(float f) { return std::exp2(f); }
uint16_t float_to_bf16(float f) { uint32_t u=__float_as_uint(f); return (u+0x7fff+((u>>16)&1))>>16; }
float bf16_to_float(uint16_t u) { return __uint_as_float(uint32_t(u)<<16); }
uint32_t atomicAdd(uint32_t *p,uint32_t v) { return __atomic_fetch_add(p,v,__ATOMIC_RELAXED); }
std::mutex barrier_lock;
std::condition_variable barrier_changed;
unsigned arrived=0,generation=0;
void __syncthreads() {
    std::unique_lock<std::mutex> l(barrier_lock); unsigned old=generation;
    if (++arrived==kNativeThreads) { arrived=0; ++generation; barrier_changed.notify_all(); }
    else barrier_changed.wait(l,[&]{return generation!=old;});
}
std::atomic<unsigned> dots{0};
namespace qrt_sm121_subgroup {
template<unsigned Lanes>
float dot(const uint16_t *a,const uint16_t *b,unsigned k) {
    if ((threadIdx.x&(Lanes-1u))==0) ++dots;
    float value=0; for(unsigned i=0;i<k;++i) value+=bf16_to_float(a[i])*bf16_to_float(b[i]);
    return value;
}
}
namespace qrt_sm121_prepared_projection {
template<unsigned Lanes, unsigned Groups>
float dot(const uint16_t* a,const uint16_t* b,unsigned k) {
    static_assert(Groups==4); return qrt_sm121_subgroup::dot<Lanes>(a,b,k);
}
}
namespace qrt_sm121_float_subgroup {
template<unsigned Lanes, unsigned Groups>
float dot(const uint16_t* a,const uint16_t* b,unsigned k) {
    static_assert(Groups==4); return qrt_sm121_subgroup::dot<Lanes>(a,b,k);
}
}
std::atomic<unsigned> validated_calls{0},fallback_calls{0};
namespace qrt_sm121_scalar_projection {
template<unsigned Lanes, unsigned Groups>
float validated_dot(const uint16_t* a,const uint16_t* b,unsigned k,bool eligible) {
    static_assert(Groups==4);
    if((threadIdx.x&(Lanes-1u))==0) { if(eligible)++validated_calls;else ++fallback_calls; }
    return qrt_sm121_subgroup::dot<Lanes>(a,b,k);
}
}
namespace qrt_sm121_scaled_fallback {
template<unsigned Lanes,unsigned Groups>
float dot(const uint16_t* a,const uint16_t* b,unsigned k,unsigned flags) {
    return qrt_sm121_scalar_projection::validated_dot<Lanes,Groups>(a,b,k,(flags&1u)!=0u);
}
}
std::atomic<unsigned> staged_calls{0};
namespace qrt_sm121_staged_half_projection {
// Arithmetic is checked on gfx1151; these explicit rows expose stale views,
// row-stride errors and incorrect expert/activation indices to the host test.
struct Row { uint16_t raw[16];uint32_t control; };
template<unsigned Groups> float dot(const Row* a,const Row* b,unsigned columns) {
    static_assert(Groups==2);
    if((threadIdx.x&3u)==0u) {++dots;++staged_calls;}
    float value=0;
    for(unsigned c=0;c<columns;++c)value+=bf16_to_float(a[c/16].raw[c%16])*bf16_to_float(b[c/16].raw[c%16]);
    return value;
}
}
namespace qrt_routed_consumer_audit {
template<class... T> void observe(T...) { assert(false); }
}
''' + definitions + helpers + kernels + r'''
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t=void *;
namespace qrt_routed_consumer_audit {
struct Owner {
    Owner(hipStream_t, const char*) {}
    hipError_t initialize(bool enabled) { assert(!enabled); return hipSuccess; }
    unsigned* data() const { return nullptr; }
    hipError_t finish() { return hipSuccess; }
};
}
enum class MoeL2 { Input, Weight, RoutedGateUp=Weight, RoutedActivated, RoutedDown,
    SharedInput, SharedGate, SharedUp, SharedActivated, SharedDown };
struct State {
    bool compact_routed_hawkeye=false;
    bool prepared_replay_active=false;
    bool float_replay_active=false,prevalidated_float_active=false,partition_replay=false;
    bool staged_half_replay_active=false;
    bool scaled_significand_fallback=false;
    bool shared_prevalidated_float_active=false,shared_staged_half_active=false;
    std::array<uint16_t*,9> shared_staged_operands{};
    std::array<uint32_t*,9> shared_replay_rows{};
    uint16_t *prepared_replay_weights=nullptr,*prepared_replay_inputs=nullptr;
    uint32_t *prepared_replay_weight_rows=nullptr,*prepared_replay_input_rows=nullptr;
    uint32_t moe_compaction_blocks=kMoeCompactionBlocks;
    uint32_t sm121_moe_absolute_error_ppb=1000;
    std::array<float *,9> moe_l2{};
    uint32_t *moe_compacted_indices=nullptr,*moe_compacted_count=nullptr;
} g_state;
bool execute_kernels=true;
unsigned api_calls=0, fail_api=0;
hipStream_t wanted_stream=reinterpret_cast<void *>(17);
struct Pool {
    std::mutex lock; std::condition_variable changed;
    unsigned epoch=0,finished=0,blocks=0; bool stop=false;
    std::function<void()> job; std::vector<std::thread> workers;
    Pool() {
        for(unsigned lane=0;lane<kNativeThreads;++lane) workers.emplace_back([&,lane]{
            unsigned seen=0;
            for (;;) {
                std::unique_lock<std::mutex> l(lock);
                changed.wait(l,[&]{return stop||epoch!=seen;});
                if(stop) return;
                seen=epoch; auto work=job; unsigned count=blocks; l.unlock();
                threadIdx.x=lane; blockDim.x=kNativeThreads; gridDim.x=count;
                for(unsigned block=0;block<count;++block) {
                    blockIdx.x=block; work(); __syncthreads();
                }
                l.lock(); ++finished; changed.notify_all();
            }
        });
    }
    void run(unsigned count,std::function<void()> work) {
        std::unique_lock<std::mutex> l(lock); blocks=count; job=work; finished=0;
        ++epoch; changed.notify_all(); changed.wait(l,[&]{return finished==kNativeThreads;});
    }
    ~Pool() { {std::lock_guard<std::mutex> l(lock);stop=true;changed.notify_all();} for(auto &t:workers)t.join(); }
} pool;
template<class Kernel,class... Args>
void launch(Kernel kernel,dim3 grid,dim3 block,int,hipStream_t stream,Args... args) {
    assert(stream==wanted_stream&&block.x==kNativeThreads&&grid.x<=g_state.moe_compaction_blocks);
    if (!execute_kernels) return;
    pool.run(grid.x,[=]{kernel(args...);});
    auto parameters=std::make_tuple(args...);
    const auto bounds=std::get<sizeof...(Args)-1>(parameters);
    if (bounds.compacted_count) {
        unsigned count=moe_replay_count(bounds);
        assert(count<=g_state.moe_compaction_blocks*kNativeThreads);
        std::vector<unsigned> indices;
        for(unsigned slot=0;slot<count;++slot)indices.push_back(moe_replay_cell(bounds,slot));
        std::sort(indices.begin(),indices.end());
        assert(std::adjacent_find(indices.begin(),indices.end())==indices.end());
        for (auto i:indices) assert(i>=bounds.first_block*kNativeThreads&&i<(bounds.first_block+g_state.moe_compaction_blocks)*kNativeThreads);
    }
}
#define hipLaunchKernelGGL(...) launch(__VA_ARGS__)
hipError_t hipGetLastError() { return ++api_calls==fail_api?hipErrorUnknown:hipSuccess; }
hipError_t hipMemsetAsync(void *p,int value,size_t bytes,hipStream_t stream) {
    assert(stream==wanted_stream&&bytes==(g_state.partition_replay?8u:4u));
    if(++api_calls==fail_api)return hipErrorUnknown;
    std::memset(p,value,bytes); return hipSuccess;
}
''' + launchers + r'''
struct Data {
    std::vector<float> native,down,input_norm,weight_norm,gate_f32,up_f32;
    std::vector<uint16_t> input,weights,down_weights,activated,gate_debug,up_debug,lut;
    std::vector<int32_t> ids;
    std::vector<float> topk;
    uint32_t debug_count=0;
    Data(unsigned routes):native(2*kActivatedElements+17,12345.0f),down(kRoutes*kHidden+17,12345.0f),
      input_norm(kRoutes,1),weight_norm(2*kHidden,1),gate_f32(kTopK*kIntermediate,0),up_f32(kTopK*kIntermediate,0),
      input(kRoutes*kHidden),weights(4*kIntermediate*kHidden),down_weights(2*kHidden*kIntermediate),
      activated(kActivatedElements+17,0x5a5a),gate_debug(kTopK*kIntermediate,0),up_debug(kTopK*kIntermediate,0),lut(65536),ids(routes),topk(routes) {
        for(unsigned i=0;i<lut.size();++i)lut[i]=float_to_bf16(std::fmod(float(i),19)/16);
        for(unsigned i=0;i<input.size();++i)input[i]=float_to_bf16(float(int(i*3%19)-9)/16);
        for(unsigned i=0;i<weights.size();++i)weights[i]=float_to_bf16(float(int(i*7%23)-11)/32);
        for(unsigned i=0;i<down_weights.size();++i)down_weights[i]=float_to_bf16(float(int(i*5%17)-8)/32);
        for(unsigned i=0;i<routes;++i){ids[i]=(i/2)%2;topk[i]=(i%2?-1:1)*0.25f;}
        for(unsigned i=0;i<routes*kIntermediate;++i) {
            native[i]=__uint_as_float(0x3e000000u+(i*13517u)%0x2000000u);
            native[kActivatedElements+i]=__uint_as_float(0x3e800000u+(i*7179u)%0x2000000u);
        }
        for(unsigned i=0;i<routes*kHidden;++i)down[i]=__uint_as_float(0x3e000000u+(i*17777u)%0x2000000u);
    }
};
hipError_t run(Data &d,unsigned routes,unsigned radius,unsigned exponent) {
    unsigned blocks=(routes*kIntermediate+kNativeThreads-1)/kNativeThreads;
    g_state.moe_l2={d.input_norm.data(),d.weight_norm.data()};
    g_state.moe_l2[size_t(MoeL2::RoutedActivated)]=d.input_norm.data();
    g_state.moe_l2[size_t(MoeL2::RoutedDown)]=d.weight_norm.data();
    using Row=qrt_sm121_staged_half_projection::Row;
    std::vector<Row> staged_input(d.input.size()/16),staged_weight(d.weights.size()/16);
    auto pack=[](std::vector<Row>& out,const std::vector<uint16_t>& values,size_t elements) {
        assert(elements%16==0&&elements/16<=out.size());
        for(size_t i=0;i<elements;++i)out[i/16].raw[i%16]=values[i];
    };
    if(g_state.staged_half_replay_active) {
        pack(staged_input,d.input,d.input.size());pack(staged_weight,d.weights,d.weights.size());
        g_state.prepared_replay_inputs=reinterpret_cast<uint16_t*>(staged_input.data());
        g_state.prepared_replay_weights=reinterpret_cast<uint16_t*>(staged_weight.data());
    }
    using P=MoeCorrectionPhase;
    auto status=launch_moe_routed_correction<false>(
        routed_gate_batched_hawkeye_correction_kernel<P::Local>,routed_gate_batched_hawkeye_correction_kernel<P::Collect>,
        routed_gate_batched_hawkeye_correction_kernel<P::Replay>,routed_gate_batched_hawkeye_correction_kernel<P::Local>,
        blocks,wanted_stream,MoeL2::Input,MoeL2::Weight,d.native.data(),d.input.data(),d.weights.data(),d.ids.data(),
        d.activated.data(),d.lut.data(),routes,radius,exponent,d.gate_debug.data(),d.gate_f32.data(),&d.debug_count,0u);
    if(status!=hipSuccess)return status;
    status=launch_moe_routed_correction<true>(
        routed_up_batched_hawkeye_correction_activation_kernel<P::Local>,routed_up_batched_hawkeye_correction_activation_kernel<P::Collect>,
        routed_up_batched_hawkeye_correction_activation_kernel<P::Replay>,routed_up_batched_hawkeye_correction_activation_kernel<P::Finalize>,
        blocks,wanted_stream,MoeL2::Input,MoeL2::Weight,d.native.data(),d.input.data(),d.weights.data(),d.ids.data(),
        d.activated.data(),d.lut.data(),routes,radius,exponent,d.up_debug.data(),d.up_f32.data(),&d.debug_count,0u);
    if(status!=hipSuccess)return status;
    blocks=(routes*kHidden+kNativeThreads-1)/kNativeThreads;
    if(g_state.staged_half_replay_active) {
        pack(staged_input,d.activated,routes*kIntermediate);pack(staged_weight,d.down_weights,d.down_weights.size());
    }
    return launch_moe_routed_correction<false>(
        routed_down_batched_hawkeye_correction_kernel<P::Local>,routed_down_batched_hawkeye_correction_kernel<P::Collect>,
        routed_down_batched_hawkeye_correction_kernel<P::Replay>,routed_down_batched_hawkeye_correction_kernel<P::Local>,
        blocks,wanted_stream,MoeL2::RoutedActivated,MoeL2::RoutedDown,d.down.data(),d.topk.data(),d.ids.data(),d.activated.data(),
        d.down_weights.data(),routes,radius,exponent);
}
int main() {
    std::vector<uint32_t> indices(kMoeCompactionCapacity+17,0xabcdef),counter(18,0xabcdef);
    g_state.moe_compacted_indices=indices.data();g_state.moe_compacted_count=counter.data();
    std::vector<uint32_t> input_flags(kRoutes),weight_flags(2*kHidden);
    for(unsigned i=0;i<input_flags.size();++i)input_flags[i]=i%2;
    for(unsigned i=0;i<weight_flags.size();++i)weight_flags[i]=i%3!=0;
    g_state.prepared_replay_input_rows=input_flags.data();g_state.prepared_replay_weight_rows=weight_flags.data();
    for(unsigned route_mode:{0u,1u,2u,3u,4u}) {
    std::fill(counter.begin(),counter.end(),0xabcdef);
    g_state.float_replay_active=route_mode==1;g_state.prevalidated_float_active=route_mode>=2;g_state.partition_replay=route_mode==3;
    g_state.staged_half_replay_active=route_mode==4;
    for(unsigned window:{kMoeCompactionBlocks,kMaximumMoeCompactionBlocks}) {
    g_state.moe_compaction_blocks=window;
    for(unsigned routes:{1u,3u,9u,19u})for(unsigned mode:{0u,1u,2u,3u}) {
        Data original(routes),local=original,compact=original;
        unsigned radius=mode==1?32768u:mode==2?128u:0u,exponent=mode==2?125u:0u;
        // Tiny nonzero scale retains the original bounded launcher for empty controls.
        g_state.sm121_moe_absolute_error_ppb=mode==3?1000000u:1u;
        dots=0;g_state.compact_routed_hawkeye=false;assert(run(local,routes,radius,exponent)==hipSuccess);
        unsigned local_dots=dots;dots=0;
        g_state.compact_routed_hawkeye=true;assert(run(compact,routes,radius,exponent)==hipSuccess);
        assert(dots==local_dots&&compact.native==local.native&&compact.activated==local.activated&&compact.down==local.down);
        assert(compact.gate_debug==local.gate_debug&&compact.up_debug==local.up_debug&&compact.debug_count==local.debug_count);
        assert(compact.gate_f32==local.gate_f32&&compact.up_f32==local.up_f32);
        assert(compact.input==original.input&&compact.weights==original.weights&&compact.down_weights==original.down_weights);
        assert(compact.ids==original.ids&&compact.topk==original.topk&&compact.lut==original.lut);
        for(size_t i=kMoeCompactionCapacity;i<indices.size();++i)assert(indices[i]==0xabcdef);
        for(size_t i=g_state.partition_replay?2u:1u;i<counter.size();++i)assert(counter[i]==0xabcdef);
    }
    execute_kernels=false;Data d(19);
    api_calls=0;assert(run(d,19,32768,0)==hipSuccess);unsigned total=api_calls;
    for(fail_api=1;fail_api<=total;++fail_api) {
        api_calls=0;assert(run(d,19,32768,0)==hipErrorUnknown);assert(api_calls==fail_api);
    }
    fail_api=0;api_calls=0;
    for(unsigned invalid:{0u,2u,5u,128u}) {
        g_state.moe_compaction_blocks=invalid;
        assert(run(d,19,32768,0)==hipErrorInvalidValue&&api_calls==0);
    }
    g_state.moe_compaction_blocks=window;
    execute_kernels=true;
    }
    }
    assert(validated_calls>0&&fallback_calls>0&&staged_calls>0);
    Data d(19);g_state.moe_compacted_count=nullptr;api_calls=0;
    assert(run(d,19,32768,0)==hipErrorInvalidValue&&api_calls==0);

    g_state=State{};g_state.shared_prevalidated_float_active=true;g_state.shared_staged_half_active=true;
    std::atomic<unsigned> shared_calls{0};
    for(auto pair:{std::pair<MoeL2,MoeL2>{MoeL2::SharedInput,MoeL2::SharedGate},
                  {MoeL2::SharedInput,MoeL2::SharedUp},{MoeL2::SharedActivated,MoeL2::SharedDown}}){
        const auto i=size_t(pair.first),w=size_t(pair.second);
        unsigned fi=1,fw=1;uint16_t xi=17,xw=23;
        g_state.shared_replay_rows[i]=&fi;g_state.shared_replay_rows[w]=&fw;
        g_state.shared_staged_operands[i]=&xi;g_state.shared_staged_operands[w]=&xw;
        auto inspect=[&](MoeCorrectionBounds b){
            assert(b.prevalidated_float&&b.staged_half_replay&&b.prepared_input==&xi&&b.prepared_weights==&xw);
            assert(b.prepared_input_rows==&fi&&b.prepared_weight_rows==&fw);++shared_calls;
        };
        shared_calls=0;api_calls=0;
        assert(launch_moe_correction(inspect,1,wanted_stream,pair.first,pair.second)==hipSuccess&&shared_calls==kNativeThreads);
        for(bool input:{true,false}){
            g_state.shared_staged_operands[input?i:w]=nullptr;api_calls=0;
            assert(launch_moe_correction(inspect,1,wanted_stream,pair.first,pair.second)==hipErrorInvalidValue&&api_calls==0);
            g_state.shared_staged_operands[input?i:w]=input?&xi:&xw;
        }
        g_state.shared_staged_half_active=false;
        auto disabled=[&](MoeCorrectionBounds b){assert(b.prevalidated_float&&!b.staged_half_replay&&!b.prepared_input&&!b.prepared_weights);};
        assert(launch_moe_correction(disabled,1,wanted_stream,pair.first,pair.second)==hipSuccess);
        g_state.shared_staged_half_active=true;
    }
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-moe-compaction-') as tmp:
            exe = str(Path(tmp) / 'compaction-test')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-pthread',
                            '-fsanitize=address,undefined', '-g', '-O1', '-I', str(ROOT / 'native/providers'),
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=60)
            subprocess.run([exe], check=True, timeout=60)


if __name__ == '__main__':
    unittest.main()
