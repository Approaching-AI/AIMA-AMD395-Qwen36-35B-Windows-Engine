#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include "native/providers/moe_accumulator/sm121_narrow_f32_carry.h"

// The host transport uses64 threads. Native fixtures use the production256.
constexpr unsigned host_threads=64u;
struct Dim {unsigned x=0,y=0,z=0;};
thread_local Dim threadIdx,blockIdx;
struct Barrier {
    std::mutex lock;std::condition_variable changed;unsigned arrived=0,epoch=0;
    void wait(){std::unique_lock<std::mutex> guard(lock);const unsigned before=epoch;
        if(++arrived==host_threads){arrived=0;++epoch;changed.notify_all();}
        else changed.wait(guard,[&]{return epoch!=before;});}
} barrier;
void __syncthreads(){barrier.wait();}
unsigned atomicAnd(unsigned* p,unsigned v){return __atomic_fetch_and(p,v,__ATOMIC_RELAXED);}
float from(uint16_t b){uint32_t bits=uint32_t(b)<<16u;float f;std::memcpy(&f,&bits,4u);return f;}
uint16_t rounded(float f){uint32_t bits;std::memcpy(&bits,&f,4u);return uint16_t((bits+0x7fffu+((bits>>16u)&1u))>>16u);}
float reference_dot(const uint16_t* a,const uint16_t* b,unsigned count){
    return qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,a,b,count);
}
namespace qrt_fla_blackwell_scalar {
constexpr unsigned threads=host_threads,columns=8u;
float from_bf16(uint16_t b){return from(b);}uint16_t to_bf16(float f){return rounded(f);}
float exponential(float x,const unsigned char*){return std::exp2(x*1.4426950408889634074f);}
bool eligible(uint16_t a,uint16_t b){return qrt_sm121_float_alignment::eligible(a)&&qrt_sm121_float_alignment::eligible(b);}
uint32_t pack(uint16_t a,uint16_t b){return uint32_t(a)|(uint32_t(b)<<16u);}
template<unsigned Width,unsigned Columns=columns>
float dot(const uint32_t* left,const uint32_t (&right)[Width/2u][Columns],unsigned column,bool){
    uint16_t a[Width],b[Width];
    for(unsigned i=0;i<Width;++i){a[i]=uint16_t(left[i/2u]>>((i%2u)*16u));b[i]=uint16_t(right[i/2u][column]>>((i%2u)*16u));}
    return reference_dot(a,b,Width);
}
}
#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__ static
using std::min;
#include "separate_state_under_test.h"

struct Pool {
    std::mutex lock;std::condition_variable changed;unsigned epoch=0,finished=0;bool stop=false;
    Dim index;std::function<void()> job;std::vector<std::thread> workers;
    Pool(){for(unsigned lane=0;lane<host_threads;++lane)workers.emplace_back([&,lane]{
        unsigned seen=0;for(;;){std::unique_lock<std::mutex> guard(lock);
            changed.wait(guard,[&]{return stop||epoch!=seen;});if(stop)return;
            seen=epoch;auto work=job;const Dim at=index;guard.unlock();
            threadIdx.x=lane;blockIdx=at;work();guard.lock();++finished;changed.notify_all();}
    });}
    void run(Dim at,std::function<void()> work){std::unique_lock<std::mutex> guard(lock);
        index=at;job=work;finished=0;++epoch;changed.notify_all();changed.wait(guard,[&]{return finished==host_threads;});}
    ~Pool(){{std::lock_guard<std::mutex> guard(lock);stop=true;changed.notify_all();}for(auto& worker:workers)worker.join();}
} pool;
namespace candidate=qrt_fla_separate_state;
constexpr uint16_t sentinel=0x5a5au;
constexpr unsigned flag_guard=0x5a1234a5u;
unsigned total_fast=0,total_replay=0,late_partial_replays=0;
bool owned(unsigned head,unsigned column){return (head==0u||head==31u)&&(column<8u||column>=120u);}
float reference_exp(float x){return qrt_fla_blackwell_scalar::exponential(x,nullptr);}
void fill(std::vector<uint16_t>& a){for(size_t i=0;i<a.size();++i)a[i]=rounded(float(int(i*7u%17u)-8)/64.0f);}
template<class T> bool exact(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(T));}

void check(unsigned count,unsigned family){
    const unsigned chunks=(count+63u)/64u;
    std::vector<uint16_t> k(size_t(count)*2048u),u(size_t(count)*4096u),w(u.size());
    fill(k);fill(u);fill(w);
    std::vector<float> g(size_t(count)*32u),state(524288u);
    for(size_t i=0;i<g.size();++i)g[i]=-float(i/32u%64u)/256.0f;
    for(size_t i=0;i<state.size();++i)state[i]=float(int(i*3u%13u)-6)/128.0f;
    if(family==1u){assert(count>64u);w[size_t(64u)*4096u]=uint16_t(94u<<7u|3u);}
    if(family==2u)state[(31u*128u+120u)*128u]=from(uint16_t(94u<<7u|3u));
    if(family==3u){assert(count>64u);k[(size_t(64u)*16u+15u)*128u]=uint16_t(94u<<7u|3u);}
    if(family==4u)for(unsigned row=0;row<count;++row)g[row*32u+31u]=-float(row%64u)*2.0f;
    if(family==5u){w[0]=1u;state[(31u*128u+120u)*128u]=from(uint16_t(191u<<7u|3u));}
    const auto original_k=k,original_u=u,original_w=w;
    const auto original_g=g,original_state=state;
    auto expected=state;
    std::vector<uint16_t> h(size_t(chunks)*524288u+2u,sentinel),vn(u.size()+2u,sentinel),want_h=h,want_vn=vn;
    for(unsigned head:{0u,31u})for(unsigned column=0;column<128u;++column){if(!owned(head,column))continue;
        const size_t base=(head*128u+column)*128u;
        for(unsigned offset=0;offset<count;offset+=64u){const unsigned valid=min(64u,count-offset);uint16_t prior[128],residual[64]{};
            for(unsigned feature=0;feature<128u;++feature){prior[feature]=rounded(expected[base+feature]);want_h[1u+size_t(offset/64u)*524288u+base+feature]=prior[feature];}
            for(unsigned row=0;row<valid;++row){const size_t token=offset+row,at=(token*32u+head)*128u+column;
                const float value=from(u[at])-reference_dot(w.data()+(token*32u+head)*128u,prior,128u);
                want_vn[at+1u]=rounded(value);residual[row]=rounded(value*reference_exp(g[(offset+valid-1u)*32u+head]-g[token*32u+head]));}
            for(unsigned feature=0;feature<128u;++feature){uint16_t keys[64]{};
                for(unsigned row=0;row<valid;++row)keys[row]=k[((size_t(offset)+row)*16u+head/2u)*128u+feature];
                expected[base+feature]=fmaf(expected[base+feature],reference_exp(g[(offset+valid-1u)*32u+head]),reference_dot(keys,residual,64u));}
        }
    }
    std::vector<unsigned> flags(514u,flag_guard);
    for(unsigned head:{0u,31u})for(unsigned tile:{0u,15u})
        pool.run({tile,head,0u},[&]{candidate::fast_kernel<8u>(k.data(),u.data(),w.data(),g.data(),h.data()+1u,vn.data()+1u,state.data(),count,nullptr,flags.data()+1u);});
    unsigned local_fast=0,local_replay=0;
    for(unsigned head=0;head<32u;++head)for(unsigned tile=0;tile<16u;++tile){
        const unsigned flag=flags[1u+head*16u+tile];
        if(!owned(head,tile*8u)){assert(flag==flag_guard);continue;}
        assert(flag<=1u);if(flag){++total_fast;++local_fast;}else{++total_replay;++local_replay;}
        for(unsigned column=tile*8u;column<(tile+1u)*8u;++column){
            const size_t base=(head*128u+column)*128u;
            const auto& wanted=flag?expected:original_state;
            assert(!std::memcmp(state.data()+base,wanted.data()+base,128u*sizeof(float)));
        }
        if(!flag&&count>64u&&vn[1u+(head*128u+tile*8u)]!=sentinel)++late_partial_replays;
    }
    if(!family)assert(local_fast==4u&&!local_replay);
    if(family)assert(local_replay>0u);
    for(unsigned head:{31u,0u})for(unsigned tile:{15u,0u})
        pool.run({tile,head,0u},[&]{candidate::replay_kernel<8u>(k.data(),u.data(),w.data(),g.data(),h.data()+1u,vn.data()+1u,state.data(),count,nullptr,flags.data()+1u);});
    assert(exact(h,want_h)&&exact(vn,want_vn)&&exact(state,expected));
    assert(flags.front()==flag_guard&&flags.back()==flag_guard);
    assert(exact(k,original_k)&&exact(u,original_u)&&exact(w,original_w)&&exact(g,original_g));
}
int main(){
    check(1u,0u);check(65u,0u);
    for(unsigned family=1u;family<=5u;++family)check(129u,family);
    assert(total_fast&&total_replay&&late_partial_replays);
    std::printf("separate_state_host=pass fast_ctas=%u replay_ctas=%u partial_replays=%u gpu_execution=0 inference_acceptance=0\n",total_fast,total_replay,late_partial_replays);
}
