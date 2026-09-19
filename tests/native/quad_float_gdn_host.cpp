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
#include "native/providers/moe_accumulator/sm121_strong_float_subgroup.h"

constexpr unsigned host_threads=32u;
struct Dim { unsigned x=0,y=0,z=0; };
thread_local Dim threadIdx,blockIdx;
struct Barrier {
    std::mutex lock;std::condition_variable changed;unsigned arrived=0,epoch=0;
    void wait(unsigned count) {
        std::unique_lock<std::mutex> guard(lock);const unsigned before=epoch;
        if(++arrived==count){arrived=0;++epoch;changed.notify_all();}
        else changed.wait(guard,[&]{return epoch!=before;});
    }
} cta,quads[host_threads/4u];
void __syncthreads(){cta.wait(host_threads);}
unsigned atomicAnd(unsigned* p,unsigned v){return __atomic_fetch_and(p,v,__ATOMIC_RELAXED);}
std::atomic<unsigned> fast_groups{0},fallback_dots{0};
namespace qrt_sm121_strong_float {
template<unsigned Lanes> Value accumulate(Value carry,const Product* input){
    static_assert(Lanes==4u);
    static Product operands[host_threads/4u][16];
    const unsigned lane=threadIdx.x%4u,quad=threadIdx.x/4u;
    for(unsigned i=0;i<4u;++i)operands[quad][lane*4u+i]=input[i];
    quads[quad].wait(4u);
    const auto result=group(carry,operands[quad]);
    if(!lane)++fast_groups;
    quads[quad].wait(4u);
    return result;
}
}
namespace original=qrt_q1_moe_hawkeye;
float from(uint16_t b){uint32_t bits=uint32_t(b)<<16u;float f;std::memcpy(&f,&bits,4u);return f;}
uint16_t rounded(float f){uint32_t bits;std::memcpy(&bits,&f,4u);return uint16_t((bits+0x7fffu+((bits>>16u)&1u))>>16u);}
float reference_dot(const uint16_t* a,const uint16_t* b,unsigned count){
    return original::accumulate_bf16_hopper_blackwell(0.0f,a,b,count);
}
namespace qrt_fla_blackwell_scalar {
constexpr unsigned threads=host_threads,columns=8u;
float from_bf16(uint16_t b){return from(b);} uint16_t to_bf16(float f){return rounded(f);}
float exponential(float x,const unsigned char*){return std::exp2(x*1.4426950408889634074f);}
bool eligible(uint16_t a,uint16_t b){return qrt_sm121_float_alignment::eligible(a)&&qrt_sm121_float_alignment::eligible(b);}
uint32_t pack(uint16_t a,uint16_t b){return uint32_t(a)|(uint32_t(b)<<16u);}
template<unsigned Width,unsigned Columns=columns>
float dot(const uint32_t* left,const uint32_t (&right)[Width/2u][Columns],unsigned col,bool){
    ++fallback_dots;uint16_t a[Width],b[Width];
    for(unsigned i=0;i<Width;++i){a[i]=uint16_t(left[i/2u]>>((i%2u)*16u));b[i]=uint16_t(right[i/2u][col]>>((i%2u)*16u));}
    return reference_dot(a,b,Width);
}
}
#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__ static
using std::min;
#include "quad_under_test.h"

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
namespace quad=qrt_fla_quad_float;
constexpr uint16_t sentinel=0x5a5au;
constexpr float fguard=12345.0f;
bool owned(unsigned column){return column<8u||column>=120u;}
bool head_owned(unsigned head){return head==0u||head==31u;}
float reference_exp(float x){return qrt_fla_blackwell_scalar::exponential(x,nullptr);}
void fill(std::vector<uint16_t>& a){for(size_t i=0;i<a.size();++i)a[i]=rounded(float(int(i*7u%17u)-8)/64.0f);}
void check(unsigned count){
    const unsigned chunks=(count+63u)/64u;
    std::vector<uint16_t> q(count*2048u),k(q.size()),v(count*4096u),beta(count*32u),inv(q.size()),w(v.size(),sentinel);
    std::vector<float> g(count*32u);
    fill(q);fill(k);fill(v);fill(beta);fill(inv);
    for(size_t i=0;i<g.size();++i)g[i]=-float(i/32u%64u)/256.0f;
    q[0]=uint16_t(83u<<7u|17u);inv[0]=uint16_t(83u<<7u|11u);
    const auto original_q=q,original_k=k,original_v=v,original_beta=beta,original_inv=inv;
    const auto original_g=g;
    auto want_w=w,want_u=v;
    for(unsigned token=0;token<count;++token)for(unsigned head:{0u,31u})for(unsigned col=0;col<128u;++col){
        if(!owned(col))continue;const unsigned offset=token/64u*64u,valid=min(64u,count-offset);
        uint16_t a[64]{},b[64]{},c[64]{};
        for(unsigned r=0;r<valid;++r){const unsigned pos=offset+r;const float scale=from(beta[pos*32u+head]);
            a[r]=inv[(size_t(token)*32u+head)*64u+r];
            b[r]=rounded(from(rounded(from(k[(size_t(pos)*16u+head/2u)*128u+col])*scale))*reference_exp(g[pos*32u+head]));
            c[r]=rounded(from(original_v[(size_t(pos)*32u+head)*128u+col])*scale);}
        const size_t at=(size_t(token)*32u+head)*128u+col;
        want_w[at]=rounded(reference_dot(a,b,64u));want_u[at]=rounded(reference_dot(a,c,64u));
    }
    for(unsigned chunk=0;chunk<chunks;++chunk)for(unsigned head:{0u,31u})for(unsigned tile:{15u,0u})
        pool.run({tile,head,chunk},[&]{quad::wu_kernel(k.data(),v.data(),beta.data(),inv.data(),g.data(),w.data(),v.data(),count,nullptr);});
    assert(w==want_w&&v==want_u);
    fill(w);v=original_v;w[0]=uint16_t(83u<<7u|11u);
    const auto state_w=w;
    std::vector<float> state(524288u),expected;
    for(size_t i=0;i<state.size();++i)state[i]=float(int(i*3u%13u)-6)/128.0f;
    expected=state;
    std::vector<uint16_t> h(size_t(chunks)*524288u,sentinel),vn(v.size(),sentinel),want_h=h,want_vn=vn;
    for(unsigned head:{0u,31u})for(unsigned col=0;col<128u;++col){if(!owned(col))continue;
        const size_t base=(head*128u+col)*128u;
        for(unsigned offset=0;offset<count;offset+=64u){const unsigned valid=min(64u,count-offset);uint16_t prior[128],residual[64]{};
            for(unsigned j=0;j<128u;++j){prior[j]=rounded(expected[base+j]);want_h[size_t(offset/64u)*524288u+base+j]=prior[j];}
            for(unsigned r=0;r<valid;++r){const size_t pos=offset+r,at=(pos*32u+head)*128u+col;
                const float value=from(v[at])-reference_dot(w.data()+(pos*32u+head)*128u,prior,128u);
                want_vn[at]=rounded(value);residual[r]=rounded(value*reference_exp(g[(offset+valid-1u)*32u+head]-g[pos*32u+head]));}
            for(unsigned j=0;j<128u;++j){uint16_t keys[64]{};
                for(unsigned r=0;r<valid;++r)keys[r]=k[((size_t(offset)+r)*16u+head/2u)*128u+j];
                expected[base+j]=fmaf(expected[base+j],reference_exp(g[(offset+valid-1u)*32u+head]),reference_dot(keys,residual,64u));}
        }
    }
    for(unsigned head:{0u,31u})for(unsigned tile:{0u,15u})
        pool.run({tile,head,0},[&]{quad::state_kernel<8u>(k.data(),v.data(),w.data(),g.data(),h.data(),vn.data(),state.data(),count,nullptr);});
    assert(h==want_h&&vn==want_vn&&state==expected&&v==original_v&&w==state_w);
    std::vector<uint16_t> scores(count*2048u);fill(scores);fill(h);fill(v);
    const auto original_h=h,output_v=v,original_scores=scores;
    std::vector<float> output(v.size(),fguard),want_output=output;
    for(unsigned token=0;token<count;++token)for(unsigned head:{0u,31u})for(unsigned col=0;col<128u;++col){if(!owned(col))continue;
        const unsigned offset=token/64u*64u,valid=min(64u,count-offset);uint16_t values[64]{};
        for(unsigned r=0;r<valid;++r)values[r]=v[((size_t(offset)+r)*32u+head)*128u+col];
        const float old=reference_dot(q.data()+(size_t(token)*16u+head/2u)*128u,h.data()+size_t(token/64u)*524288u+(head*128u+col)*128u,128u);
        const float local=reference_dot(scores.data()+(size_t(token)*32u+head)*64u,values,64u);
        constexpr float scale=0.08838834764831845f;
        want_output[(size_t(token)*32u+head)*128u+col]=from(rounded(fmaf(local,scale,(old*reference_exp(g[token*32u+head]))*scale)));
    }
    for(unsigned chunk=0;chunk<chunks;++chunk)for(unsigned head:{0u,31u})for(unsigned tile:{0u,15u})
        pool.run({tile,head,chunk},[&]{quad::output_kernel(q.data(),v.data(),h.data(),g.data(),scores.data(),output.data(),count,nullptr);});
    assert(output==want_output&&h==original_h&&v==output_v&&scores==original_scores);
    assert(q==original_q&&k==original_k&&beta==original_beta&&inv==original_inv&&g==original_g);
}
int main(){check(1u);check(65u);assert(fast_groups>0&&fallback_dots>0);
    std::printf("quad_gdn_host_transport=pass fast_groups=%u fallback_dots=%u gpu_dpp_tested=0 inference_acceptance=0\n",fast_groups.load(),fallback_dots.load());}
