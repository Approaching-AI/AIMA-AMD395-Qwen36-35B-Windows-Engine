#pragma once
#include "../moe_accumulator/sm121_decoded_bf16.h"
#include "../moe_accumulator/sm121_f32_carry.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// Isolated host/GPU partition experiment. Preparation is lossless. Each SIMD
// lane owns an independent original K256 dot, including every ordered K16
// carry. Unsupported rows or carried endpoints restart the original wide dot.
// No runtime provider includes this header.
namespace qrt_cpu_exact_qk {
constexpr unsigned width=256u,query_heads=16u,key_heads=2u;
struct Prepared {
    const uint16_t* query;
    const uint16_t* key;
    unsigned tokens,pitch;
    std::vector<float> values;
    std::vector<int> exponents;
    std::vector<unsigned char> query_ok,key_ok;
    Prepared(const uint16_t* q,const uint16_t* k,unsigned n):query(q),key(k),tokens(n),pitch((n+15u)&~15u) {
        if(!q||!k||!n||n>8192u)throw std::invalid_argument("CPU QK preparation shape");
        values.resize(size_t(key_heads)*width*pitch);
        exponents.resize(values.size(),qrt_sm121_decoded_bf16::zero_exponent);
        query_ok.assign(size_t(n)*query_heads,1u);key_ok.assign(size_t(n)*key_heads,1u);
        for(size_t row=0;row<query_ok.size();++row)
            for(unsigned i=0;i<width;++i)query_ok[row]&=qrt_sm121_float_alignment::eligible(q[row*width+i]);
        for(unsigned t=0;t<n;++t)for(unsigned h=0;h<key_heads;++h)for(unsigned i=0;i<width;++i) {
            const uint16_t x=k[(size_t(t)*key_heads+h)*width+i];
            const size_t cell=(size_t(h)*width+i)*pitch+t;
            values[cell]=qrt_sm121_decoded_bf16::value(x);
            exponents[cell]=qrt_sm121_decoded_bf16::exponent(x);
            key_ok[size_t(t)*key_heads+h]&=qrt_sm121_float_alignment::eligible(x);
        }
    }
    size_t bytes() const {return values.size()*sizeof(float)+exponents.size()*sizeof(int)+query_ok.size()+key_ok.size();}
    void verify() const {
        for(unsigned t=0;t<pitch;++t)for(unsigned h=0;h<key_heads;++h)for(unsigned i=0;i<width;++i) {
            const uint16_t x=t<tokens?key[(size_t(t)*key_heads+h)*width+i]:0u;
            const size_t cell=(size_t(h)*width+i)*pitch+t;
            if(qrt_sm121_f32_carry::bits(values[cell])!=(uint32_t(x)<<16u)||exponents[cell]!=qrt_sm121_decoded_bf16::exponent(x))
                throw std::runtime_error("CPU QK prepared value or padding changed");
        }
        for(unsigned k=0;k<2u;++k) {
            const auto& flags=k?key_ok:query_ok;const auto* words=k?key:query;
            for(size_t row=0;row<flags.size();++row) {
                bool valid=true;for(unsigned i=0;i<width;++i)valid&=qrt_sm121_float_alignment::eligible(words[row*width+i]);
                if(flags[row]!=unsigned(valid))throw std::runtime_error("CPU QK row flag changed");
            }
        }
    }
};
struct Trace {uint32_t bits[16u][16u]{};uint16_t accepted[16u]{};};
} // namespace qrt_cpu_exact_qk

#if !defined(__HIP_DEVICE_COMPILE__) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#if defined(_WIN32)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#define QRT_CPU_QK_TARGET __attribute__((target("avx512f,avx512cd")))
namespace qrt_cpu_exact_qk {
struct Features {bool osxsave=false,avx=false,avx512f=false,avx512cd=false,zmm=false;
    bool supported() const {return osxsave&&avx&&avx512f&&avx512cd&&zmm;}};
inline void cpuid(unsigned leaf,unsigned subleaf,unsigned (&r)[4]) {
#if defined(_WIN32)
    int a[4];__cpuidex(a,int(leaf),int(subleaf));for(unsigned i=0;i<4u;++i)r[i]=unsigned(a[i]);
#else
    __cpuid_count(leaf,subleaf,r[0],r[1],r[2],r[3]);
#endif
}
__attribute__((target("xsave"))) inline uint64_t xcr0(){return _xgetbv(0);}
inline Features features() {
    Features f;unsigned r[4];cpuid(0,0,r);const unsigned last=r[0];if(last<1u)return f;
    cpuid(1,0,r);f.osxsave=(r[2]&(1u<<27u))!=0;f.avx=(r[2]&(1u<<28u))!=0;
    if(f.osxsave)f.zmm=(xcr0()&0xe6u)==0xe6u;
    if(last>=7u){cpuid(7,0,r);f.avx512f=(r[1]&(1u<<16u))!=0;f.avx512cd=(r[1]&(1u<<28u))!=0;}
    return f;
}
struct Environment {
    unsigned saved;
    Environment():saved(_mm_getcsr()) {
        // Round-to-nearest, gradual FP32 underflow, and masked exceptions.
        // Restore the caller's complete MXCSR on every worker exit.
        _mm_setcsr((saved&~unsigned(0xe040u))|0x1f80u);
    }
    ~Environment(){_mm_setcsr(saved);}
};

// The caller has established CPU/OS support and owns an Environment scope.
// Keys are consecutive SIMD lanes. All masks are per independent score;
// rejected arithmetic never supplies an output or an accepted trace state.
QRT_CPU_QK_TARGET inline uint16_t dot16(const Prepared& p,unsigned query_row,unsigned first_key,
    unsigned count,float* output,Trace* trace=nullptr) {
    const unsigned head=query_row%query_heads,kv=head/8u;
    const uint16_t* q=p.query+size_t(query_row)*width;
    const unsigned live=(1u<<count)-1u;
    unsigned valid=0u;
    for(unsigned i=0;i<count;++i)if(p.query_ok[query_row]&&p.key_ok[size_t(first_key+i)*key_heads+kv])valid|=1u<<i;
    __mmask16 active=__mmask16(valid);
    const __m512i zero=_mm512_setzero_si512(),one=_mm512_set1_epi32(1);
    __m512 carry=_mm512_setzero_ps();
    if(active)for(unsigned base=0;base<width;base+=16u) {
        const __m512i absolute=_mm512_and_si512(_mm512_castps_si512(carry),_mm512_set1_epi32(0x7fffffff));
        const __mmask16 nonzero=_mm512_cmpneq_epi32_mask(absolute,zero);
        __m512i maximum=_mm512_mask_mov_epi32(_mm512_set1_epi32(-133),nonzero,
            _mm512_sub_epi32(_mm512_srli_epi32(absolute,23),_mm512_set1_epi32(127)));
        __m512 products[16];__mmask16 first_negative=0;
#pragma unroll
        for(unsigned i=0;i<16u;++i) {
            const size_t cell=(size_t(kv)*width+base+i)*p.pitch+first_key;
            const __m512 b=_mm512_maskz_loadu_ps(active,p.values.data()+cell);
            const __m512 a=_mm512_set1_ps(qrt_sm121_decoded_bf16::value(q[base+i]));
            products[i]=_mm512_mul_ps(a,b);
            const __m512i exponent=_mm512_add_epi32(_mm512_set1_epi32(qrt_sm121_decoded_bf16::exponent(q[base+i])),
                _mm512_maskz_loadu_epi32(active,p.exponents.data()+cell));
            maximum=_mm512_max_epi32(maximum,exponent);
            if(!i)first_negative=_mm512_cmplt_epi32_mask(_mm512_xor_si512(_mm512_castps_si512(a),_mm512_castps_si512(b)),zero);
        }
        const __mmask16 all_zero=_mm512_cmpeq_epi32_mask(maximum,_mm512_set1_epi32(-133))&~nonzero;
        active&=all_zero|(_mm512_cmpge_epi32_mask(maximum,_mm512_set1_epi32(-101))&_mm512_cmple_epi32_mask(maximum,_mm512_set1_epi32(127)));
        // Inactive lanes get a finite harmless scale, avoiding exceptional
        // conversions; their complete original dots are evaluated below.
        const __m512i safe_max=_mm512_maskz_mov_epi32(active&~all_zero,maximum);
        const __m512 scale=_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(152),safe_max),23));
        __m512i modulo=_mm512_cvttps_epi32(_mm512_maskz_mul_ps(active,carry,scale));
#pragma unroll
        for(unsigned i=0;i<16u;++i)modulo=_mm512_add_epi32(modulo,_mm512_cvttps_epi32(_mm512_maskz_mul_ps(active,products[i],scale)));
        const __mmask16 negative=_mm512_cmp_epu32_mask(modulo,_mm512_set1_epi32(int(qrt_sm121_group16::kMinNegativeModulo)),_MM_CMPINT_GE)&
            (_mm512_cmp_epu32_mask(modulo,_mm512_set1_epi32(int(qrt_sm121_group16::kMaxMagnitude)),_MM_CMPINT_GT)|first_negative);
        const __m512i magnitude=_mm512_mask_sub_epi32(modulo,negative,zero,modulo);
        const __m512i leading=_mm512_lzcnt_epi32(_mm512_or_si512(magnitude,one));
        const __m512i exponent=_mm512_sub_epi32(_mm512_add_epi32(maximum,_mm512_set1_epi32(6)),leading);
        const __mmask16 is_zero=_mm512_cmpeq_epi32_mask(magnitude,zero);
        active&=is_zero|(_mm512_cmpge_epi32_mask(exponent,_mm512_set1_epi32(-126))&_mm512_cmple_epi32_mask(exponent,_mm512_set1_epi32(127)));
        const __m512i mantissa=_mm512_and_si512(_mm512_srli_epi32(_mm512_sllv_epi32(magnitude,leading),8),_mm512_set1_epi32(0x7fffff));
        const __m512i encoded=_mm512_or_si512(mantissa,_mm512_slli_epi32(_mm512_add_epi32(exponent,_mm512_set1_epi32(127)),23));
        carry=_mm512_castsi512_ps(_mm512_maskz_mov_epi32(active&~is_zero,
            _mm512_or_si512(encoded,_mm512_maskz_set1_epi32(negative,int(0x80000000u)))));
        if(trace){_mm512_storeu_si512(trace->bits[base/16u],_mm512_castps_si512(carry));trace->accepted[base/16u]=active;}
        if(!active)break;
    }
    _mm512_mask_storeu_ps(output,__mmask16(live),_mm512_mul_ps(carry,_mm512_set1_ps(0.0625f)));
    const unsigned fallback=live&~unsigned(active);
    for(unsigned i=0;i<count;++i)if(fallback&(1u<<i)) {
        const auto* k=p.key+(size_t(first_key+i)*key_heads+kv)*width;
        output[i]=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,q,k,width)*0.0625f;
    }
    return uint16_t(fallback);
}

class Workers {
    std::mutex mutex;
    std::condition_variable start,done;
    std::vector<std::thread> threads;
    std::function<void(size_t)> work;
    std::atomic<size_t> next{0};
    std::atomic<bool> cancel{false};
    size_t total=0,epoch=0,completed=0;
    bool stopping=false;
    std::exception_ptr failure;
    void loop() {
        Environment environment;size_t observed=0;
        for(;;) {
            std::unique_lock<std::mutex> lock(mutex);
            start.wait(lock,[&]{return stopping||epoch!=observed;});if(stopping)return;
            observed=epoch;lock.unlock();
            try {for(;;){const auto i=next.fetch_add(1,std::memory_order_relaxed);if(i>=total||cancel.load(std::memory_order_relaxed))break;work(i);}}
            catch(...){lock.lock();if(!failure)failure=std::current_exception();cancel.store(true);lock.unlock();}
            lock.lock();if(++completed==threads.size())done.notify_one();
        }
    }
public:
    explicit Workers(unsigned count) {
        if(!features().supported()||!count||count>32u)throw std::invalid_argument("CPU QK worker configuration or ISA unsupported");
        try {for(unsigned i=0;i<count;++i)threads.emplace_back([this]{loop();});}
        catch(...){{std::lock_guard<std::mutex> lock(mutex);stopping=true;}start.notify_all();for(auto& t:threads)t.join();throw;}
    }
    Workers(const Workers&)=delete;
    ~Workers(){{std::lock_guard<std::mutex> lock(mutex);stopping=true;}start.notify_all();for(auto& t:threads)t.join();}
    void run(size_t count,std::function<void(size_t)> function) {
        std::unique_lock<std::mutex> lock(mutex);work=std::move(function);total=count;next=0;completed=0;failure=nullptr;cancel=false;++epoch;start.notify_all();
        const bool timely=done.wait_for(lock,std::chrono::seconds(30),[&]{return completed==threads.size();});
        if(!timely){cancel=true;done.wait(lock,[&]{return completed==threads.size();});throw std::runtime_error("CPU QK slab deadline");}
        work={};if(failure)std::rethrow_exception(failure);
    }
};

inline uint64_t scores(const Prepared& p,Workers& workers,unsigned first,unsigned count,float* output,size_t capacity) {
    if(!output||!count||count>128u||first>=p.tokens||count>p.tokens-first||capacity<size_t(count)*query_heads*(first+count))
        throw std::invalid_argument("CPU QK slab shape");
    constexpr unsigned query_block=8u,key_block=128u;
    const unsigned stride=first+count,key_blocks=(stride+key_block-1u)/key_block;
    std::atomic<uint64_t> fallbacks{0};
    workers.run(size_t((count+query_block-1u)/query_block)*query_heads*key_blocks,[&](size_t job) {
        const unsigned key_begin=unsigned(job%key_blocks)*key_block,head=unsigned(job/key_blocks)%query_heads;
        const unsigned query_begin=unsigned(job/key_blocks/query_heads)*query_block;
        uint64_t local_fallback=0;
        for(unsigned row=query_begin;row<std::min(count,query_begin+query_block);++row) {
            float* out=output+(size_t(row)*query_heads+head)*stride;
            const unsigned end=std::min(stride,key_begin+key_block),causal=first+row+1u;
            for(unsigned k=key_begin;k<end;k+=16u) {
                const unsigned n=std::min(16u,end-k),live=k<causal?std::min(n,causal-k):0u;
                if(live)local_fallback+=unsigned(__builtin_popcount(unsigned(dot16(p,(first+row)*query_heads+head,k,live,out+k))));
                for(unsigned i=live;i<n;++i)out[k+i]=qrt_sm121_float_alignment::from_bits(0xff800000u);
            }
        }
        if(local_fallback)fallbacks.fetch_add(local_fallback,std::memory_order_relaxed);
    });
    return fallbacks.load();
}
} // namespace qrt_cpu_exact_qk
#undef QRT_CPU_QK_TARGET
#endif
