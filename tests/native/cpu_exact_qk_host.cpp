#include "../../native/providers/ck_fmha/cpu_exact_qk.h"
#include <cstdio>
#include <array>

namespace cpu_qk_tests {
namespace cpu=qrt_cpu_exact_qk;
namespace original=qrt_q1_moe_hawkeye;
uint32_t bits(float value){uint32_t result;std::memcpy(&result,&value,4u);return result;}
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
uint32_t random(uint32_t x){x^=x<<13u;x^=x>>17u;x^=x<<5u;return x;}
uint16_t operand(unsigned row,unsigned feature,unsigned mode,unsigned salt) {
    const auto x=random(0x3958192u+row*7919u+feature*997u+salt*11939u);
    if(mode==0u)return uint16_t(x&0x8000u);
    if(mode==1u)return 0x3fffu;
    if(mode==2u)return uint16_t(0x3fffu|((feature+salt)%2u?0x8000u:0u));
    if(mode==3u)return uint16_t((x&0x807fu)|((120u+x%15u)<<7u));
    if(mode==4u)return uint16_t((x&0x807fu)|((64u+x%127u)<<7u));
    if(mode==5u)return uint16_t((x&0x807fu)|(64u<<7u));
    if(mode==6u)return uint16_t((x&0x807fu)|(190u<<7u));
    if(mode==8u||mode==9u) {
        if(feature<16u)return feature?0u:uint16_t(0x3f80u|(!salt&&mode==9u?0x8000u:0u));
        return uint16_t(0x3fffu|(!salt&&mode==9u?0x8000u:0u));
    }
    return uint16_t(x);
}
void portable() {
    uint64_t cells=0;
    for(unsigned n:{1u,3u,17u,33u,129u})for(unsigned padding:{0u,16u}) {
        std::vector<uint16_t> q(size_t(n)*4096u),k(size_t(n)*512u);
        for(size_t i=0;i<q.size();++i)q[i]=uint16_t(i);
        for(size_t i=0;i<k.size();++i)k[i]=uint16_t(i*7919u);
        const auto before_q=q,before_k=k;
        cpu::Prepared p(q.data(),k.data(),n,padding);p.verify();
        require(q==before_q&&k==before_k,"preparation mutated originals");cells+=p.values.size();
    }
    bool rejected=false;uint16_t word=0;
    try {cpu::Prepared p(&word,&word,0u);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"zero shape accepted");
    std::printf("{\"kind\":\"cpu_exact_qk_preparation\",\"shapes\":10,\"prepared_cells\":%llu,\"all_bf16_encodings\":true,\"padding_and_flags_pass\":true,\"immutable_inputs\":true}\n",(unsigned long long)cells);
}
#if !defined(__HIP_DEVICE_COMPILE__) && (defined(__x86_64__) || defined(_M_X64))
void native() {
    const auto f=cpu::features();require(f.supported(),"AVX512F/CD or OS ZMM state unavailable");
    cpu::Workers serial(1u),parallel(16u);uint64_t comparisons=0,fallbacks=0,accepted_states=0,rejected_states=0;
    unsigned cases=0;
    for(unsigned mode=0;mode<10u;++mode)for(unsigned n:{1u,17u,129u}) {
        std::vector<uint16_t> q(size_t(n)*4096u),k(size_t(n)*512u);
        for(unsigned row=0;row<n*16u;++row)for(unsigned c=0;c<256u;++c)q[size_t(row)*256u+c]=operand(row,c,mode,0u);
        for(unsigned row=0;row<n*2u;++row)for(unsigned c=0;c<256u;++c)k[size_t(row)*256u+c]=operand(row,c,mode,1u);
        const auto before_q=q,before_k=k;cpu::Prepared p(q.data(),k.data(),n);
        for(unsigned first=0;first<n;first+=37u) {
            const unsigned count=std::min(37u,n-first),stride=first+count;
            const size_t cells=size_t(count)*16u*stride;
            std::vector<float> a(cells+128u,qrt_sm121_float_alignment::from_bits(0xa5a5a5a5u)),b=a;
            const auto first_fallback=cpu::scores(p,serial,first,count,a.data()+64u,cells);
            fallbacks+=first_fallback;
            const auto second=cpu::scores(p,parallel,first,count,b.data()+64u,cells);
            require(first_fallback==second&&!std::memcmp(a.data(),b.data(),a.size()*4u),"serial/parallel CPU scores differ");
            for(size_t i=0;i<64u;++i)require(bits(a[i])==0xa5a5a5a5u&&bits(a[cells+64u+i])==0xa5a5a5a5u,"CPU score redzone");
            for(unsigned row=0;row<count;++row)for(unsigned head=0;head<16u;++head)for(unsigned key=0;key<stride;++key) {
                const uint32_t expected=key>first+row?0xff800000u:bits(original::accumulate_bf16_hopper_blackwell(0.0f,
                    q.data()+(size_t(first+row)*16u+head)*256u,k.data()+(size_t(key)*2u+head/8u)*256u,256u)*0.0625f);
                const auto actual=bits(a[64u+(size_t(row)*16u+head)*stride+key]);
                if(actual!=expected){std::fprintf(stderr,"CPU_QK_DIFF mode=%u tokens=%u query=%u head=%u key=%u expected=%08x actual=%08x\n",mode,n,first+row,head,key,expected,actual);throw std::runtime_error("CPU SIMD differs from original dot");}++comparisons;
            }
        }
        if(n==129u) {
            cpu::Environment environment;
            for(unsigned sample=0;sample<64u;++sample) {
                const unsigned row=(sample*31u)%(n*16u),first=(sample*16u)%128u,count=std::min(16u,n-first);
                cpu::Trace trace;std::array<float,32> output;output.fill(123.0f);
                cpu::dot16(p,row,first,count,output.data(),&trace);
                for(unsigned lane=count;lane<32u;++lane)require(output[lane]==123.0f,"CPU SIMD tail store");
                for(unsigned lane=0;lane<count;++lane) {
                    original::Value carry{0u,-133,false};
                    for(unsigned group=0;group<16u;++group) {
                        original::Value values[17];values[0]=carry;
                        for(unsigned i=0;i<16u;++i)values[i+1u]=original::multiply_bf16(
                            q[size_t(row)*256u+group*16u+i],k[(size_t(first+lane)*2u+row%16u/8u)*256u+group*16u+i],-133);
                        carry=original::group_sum<26,-133>(values,17u);
                        if(trace.accepted[group]&(1u<<lane)) {
                            ++accepted_states;require(trace.bits[group][lane]==bits(original::value_to_float(qrt_sm121_group16::finish_accumulator(carry))),"CPU SIMD raw K16 state");
                        }else ++rejected_states;
                    }
                }
            }
        }
        require(q==before_q&&k==before_k,"CPU SIMD input changed");p.verify();++cases;
    }
    bool failed=false;
    try {parallel.run(100u,[](size_t i){if(i==7u)throw std::runtime_error("injected worker failure");});}catch(const std::runtime_error&){failed=true;}
    require(failed,"worker exception lost");
    std::array<std::atomic<unsigned>,100> visits{};for(auto& v:visits)v=0;
    parallel.run(visits.size(),[&](size_t i){++visits[i];});
    for(auto& v:visits)require(v==1u,"worker recovery ownership");
    parallel.run(0u,[](size_t){throw std::runtime_error("empty work executed");});
    require(fallbacks&&accepted_states&&rejected_states,"CPU safety paths not exercised");
    std::printf("{\"kind\":\"cpu_exact_qk_safety\",\"cases\":%u,\"score_comparisons\":%llu,\"fallback_scores\":%llu,\"accepted_raw_k16_states\":%llu,\"rejected_raw_k16_states\":%llu,\"raw_mismatches\":0,\"serial_parallel_agree\":true,\"worker_failure_recovery_pass\":true,\"redzones_and_tails_pass\":true,\"immutable_inputs\":true,\"cpuid_avx512f\":true,\"cpuid_avx512cd\":true,\"os_zmm_state\":true}\n",cases,(unsigned long long)comparisons,(unsigned long long)fallbacks,(unsigned long long)accepted_states,(unsigned long long)rejected_states);
}
#endif
} // namespace cpu_qk_tests
#ifndef QRT_CPU_QK_NO_MAIN
int main()try {
    cpu_qk_tests::portable();
#if !defined(__HIP_DEVICE_COMPILE__) && (defined(__x86_64__) || defined(_M_X64))
    cpu_qk_tests::native();
#endif
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"cpu_qk_error=%s\n",e.what());return 2;}
#endif
