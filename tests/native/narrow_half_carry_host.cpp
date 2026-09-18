#include "../../native/providers/moe_accumulator/sm121_narrow_half_carry.h"
#include "narrow_half_cases.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>
namespace candidate=qrt_sm121_narrow_half_carry;
namespace original=qrt_q1_moe_hawkeye;
bool independent_admission(const uint16_t* input){
    unsigned low=255u,high=0u;
    for(unsigned i=0u;i<16u;++i)if(input[i]&0x7fffu){
        const unsigned exponent=(input[i]>>7u)&255u;
        if(exponent<95u||exponent>159u)return false;
        low=std::min(low,exponent);high=std::max(high,exponent);
    }
    return !high||high-low<=29u;
}
int main(){
    uint64_t dots=0u,fast=0u,rejected=0u,groups=0u,encoded=0u;
    for(unsigned value=0u;value<65536u;++value){
        uint16_t row[16];std::fill(row,row+16u,uint16_t(value));
        const auto packed=candidate::half::prepare(row);
        assert(candidate::eligible(packed)==independent_admission(row));
    }
    const unsigned widths[]={16u,32u,256u,2048u,4096u,8192u};
    for(unsigned width:widths)for(unsigned row=0u;row<512u;++row){
        std::vector<uint16_t>a(width),b(width);std::vector<candidate::Row>pa(width/16u),pb(width/16u);bool admitted=true;
        for(unsigned g=0u;g<width/16u;++g){
            for(unsigned i=0u;i<16u;++i){const auto p=qrt_narrow_half_cases::input(row,g,i);a[g*16u+i]=p.x;b[g*16u+i]=p.y;}
            pa[g]=candidate::half::prepare(a.data()+g*16u);pb[g]=candidate::half::prepare(b.data()+g*16u);
            for(unsigned side=0u;side<2u;++side){const auto& p=side?pb[g]:pa[g];const uint16_t* raw=(side?b:a).data()+g*16u;
                const bool okay=independent_admission(raw);assert(candidate::eligible(p)==okay);admitted&=okay;
                for(unsigned i=0u;i<16u;++i){assert(candidate::half::original(p,i)==raw[i]);++encoded;}
            }
        }
        ++dots;if(!admitted){++rejected;continue;}++fast;
        original::Value carry{0u,-133,false};float actual=0.0f;
        for(unsigned g=0u;g<width/16u;++g){
            original::Value terms[17];terms[0]=carry;
            for(unsigned i=0u;i<16u;++i)terms[i+1u]=original::multiply_bf16(a[g*16u+i],b[g*16u+i],-133);
            carry=original::group_sum<26,-133>(terms,17u);actual=candidate::accumulate(actual,pa[g],pb[g]);
            assert(candidate::f32::bits(actual)==candidate::f32::bits(original::value_to_float(carry)));
            const auto absolute=candidate::f32::bits(actual)&0x7fffffffu;
            assert(!absolute||(absolute>=uint32_t(127-89)<<23u&&absolute<uint32_t(127+79)<<23u));++groups;
        }
    }
    assert(fast&&rejected);
    std::printf("{\"kind\":\"narrow_half_carry_host\",\"widths\":[16,32,256,2048,4096,8192],\"dots\":%llu,\"admitted_dots\":%llu,\"rejected_dots\":%llu,\"ordered_k16_groups\":%llu,\"lossless_word_checks\":%llu,\"all_bf16_encodings_classified\":65536,\"raw_carry_mismatches\":0,\"wide_domain_range_checks\":true}\n",(unsigned long long)dots,(unsigned long long)fast,(unsigned long long)rejected,(unsigned long long)groups,(unsigned long long)encoded);
}
