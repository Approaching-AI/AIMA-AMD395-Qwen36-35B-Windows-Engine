#include "../../native/providers/moe_accumulator/sm121_byte_exponents.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
namespace byte=qrt_sm121_byte_exponents;
namespace original=qrt_q1_moe_hawkeye;
uint32_t state=0x5182395u;
uint32_t random_word(){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
uint64_t metadata_checks=0u,maximum_checks=0u,groups=0u;
byte::Metadata checked_metadata(const uint16_t* words){
    auto actual=byte::prepare(words);int high=0,low=255;bool valid=true;
    for(unsigned i=0;i<16u;++i){
        unsigned e=(words[i]>>7u)&255u;
        valid&=!(words[i]&0x7fffu)||(e>=95u&&e<=159u);
        if(words[i]&0x7fffu){high=std::max(high,int(e));low=std::min(low,int(e));}
    }
    valid&=high==0||high-low<=31;
    assert(actual.maximum==(valid?high:-1));
    if(valid)for(unsigned i=0;i<16u;++i){
        unsigned expected=(words[i]&0x7fffu)?unsigned(high)-((words[i]>>7u)&255u):63u;
        assert(((actual.deficits[i/4u]>>(8u*(i%4u)))&255u)==expected);
    }
    ++metadata_checks;return actual;
}
float checked_group(float carry,original::Value& previous,const uint16_t* a,const uint16_t* b){
    const auto am=checked_metadata(a),bm=checked_metadata(b);assert(am.maximum>=0&&bm.maximum>=0);
    int maximum=-133;original::Value terms[17];terms[0]=previous;
    for(unsigned i=0;i<16u;++i){
        terms[i+1u]=original::multiply_bf16(a[i],b[i],-133);
        if(terms[i+1u].significand)maximum=std::max(maximum,int(terms[i+1u].exponent));
    }
    assert(byte::product_maximum(am,bm)==maximum);++maximum_checks;
    previous=original::group_sum<26,-133>(terms,17u);
    const float actual=byte::accumulate(carry,a,b,am,bm),expected=original::value_to_float(previous);
    assert(byte::f32::bits(actual)==byte::f32::bits(expected));++groups;return actual;
}
int main(){
    uint64_t dots=0u;
    const unsigned widths[]={16u,32u,256u,2048u,4096u,8192u};
    for(unsigned width:widths)for(unsigned n=0u;n<512u;++n){
        original::Value previous{0u,-133,false};float carry=0.0f;
        for(unsigned g=0u;g<width/16u;++g){
            uint16_t a[16],b[16];const unsigned ap=126u+random_word()%34u,bp=126u+random_word()%34u;
            for(unsigned i=0u;i<16u;++i){
                a[i]=uint16_t((random_word()&0x807fu)|((ap-random_word()%32u)<<7u));
                b[i]=uint16_t((random_word()&0x807fu)|((bp-random_word()%32u)<<7u));
                switch(n%8u){
                case 0u:a[i]=b[i]=uint16_t((95u<<7u)|127u);break;
                case 1u:a[i]=b[i]=uint16_t((159u<<7u)|127u);break;
                case 2u:a[i]=uint16_t((159u<<7u)|127u);b[i]=uint16_t(a[i]^((g&1u)?0x8000u:0u));break;
                case 3u:if(i&1u)a[i]&=0x8000u;else b[i]&=0x8000u;break;
                case 4u:a[i]=b[i]=uint16_t((95u<<7u)|127u);if(g&1u)b[i]^=0x8000u;break;
                case 5u:if(g>width/32u)a[i]&=0x8000u;break;
                default:break;
                }
            }
            carry=checked_group(carry,previous,a,b);
            const auto absolute=byte::f32::bits(carry)&0x7fffffffu;
            assert(!absolute||(absolute>=uint32_t(127-89)<<23u&&absolute<uint32_t(127+79)<<23u));
        }
        ++dots;
    }
    std::printf("{\"kind\":\"wide_byte_carry_host\",\"widths\":[16,32,256,2048,4096,8192],\"dots\":%llu,\"ordered_k16_groups\":%llu,\"metadata_checks\":%llu,\"raw_carry_mismatches\":0,\"wide_domain_range_checks\":true}\n",(unsigned long long)dots,(unsigned long long)groups,(unsigned long long)metadata_checks);
    return 0;
}
