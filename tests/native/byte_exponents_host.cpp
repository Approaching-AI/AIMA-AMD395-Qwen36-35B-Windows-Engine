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
    uint64_t packed_minimum_checks=0u;
    for(unsigned a=0;a<=126u;++a)for(unsigned b=0;b<=126u;++b){
        uint32_t x=0u,y=0u,expected=0u;
        for(unsigned i=0;i<4u;++i){
            const unsigned ai=i&1u?126u-a:a,bi=i&2u?126u-b:b;
            x|=ai<<(i*8u);y|=bi<<(i*8u);expected|=std::min(ai,bi)<<(i*8u);
        }
        assert(byte::minimum_bytes(x,y)==expected);++packed_minimum_checks;
    }
    for(unsigned n=0;n<1048576u;++n){
        uint32_t a=0u,b=0u,expected=0u;
        for(unsigned i=0;i<4u;++i){unsigned x=random_word()%127u,y=random_word()%127u;
            a|=x<<(i*8u);b|=y<<(i*8u);expected|=std::min(x,y)<<(i*8u);}
        assert(byte::minimum_bytes(a,b)==expected);++packed_minimum_checks;
    }
    unsigned admitted=0u;
    for(unsigned bits=0;bits<65536u;++bits){
        uint16_t a[16],b[16];std::fill(a,a+16u,uint16_t(bits));std::fill(b,b+16u,uint16_t(bits));
        const auto m=checked_metadata(a);if(m.maximum<0)continue;++admitted;
        original::Value previous{0u,-133,false};float carry=0.0f;
        for(unsigned g=0;g<16u;++g){
            if(g==8u)for(unsigned i=0;i<16u;++i)b[i]=uint16_t(i&1u?0x8000u:0u);
            carry=checked_group(carry,previous,a,b);
        }
    }
    assert(admitted==16642u);
    for(unsigned n=0;n<16384u;++n){
        original::Value previous{0u,-133,false};float carry=0.0f;
        for(unsigned g=0;g<16u;++g){
            uint16_t a[16],b[16];const unsigned ap=126u+random_word()%34u,bp=126u+random_word()%34u;
            for(unsigned i=0;i<16u;++i){
                a[i]=uint16_t((random_word()&0x807fu)|((ap-random_word()%32u)<<7u));
                b[i]=uint16_t((random_word()&0x807fu)|((bp-random_word()%32u)<<7u));
                if(n%5u==0u && i%2u==0u)a[i]&=0x8000u;
                if(n%5u==0u && i%2u==1u)b[i]&=0x8000u;
                if(n%5u==1u)b[i]=uint16_t(a[i]^(i&1u?0x8000u:0u));
            }
            carry=checked_group(carry,previous,a,b);
        }
    }
    for(unsigned low:{95u,127u,128u})for(unsigned span:{31u,32u}){
        uint16_t words[16]{};words[0]=uint16_t(low<<7u);words[15]=uint16_t((low+span)<<7u);
        const auto m=checked_metadata(words);assert((m.maximum>=0)==(span==31u));
    }
    std::printf("{\"kind\":\"byte_exponents_host\",\"packed_minimum_checks\":%llu,\"encoding_predicates\":65536,\"admitted_encodings\":%u,\"metadata_checks\":%llu,\"exact_maximum_checks\":%llu,\"ordered_k16_groups\":%llu,\"raw_carry_mismatches\":0,\"domain_and_span_rejections\":true}\n",
        (unsigned long long)packed_minimum_checks,admitted,(unsigned long long)metadata_checks,(unsigned long long)maximum_checks,(unsigned long long)groups);
}
