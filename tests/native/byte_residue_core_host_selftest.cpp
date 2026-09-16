#include "../../native/providers/moe_accumulator/sm121_byte_residue_core.h"
#include "../../native/providers/moe_accumulator/sm121_byte_residue4_core.h"
#include "../../native/providers/moe_accumulator/sm121_canonical_normalize.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
namespace core=qrt_sm121_byte_residue_core;
uint32_t state=0x3958b17u;
uint32_t random_word(){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
double half(uint16_t x){const unsigned e=(x>>10u)&31u;return e?((x&32768u)?-1.0:1.0)*std::ldexp(double(1024u+(x&1023u)),int(e)-25):0.0;}
int independent_core(uint16_t x,int unit){if(unit<0 || !(x&0x7fffu))return 0;const int m=int(std::ldexp(double(128u+(x&127u)),int((x>>7u)&255u)-unit));return x&32768u?-m:m;}
template<unsigned Headroom> void run(){
    constexpr int64_t maximum_dot=int64_t(16)*(255u<<Headroom)*(255u<<Headroom);
    size_t encodings=0u,recoveries=0u,groups=0u,accepted=0u,declined=0u,remainders=0u;
    for(unsigned pattern=0u;pattern<65536u;++pattern){
        core::Row row{};int maximum=0;bool valid=true;uint32_t nonzero=0u,exceptions=0u;
        for(unsigned i=0u;i<16u;++i){const uint16_t x=uint16_t(pattern+(i&7u)*8192u);row.original[i]=x;if(x&0x7fffu){nonzero|=1u<<i;const int e=int((x>>7u)&255u);valid=valid&&e&&e!=255;maximum=maximum>e?maximum:e;}}
        const auto before=row;core::prepare<Headroom>(row);
        if constexpr(Headroom==4u){auto narrow=before;qrt_sm121_byte_residue4::prepare(narrow);assert(!std::memcmp(&row,&narrow,sizeof(row)));}
        const int unit=!valid?-1:!nonzero?127:maximum>int(Headroom+1u)?maximum-int(Headroom):1;
        assert(row.unit==unit && row.nonzero==nonzero && !std::memcmp(row.original,before.original,32u));
        for(unsigned i=0u;i<16u;++i){
            const int x=independent_core(row.original[i],unit);assert(std::abs(x)<=int(255u<<Headroom) && half(row.half[i])==x);
            assert(((uint32_t(row.low[i/4u])>>((i&3u)*8u))&255u)==(uint32_t(x)&255u));
            // Zero uses the existing saturated31-bit trailing count. All
            // alignment masks under test have shifts below30.
            unsigned trailing=31u;if(x){trailing=0u;for(unsigned m=unsigned(x<0?-x:x);!(m&1u);m>>=1u)++trailing;}
            assert(((row.trailing[i/4u]>>((i&3u)*8u))&255u)==trailing);
            if(unit>=0 && (row.original[i]&0x7fffu)){
                const double original=std::ldexp(double(128u+(row.original[i]&127u)),int((row.original[i]>>7u)&255u)-unit);
                if(original!=std::abs(x))exceptions|=1u<<i;
            }
            ++encodings;
        }
        assert(row.exceptions==exceptions);
    }
    for(unsigned low=0u;low<256u;++low)for(int64_t high:{int64_t(-65536),int64_t(-256),int64_t(0),int64_t(256),int64_t(16777216),maximum_dot-512,-maximum_dot}){
        const int64_t expected=high+low;if(expected < -maximum_dot || expected>maximum_dot)continue;
        for(double delta:{-128.0,-127.5,-64.0,-0.5,0.0,0.5,64.0,127.5,128.0}){
            const float approximate=float(double(expected)+delta);if(std::abs(double(approximate)-double(expected))>=128.0)continue;
            int64_t result=0x3958192;assert(core::recover<Headroom>(approximate,uint32_t(expected),&result) && result==expected);++recoveries;
            if constexpr(Headroom==4u){int32_t narrow=123;assert(qrt_sm121_byte_residue4::recover(approximate,uint32_t(expected),&narrow) && narrow==expected);}
        }
    }
    for(float x:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),float(maximum_dot+1024),-float(maximum_dot+1024),128.0f,-128.0f}){int64_t result=123;assert(!core::recover<Headroom>(x,0u,&result) && result==123);}
    core::Value previous{0u,-133,false};
    for(unsigned sample=0u;sample<262144u;++sample){
        core::Row a{},b{};core::Value terms[17];
        const unsigned ae=1u+random_word()%220u,be=1u+random_word()%220u,spread=1u+sample%34u;
        for(unsigned i=0u;i<16u;++i){
            a.original[i]=uint16_t((random_word()&0x807fu)|((ae+random_word()%spread)<<7u));
            b.original[i]=uint16_t((random_word()&0x807fu)|((be+random_word()%spread)<<7u));
            if(sample%11u==0u && i%3u==0u)a.original[i]=0u;
            if(sample%13u==0u && i%3u==1u)b.original[i]=0x8000u;
            if(sample%17u==0u){a.original[i]=uint16_t((random_word()&0x807fu)|((ae+i%9u)<<7u));b.original[i]=uint16_t((random_word()&0x807fu)|((be+8u-i%9u)<<7u));}
            if(sample%19u==0u && i==2u)a.original[i]=1u;
            if(sample%23u==0u && i==3u)b.original[i]=0x7fc1u;
            if(sample%29u==0u){a.original[i]=uint16_t(0x3fffu|((sample&1u)<<15u));b.original[i]=0x3fffu;}
            if(sample%31u==0u){a.original[i]=uint16_t((0x3f81u+(i/2u%4u)*128u)|((i&1u)<<15u));b.original[i]=0x3f85u;}
            if(sample%37u==0u){a.original[i]=i%2u?0x8000u:0x7f7fu;b.original[i]=i%2u?0x7f7fu:0u;}
            terms[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(a.original[i],b.original[i],-133);
        }
        core::prepare<Headroom>(a);core::prepare<Headroom>(b);const auto before_a=a,before_b=b;
        int64_t mathematical=0;uint32_t low_product=0u;
        for(unsigned i=0u;i<16u;++i){const int x=independent_core(a.original[i],a.unit),y=independent_core(b.original[i],b.unit);mathematical+=int64_t(x)*y;low_product+=(uint32_t(x)&255u)*(uint32_t(y)&255u);}
        assert((low_product&255u)==(uint32_t(mathematical)&255u));
        for(unsigned shift=1u;shift<30u;++shift){
            uint32_t expected=0u;
            for(unsigned i=0u;i<16u;++i){const int64_t p=int64_t(independent_core(a.original[i],a.unit))*independent_core(b.original[i],b.unit);if(p%(int64_t(1)<<shift))expected|=1u<<i;}
            assert(core::remainder_mask(a,b,shift)==expected);++remainders;
        }
        terms[0]={(random_word()&0x7fffffu)|0x800000u,int16_t(a.maximum+b.maximum-254+int(sample%67u)-26),bool(sample&1u)};
        if(sample%3u==0u || sample%31u==0u || sample%37u==0u)terms[0]={0u,-133,false};
        if(sample%4u==1u)terms[0]=previous;
        previous=qrt_q1_moe_hawkeye::group_sum<26,-133>(terms,17u);
        core::AlignedSum actual{{0xdeadbeefu,true},123};
        const bool success=Headroom==4u?qrt_sm121_byte_residue4::sum(terms[0],a,b,int32_t(mathematical),&actual):core::sum(terms[0],a,b,mathematical,&actual);
        if(success){
            const auto value=qrt_sm121_canonical::normalize(actual.value.magnitude,actual.value.negative,actual.max_exponent);
            assert(value.significand==previous.significand && value.exponent==previous.exponent && value.negative==previous.negative);++accepted;
        }else{assert(actual.value.magnitude==0xdeadbeefu && actual.value.negative && actual.max_exponent==123);++declined;}
        assert(!std::memcmp(&a,&before_a,sizeof(a)) && !std::memcmp(&b,&before_b,sizeof(b)));++groups;
    }
    assert(accepted>100000u && declined>1000u);
    std::printf("{\"kind\":\"byte_residue_core_host\",\"headroom\":%u,\"exact_encodings\":%zu,\"conditional_recovery_checks\":%zu,\"remainder_mask_checks\":%zu,\"canonical_groups\":%zu,\"accepted_groups\":%zu,\"declined_groups\":%zu,\"mismatches\":0,\"row_bytes\":116,\"immutable_inputs\":true,\"hardware_error_bound_proven\":false}\n",Headroom,encodings,recoveries,remainders,groups,accepted,declined);
}
int main(){run<4u>();run<5u>();run<6u>();}
