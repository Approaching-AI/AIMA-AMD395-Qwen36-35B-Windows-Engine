#include "../../native/providers/moe_accumulator/sm121_narrow_f32_carry.h"
#include <array>
#include <cassert>
#include <cstdio>
namespace narrow=qrt_sm121_narrow_f32_carry;
namespace original=qrt_q1_moe_hawkeye;
using original::Value;
uint32_t state=0x71be249du;
uint32_t random_word(){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
uint16_t operand(){const auto x=random_word();return (x&31u)==0u?uint16_t(x&0x8000u):uint16_t((x&0x807fu)|((95u+(x>>16u)%65u)<<7u));}
uint64_t groups=0u,products=0u;
float checked(float carry,const Value& canonical,const uint16_t* a,const uint16_t* b,Value* result){
    qrt_sm121_float_alignment::Group group;Value terms[17];terms[0]=canonical;
    for(unsigned i=0u;i<16u;++i){assert(narrow::eligible(a[i])&&narrow::eligible(b[i]));
        group.set(i,a[i],b[i]);terms[i+1u]=original::multiply_bf16(a[i],b[i],-133);}
    *result=original::group_sum<26,-133>(terms,17u);
    const float actual=narrow::accumulate(carry,group),expected=original::value_to_float(*result);
    assert(qrt_sm121_f32_carry::bits(actual)==qrt_sm121_f32_carry::bits(expected));
    if(result->significand)assert(result->exponent>=-89&&result->exponent<=73);
    ++groups;products+=16u;return actual;
}
int main(){
    unsigned admitted=0u;
    for(unsigned bits=0u;bits<65536u;++bits){
        const unsigned exponent=(bits>>7u)&255u;
        const bool expected=!(bits&0x7fffu)||(exponent>=95u&&exponent<=159u);
        assert(narrow::eligible(uint16_t(bits))==expected);if(!expected)continue;++admitted;
        float carry=0.0f;Value canonical{0u,-133,false};
        for(unsigned g=0u;g<16u;++g){
            uint16_t a[16],b[16];for(unsigned i=0u;i<16u;++i){a[i]=operand();b[i]=operand();}
            a[g%16u]=uint16_t(bits);b[(g+7u)%16u]=uint16_t(bits);
            if((bits&7u)==0u && g>=8u)for(unsigned i=0u;i<16u;++i)a[i]=uint16_t(i&1u?0x8000u:0u);
            carry=checked(carry,canonical,a,b,&canonical);
        }
    }
    assert(admitted==16642u);
    // Explicit minimum-scale cancellation, maximum growth and signed zeros.
    for(unsigned mode=0u;mode<8u;++mode){
        float carry=0.0f;Value canonical{0u,-133,false};
        for(unsigned g=0u;g<16u;++g){
            uint16_t a[16],b[16];for(unsigned i=0u;i<16u;++i){
                a[i]=uint16_t(((mode&1u)?95u:159u)<<7u|127u);
                b[i]=uint16_t(a[i]|((mode&2u)&&(i&1u)?0x8000u:0u));
                if(mode&4u){if(g)b[i]=uint16_t(i&1u?0x8000u:0u);else if(i==15u)b[i]^=1u;}
            }
            carry=checked(carry,canonical,a,b,&canonical);
        }
    }
    // The conservative carried-domain lower endpoint survives zero groups.
    for(int exponent:{-89,-88,0,72,73})for(bool negative:{false,true}){
        Value canonical{0x800001u,int16_t(exponent),negative};float carry=original::value_to_float(canonical);
        uint16_t zero[16]{};carry=checked(carry,canonical,zero,zero,&canonical);(void)carry;
    }
    std::printf("{\"kind\":\"narrow_domain_qk_host\",\"encoding_predicates\":65536,\"admitted_encodings\":%u,\"ordered_groups\":%llu,\"original_products\":%llu,\"raw_carry_mismatches\":0,\"zero_tail_and_domain_extremes\":true}\n",admitted,(unsigned long long)groups,(unsigned long long)products);
}
