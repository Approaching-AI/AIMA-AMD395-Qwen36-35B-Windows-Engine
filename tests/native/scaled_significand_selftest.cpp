#include "../../native/providers/moe_accumulator/sm121_scaled_significand.h"
#include "../../native/providers/moe_accumulator/sm121_canonical_normalize.h"
#include "float_alignment_cases.h"
#include <cstdio>

namespace scaled = qrt_sm121_scaled_significand;
namespace original = qrt_q1_moe_hawkeye;
uint32_t random_state=0x3957169u;
uint32_t next() { random_state=qrt_float_alignment_cases::random(random_state);return random_state; }

int main() {
    uint64_t pairs=0u;unsigned accepted=0u,fallback=0u;
    for(unsigned word=0u;word<65536u;++word) {
        const unsigned exponent=(word>>7u)&255u;
        const bool normal=!(word&0x7fffu) || (exponent>=1u && exponent<=254u);
        if(scaled::eligible(uint16_t(word))!=normal) return 4;
    }
    // Exhaust all significand pairs/signs and all contributing alignment
    // shifts, plus the zero cutoff and shifts beyond the integer word width.
    for(unsigned a=128u;a<256u;++a) for(unsigned b=128u;b<256u;++b)
    for(unsigned sign=0u;sign<4u;++sign) for(unsigned delta=0u;delta<34u;++delta) {
        const uint16_t x=uint16_t((1u<<7u)|(a&127u)|((sign&1u)?0x8000u:0u));
        const uint16_t y=uint16_t((254u<<7u)|(b&127u)|((sign&2u)?0x8000u:0u));
        const auto pair=scaled::prepare(x,y);const uint32_t magnitude=delta>=32u?0u:(a*b<<11u)>>delta;
        const uint32_t expected=((x^y)&0x8000u)?0u-magnitude:magnitude;
        if(scaled::aligned(pair,pair.exponent+int(delta))!=expected) return 2;
        ++pairs;
    }
    for(unsigned row=0u;row<16384u;++row) {
        original::Value carry{0u,-133,false},expected=carry;
        for(unsigned group=0u;group<128u;++group) {
            uint16_t left[16],right[16];original::Value values[17];values[0]=expected;
            for(unsigned i=0u;i<16u;++i) {
                if(row%4u==0u) {
                    const auto pair=qrt_float_alignment_cases::input(row/4u,group,i);left[i]=pair.left;right[i]=pair.right;
                } else {
                    const uint32_t a=next(),b=next();
                    const unsigned ae=row%4u==1u?1u+a%254u:row%4u==2u?108u+a%25u:250u+a%5u;
                    const unsigned be=row%4u==1u?1u+b%254u:row%4u==2u?4u+b%48u:250u+b%5u;
                    left[i]=uint16_t((a&0x807fu)|(ae<<7u));right[i]=uint16_t((b&0x807fu)|(be<<7u));
                    if(row%16u==11u && i%2u) left[i]=uint16_t(left[i]^0x8000u);
                    if(row%16u==7u && i%3u==0u) {left[i]&=0x8000u;right[i]&=0x8000u;}
                }
                values[i+1u]=original::multiply_bf16(left[i],right[i],-133);
            }
            qrt_sm121_group16::AlignedSum sum;
            if(scaled::sum(carry,left,right,&sum)) ++accepted;
            else {
                ++fallback;uint32_t products[16];
                for(unsigned i=0u;i<16u;++i) products[i]=qrt_sm121_group16::pack_product(values[i+1u]);
                sum=qrt_sm121_group16::sum_packed(carry,products);
            }
            carry=qrt_sm121_canonical::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
            expected=original::group_sum<26,-133>(values,17u);
            if(carry.significand!=expected.significand || carry.exponent!=expected.exponent || carry.negative!=expected.negative) {
                std::printf("DIFF row=%u group=%u expected=%u,%d,%u actual=%u,%d,%u\n",row,group,expected.significand,int(expected.exponent),unsigned(expected.negative),carry.significand,int(carry.exponent),unsigned(carry.negative));return 3;
            }
        }
    }
    std::printf("{\"kind\":\"scaled_significand_cpu\",\"exhaustive_aligned_pairs\":%llu,\"ordered_groups\":2097152,\"canonical_value_mismatches\":0,\"scalar_groups\":%u,\"fallback_groups\":%u,\"inference_acceptance\":false}\n",(unsigned long long)pairs,accepted,fallback);
    return !accepted || !fallback;
}
