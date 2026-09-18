#include "../../native/providers/moe_accumulator/sm121_narrow_qk_plan.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>
namespace narrow=qrt_sm121_narrow_qk_plan;
namespace original=qrt_q1_moe_hawkeye;
uint32_t random_state=0x62e137du;
uint32_t random_word(){random_state^=random_state<<13u;random_state^=random_state>>17u;random_state^=random_state<<5u;return random_state;}
int main(){
    uint64_t accepted=0u,rejected=0u,forced=0u,groups=0u;
    for(unsigned test=0u;test<32768u;++test){
        original::Value previous{0u,-133,false};float carry=0.0f;
        for(unsigned g=0u;g<16u;++g){
            uint16_t a[16],b[16];const unsigned ap=126u+random_word()%34u,bp=126u+random_word()%34u;
            for(unsigned i=0u;i<16u;++i){
                a[i]=uint16_t((random_word()&0x807fu)|((ap-random_word()%32u)<<7u));
                b[i]=uint16_t((random_word()&0x807fu)|((bp-random_word()%32u)<<7u));
                if(test%7u==0u){a[i]=uint16_t(95u<<7u|127u);b[i]=uint16_t(a[i]|(i&1u?0x8000u:0u));}
                if(test%7u==1u)a[i]=b[i]=uint16_t(159u<<7u|127u);
                if(test%7u==2u)a[i]&=0x8000u;
                if(test%7u==3u&&g>7u)b[i]&=0x8000u;
            }
            const auto am=narrow::byte::prepare(a),bm=narrow::byte::prepare(b);assert(am.maximum>=0&&bm.maximum>=0);
            original::Value terms[17];terms[0]=previous;
            for(unsigned i=0u;i<16u;++i)terms[i+1u]=original::multiply_bf16(a[i],b[i],-133);
            previous=original::group_sum<26,-133>(terms,17u);
            const float expected=original::value_to_float(previous);
            const int exact=narrow::predicted_exponent(carry);
            for(int bias:{-4,-1,0,1,4,645}){
                const auto prepared=narrow::prepare(a,b,am,bm,exact+bias);
                float output=12345.0f;const bool used=narrow::apply(carry,prepared,&output);
                if(used){assert(narrow::byte::f32::bits(output)==narrow::byte::f32::bits(expected));++accepted;}
                else{assert(output==12345.0f);++rejected;}
                if(!bias)assert(used);
            }
            const auto correct=narrow::prepare(a,b,am,bm,exact);
            const auto wrong=narrow::plan::encode(correct.modulo,narrow::plan::maximum(correct)+1,
                narrow::plan::product_maximum(correct),(correct.control&(1u<<20u))!=0u);
            float untouched=-123.0f;
            assert(!narrow::apply(carry,wrong,&untouched)&&untouched==-123.0f);++forced;
            assert(!narrow::apply(carry,correct,nullptr));
            carry=expected;++groups;
        }
    }
    for(uint32_t bits:{1u,0x007fffffu,0x7f800000u,0x7fc12345u,0xff800000u})
        assert(narrow::predicted_exponent(narrow::byte::f32::alignment::from_bits(bits))==512);
    assert(accepted&&rejected&&forced==groups);
    std::printf("{\"kind\":\"narrow_qk_plan_host\",\"ordered_groups\":%llu,\"predictions_checked\":%llu,\"accepted\":%llu,\"rejected\":%llu,\"forced_wrong_alignment_rejected\":%llu,\"raw_carry_mismatches\":0,\"rejected_output_unchanged\":true}\n",
        (unsigned long long)groups,(unsigned long long)(accepted+rejected),(unsigned long long)accepted,(unsigned long long)rejected,(unsigned long long)forced);
}
