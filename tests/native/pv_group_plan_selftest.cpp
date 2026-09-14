#include <cassert>
#include <cstdio>
#include <cstring>
#include "sm121_pv_group_plan.h"
using qrt_q1_moe_hawkeye::Value;
namespace plan = qrt_sm121_pv_group_plan;
unsigned seed = 0x8192395u;
unsigned random_word() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; }
uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4u); return u; }
int main() {
    unsigned skipped[3]{},eligible = 0u;
    for (unsigned test = 0u; test < 500000u; ++test) {
        uint16_t p[16],v[16],before_p[16],before_v[16];
        const unsigned mode = test % 5u;
        for (unsigned i = 0u; i < 16u; ++i) {
            p[i] = uint16_t(random_word()); v[i] = uint16_t(random_word());
            if (mode == 1u) {
                p[i] = uint16_t((random_word() & 0x807fu) | ((64u + random_word() % 127u) << 7u));
                v[i] = uint16_t((random_word() & 0x807fu) | ((64u + random_word() % 127u) << 7u));
            }
            if (mode == 2u) p[i] = uint16_t((i & 1u) << 15u);
            if (mode == 3u) v[i] = uint16_t((i & 1u) << 15u);
            if (mode == 4u) { p[i] = 0x3fffu; v[i] = uint16_t(0x3fffu | ((i & 1u) << 15u)); }
        }
        std::memcpy(before_p,p,sizeof(p));std::memcpy(before_v,v,sizeof(v));
        const uint32_t pm = plan::metadata(p,16u), vm = plan::metadata(v,16u);
        unsigned expected_p = 0u, expected_v = 0u; bool fp = true, fv = true;
        for(unsigned i=0u;i<16u;++i) {
            const unsigned a=p[i]&0x7fffu,b=v[i]&0x7fffu;
            if(a>expected_p)expected_p=a;if(b>expected_v)expected_v=b;
            const unsigned ae=(p[i]>>7u)&255u,be=(v[i]>>7u)&255u;
            fp=fp&&(!a||(ae>=64u&&ae<=190u));fv=fv&&(!b||(be>=64u&&be<=190u));
        }
        if(pm!=(expected_p|(fp?plan::float_eligible:0u))||vm!=(expected_v|(fv?plan::float_eligible:0u)))return 1;
        if(plan::use_float(pm,vm)!=(fp&&fv))return 2;
        eligible += unsigned(fp&&fv);
        Value carry{0x800000u|(random_word()&0x7fffffu),int16_t(int(random_word()%255u)-126),bool(test&1u)};
        if(test%7u==0u)carry={0u,-133,bool(test&1u)};
        if(mode==4u)carry={0xffffffu,int16_t(26u+test%3u),bool(test&1u)};
        const unsigned skip=plan::skip_kind(carry.exponent,pm,vm);++skipped[skip];
        if(mode==4u && skip!=unsigned(carry.exponent>=27)*2u)return 3;
        if(skip) {
            Value values[17];values[0]=carry;int maximum=carry.exponent>-133?carry.exponent:-133;
            for(unsigned i=0u;i<16u;++i) {
                values[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(p[i],v[i],-133);
                if(values[i+1u].exponent>maximum)maximum=values[i+1u].exponent;
            }
            for(unsigned i=1u;i<17u;++i) {
                const unsigned shift=unsigned(maximum-values[i].exponent);
                const uint64_t aligned=shift>=64u?0u:(uint64_t(values[i].significand)<<2u)>>shift;
                if(aligned)return 4;
            }
            const auto actual=qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u);
            const auto empty=qrt_q1_moe_hawkeye::group_sum<26,-133>(&carry,1u);
            if(bits(qrt_q1_moe_hawkeye::value_to_float(actual))!=bits(qrt_q1_moe_hawkeye::value_to_float(empty)))return 5;
        }
        if(std::memcmp(before_p,p,sizeof(p))||std::memcmp(before_v,v,sizeof(v)))return 6;
    }
    assert(skipped[0]>10000u&&skipped[1]>=200000u&&skipped[2]>60000u&&eligible>100000u);
    std::printf("{\"groups\":500000,\"not_skipped\":%u,\"zero_operand_skips\":%u,\"zero_aligned_term_skips\":%u,\"float_eligible\":%u,\"bound_violations\":0,\"immutable_inputs\":true}\n",skipped[0],skipped[1],skipped[2],eligible);
}
