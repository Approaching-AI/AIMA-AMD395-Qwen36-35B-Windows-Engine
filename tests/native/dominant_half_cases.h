#pragma once
#include "strong_float_replay_cases.h"
namespace dominant_half_cases {
using Value=qrt_q1_moe_hawkeye::Value;
inline bool eligible_row(const uint16_t* row,unsigned* maximum=nullptr) {
    unsigned low=255u,high=0u;bool any=false;
    for(unsigned i=0u;i<16u;++i)if(row[i]&0x7fffu) {
        const unsigned e=(row[i]>>7u)&255u;
        if(!e || e==255u)return false;
        low=std::min(low,e);high=std::max(high,e);any=true;
    }
    if(maximum)*maximum=high;
    return !any || high-low<=29u;
}
// Uses original BF16 words only, independently of prepared metadata.
inline bool dominated(Value carry,const uint16_t* left,const uint16_t* right) {
    unsigned l=0u,r=0u;bool live=false;
    if(!eligible_row(left,&l) || !eligible_row(right,&r))return false;
    for(unsigned i=0u;i<16u;++i)live|=(left[i]&0x7fffu)&&(right[i]&0x7fffu);
    return live && carry.exponent>=-133 && carry.exponent>=int(l)+int(r)-254;
}
inline qrt_strong_replay_cases::Pair input(unsigned row,unsigned group,unsigned i,bool ordinary) {
    namespace cases=qrt_strong_replay_cases;
    const unsigned a=cases::random_word(row*7919u+group*997u+i*17u+0x3958192u);
    const unsigned b=cases::random_word(a^0x8192395u);
    if(ordinary) {
        cases::Pair p{uint16_t((a&0x807fu)|((120u+a%12u)<<7u)),uint16_t((b&0x807fu)|((119u+b%14u)<<7u))};
        if((row*17u+group*16u+i)%37u==0u)p.x=uint16_t(a&0x8000u);
        return p;
    }
    const unsigned mode=row%23u;
    if(mode<20u)return cases::input(row/23u*20u+mode,group,i);
    const unsigned exponent=mode==21u && !group?254u:1u;
    cases::Pair p{uint16_t((a&127u)|(exponent<<7u)),uint16_t((b&127u)|(exponent<<7u))};
    if(mode==20u)p.x|=uint16_t(a&0x8000u);
    if(mode==22u && i%2u)p.x=uint16_t(a&0x8000u);
    return p;
}
} // namespace dominant_half_cases
