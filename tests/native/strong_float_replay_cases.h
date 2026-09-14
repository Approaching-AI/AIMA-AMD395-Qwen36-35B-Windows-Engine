#pragma once
#include "../../native/providers/moe_accumulator/sm121_strong_float_subgroup.h"
#include <algorithm>
#include <cstdint>
namespace qrt_strong_replay_cases {
namespace original=qrt_q1_moe_hawkeye;
namespace alignment=qrt_sm121_float_alignment;
struct Pair { uint16_t x,y; };
inline uint32_t random_word(uint32_t x) { x^=x<<13; x^=x>>17; return x^(x<<5); }

inline Pair cancellation(unsigned exponent,unsigned i) {
 if(i>=4) return {0,0};
 const uint16_t a=uint16_t(exponent<<7),b=uint16_t(a|1);
 const Pair p[4]={{b,b},{uint16_t(a|0x8000),b},{uint16_t(b|0x8000),a},{a,a}};
 return p[i];
}
inline Pair strong_input(unsigned row,unsigned group,unsigned i) {
 const uint32_t a=random_word(0x3958192u ^ (row*7919u+group*997u+i*17u)),b=random_word(a^0x8192395u);
 const unsigned mode=row%16;
 if(mode<3) { const uint16_t x=uint16_t(((84+row/16%91)<<7)|127); return {uint16_t(x | (mode==1 || (mode==2 && i%2) ? 0x8000:0)),x}; }
 if(mode==3) return {uint16_t((a&0x807f)|((84+a%91)<<7)),uint16_t((b&0x807f)|((84+b%91)<<7))};
 if(mode==4) return {uint16_t((a&0x807f)|(84<<7)),uint16_t((b&0x807f)|(84<<7))};
 if(mode==5) return {uint16_t((a&0x807f)|(174<<7)),uint16_t((b&0x807f)|(174<<7))};
 if(mode==6) return {uint16_t(a&0x8000),uint16_t((b&0x807f)|((84+b%91)<<7))};
 if(mode==7) return {uint16_t((a&0x807f)|((112+a%21)<<7)),uint16_t((b&0x807f)|((112+b%21)<<7))};
 if(mode==8 || mode==9) { auto p=group ? Pair{0,0}:cancellation(84,i);if(mode==9)p.x^=0x8000;return p; }
 if(mode==10) return cancellation(84+row/16%91,i);
 if(mode==11) return group%2 ? cancellation(84,i):cancellation(174,i);
 if(mode==12) return {uint16_t((a&0x807f)|((i%2?84:174)<<7)),uint16_t((b&0x807f)|((i%2?174:84)<<7))};
 if(mode==13) return {uint16_t((a&0x807f)|((i%2?84:174)<<7)),uint16_t((b&0x807f)|((i%2?84:174)<<7))};
 if(mode==14) return group<128 ? Pair{0,0}:cancellation(84,i);
 return {uint16_t(a&0x8000),uint16_t(b&0x8000)};
}
inline original::Value reference(original::Value carry,const Pair* p) {
 original::Value values[17];values[0]=carry;
 for(unsigned i=0;i<16;i++)values[i+1]=original::multiply_bf16(p[i].x,p[i].y,-133);
 return original::group_sum<26,-133>(values,17);
}

inline Pair input(unsigned row, unsigned group, unsigned i) {
    const unsigned mode = row % 20u;
    if (mode < 16u) return strong_input(row / 20u * 16u + mode, group, i);
    if (mode == 16u) return group ? Pair{0u, 0u} : cancellation(83u, i);
    if (mode == 17u) return group ? Pair{0u, 0u} : cancellation(80u, i);
    const uint32_t a = random_word(row * 7919u + group * 17u + i), b = random_word(a ^ 0x3958192u);
    if (mode == 18u) return {uint16_t(a), uint16_t(b)};
    return {uint16_t((a & 0x807fu) | ((i & 1u ? 64u : 190u) << 7u)),
            uint16_t((b & 0x807fu) | ((i & 1u ? 190u : 64u) << 7u))};
}
inline bool row_certificate(unsigned row, unsigned width) {
    bool result = true;
    for (unsigned group = 0u; group < width / 16u; ++group) for (unsigned i = 0u; i < 16u; ++i) {
        const auto p = input(row, group, i);
        result = result && qrt_sm121_strong_float::eligible(p.x) && qrt_sm121_strong_float::eligible(p.y);
    }
    return result;
}
inline uint32_t output_bits(original::Value carry) {
    const float value = original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
    uint32_t bits; __builtin_memcpy(&bits, &value, 4u); return bits;
}
} // namespace qrt_strong_replay_cases
