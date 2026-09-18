#pragma once
#include <cstdint>
namespace qrt_narrow_half_cases {
struct Pair { uint16_t x,y; };
inline uint32_t random_word(uint32_t x){x^=x<<13u;x^=x>>17u;return x^(x<<5u);}
inline Pair input(unsigned row,unsigned group,unsigned i){
    const unsigned a=random_word(row*7919u+group*997u+i*17u+0x3958192u),b=random_word(a^0x8192395u);
    const unsigned ap=124u+random_word(row*97u+group*17u+1u)%36u;
    const unsigned bp=124u+random_word(row*67u+group*31u+2u)%36u;
    Pair p{uint16_t((a&0x807fu)|((ap-a%30u)<<7u)),uint16_t((b&0x807fu)|((bp-b%30u)<<7u))};
    switch(row%16u){
    case 0u:return {uint16_t(a&0x8000u),uint16_t(b&0x8000u)};
    case 1u:return {uint16_t((95u<<7u)|127u),uint16_t((95u<<7u)|127u)};
    case 2u:return {uint16_t((159u<<7u)|127u),uint16_t((159u<<7u)|127u)};
    case 3u:case 14u:{
        if(i>=4u)return {0u,0u};const uint16_t x=uint16_t((row%16u==3u?95u:159u)<<7u),y=uint16_t(x|1u);
        const Pair c[4]={{y,y},{uint16_t(x|0x8000u),y},{uint16_t(y|0x8000u),x},{x,x}};return c[i];
    }
    case 4u:{const uint16_t x=uint16_t(((group&1u?95u:159u)<<7u)|127u);return {x,uint16_t(x|((group%4u==2u)?0x8000u:0u))};}
    case 5u:if(i&1u)p.x&=0x8000u;else p.y&=0x8000u;break;
    case 6u:if(group%17u)p.x&=0x8000u;break;
    case 7u:p.x=uint16_t((a&0x807fu)|((i&1u?130u:159u)<<7u));break;
    case 9u:p.x=uint16_t((a&0x807fu)|((i&1u?129u:159u)<<7u));break;
    case 10u:p.x=uint16_t((a&0x807fu)|(94u<<7u));break;
    case 11u:p.y=uint16_t((b&0x807fu)|(160u<<7u));break;
    case 12u:p.x=uint16_t(i%3u==0u?1u:i%3u==1u?0x7f80u:0x7fc1u);break;
    case 13u:if(!i&&group%17u==0u)p.y=uint16_t(94u<<7u);break;
    case 15u:if(a%7u==0u)p.x&=0x8000u;if(b%11u==0u)p.y&=0x8000u;break;
    default:break;
    }
    return p;
}
} // namespace qrt_narrow_half_cases
