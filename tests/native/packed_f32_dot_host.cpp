#include "../../native/providers/moe_accumulator/sm121_packed_f32_dot.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace original=qrt_q1_moe_hawkeye;
namespace alignment=qrt_sm121_float_alignment;
struct Counts {size_t calls=0u,accepted=0u,declined=0u,products=0u;};
void require(bool ok){if(!ok)std::abort();}
uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4u);return u;}
uint32_t mix(uint32_t x){x^=x<<13u;x^=x>>17u;return x^(x<<5u);}
uint16_t word(unsigned trial,unsigned index,unsigned salt) {
    const unsigned mode=trial%8u,r=mix(trial*65537u+index*397u+salt);
    if(mode==0u)return uint16_t(r&0x8000u);
    if(mode==2u)return uint16_t((r&0x807fu)|((64u+r%127u)<<7u));
    if(mode==4u)return 0x5f7fu;
    if(mode==5u)return 0x2000u;
    if(mode==6u)return uint16_t(0x3fffu|((index&1u)&&salt==7u?0x8000u:0u));
    if(mode==7u)return uint16_t((r&0x807fu)|((salt==7u?64u:190u)<<7u));
    return uint16_t((r&0x807fu)|((113u+r%16u)<<7u));
}
template<unsigned Width,unsigned Columns>
void run(Counts& counts) {
    constexpr unsigned guard=7u;
    for(unsigned trial=0u;trial<512u;++trial) {
        std::array<uint16_t,Width> left{};
        std::array<std::array<uint16_t,Width>,Columns> right{};
        std::array<uint32_t,Width/2u+2u*guard> packed_left;
        std::array<uint32_t,Width/2u*Columns+2u*guard> packed_right;
        packed_left.fill(0xa5a5a5a5u);packed_right.fill(0xa5a5a5a5u);
        for(unsigned i=0u;i<Width;++i) {
            left[i]=word(trial,i,7u);
            if(trial%8u==3u && i==Width-1u)left[i]=trial%16u==3u?0x8001u:0x7f80u;
            for(unsigned c=0u;c<Columns;++c)right[c][i]=word(trial,i,53u+c*4093u);
        }
        for(unsigned i=0u;i<Width;i+=2u) {
            packed_left[guard+i/2u]=uint32_t(left[i])|(uint32_t(left[i+1u])<<16u);
            for(unsigned c=0u;c<Columns;++c)packed_right[guard+i/2u*Columns+c]=uint32_t(right[c][i])|(uint32_t(right[c][i+1u])<<16u);
        }
        const auto saved_left=packed_left;
        const auto saved_right=packed_right;
        for(unsigned c=0u;c<Columns;++c) {
            bool valid=true;
            for(unsigned i=0u;i<Width;++i)valid&=alignment::eligible(left[i])&&alignment::eligible(right[c][i]);
            float result=alignment::from_bits(0x12345678u);
            const bool accepted=qrt_sm121_packed_f32_dot::try_dot<Width,Columns>(packed_left.data()+guard,packed_right.data()+guard,c,valid,&result);
            ++counts.calls;
            if(accepted) {
                original::Value carry{0u,-133,false};
                for(unsigned base=0u;base<Width;base+=16u) {
                    original::Value terms[17];terms[0]=carry;
                    for(unsigned i=0u;i<16u;++i)terms[i+1u]=original::multiply_bf16(left[base+i],right[c][base+i],-133);
                    carry=original::group_sum<26,-133>(terms,17u);
                }
                const float expected=original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
                if(bits(expected)!=bits(result)) {
                    std::fprintf(stderr,"width=%u columns=%u trial=%u column=%u expected=%08x actual=%08x\n",Width,Columns,trial,c,bits(expected),bits(result));
                    std::abort();
                }
                require(valid);++counts.accepted;counts.products+=Width;
            }else {require(bits(result)==0x12345678u);++counts.declined;}
            if(trial%8u==3u || trial%8u==4u || trial%8u==5u)require(!accepted);
        }
        require(packed_left==saved_left && packed_right==saved_right);
    }
    float inactive=alignment::from_bits(0x12345678u);
    require(!qrt_sm121_packed_f32_dot::try_dot<Width,Columns>(nullptr,nullptr,Columns-1u,false,&inactive));
    require(bits(inactive)==0x12345678u);
}
int main() {
    Counts counts;run<64u,4u>(counts);run<64u,8u>(counts);run<128u,4u>(counts);run<128u,8u>(counts);
    require(counts.accepted>0u && counts.declined>0u && counts.calls==12288u);
    std::printf("{\"kind\":\"packed_f32_dot_host\",\"calls\":%zu,\"accepted\":%zu,\"declined\":%zu,\"independent_products\":%zu,\"raw_mismatches\":0,\"immutable_inputs\":true,\"declined_output_unchanged\":true,\"inactive_null_reads\":0,\"native_executed\":false}\n",counts.calls,counts.accepted,counts.declined,counts.products);
}
