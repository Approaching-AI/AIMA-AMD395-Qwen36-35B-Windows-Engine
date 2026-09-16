#include "../../native/providers/moe_accumulator/sm121_two_pass_packed_group.h"
#include <array>
#include <cstdio>
#include <stdexcept>

namespace original=qrt_q1_moe_hawkeye;
namespace candidate=qrt_sm121_two_pass_packed_group;
namespace f32=qrt_sm121_f32_carry;
void require(bool okay) {if(!okay)throw std::runtime_error("two-pass group differs from original");}
uint32_t packed(uint16_t word) {
    const int e=(word&0x7fffu)?int((word>>7u)&255u)-127:-512;
    return (uint32_t(word)<<16u)|uint16_t(e);
}
template<unsigned Chunk,unsigned L,unsigned R>
void compare(float carry,const uint32_t* left,const uint32_t* right,uint32_t expected,unsigned& accepted,unsigned& rejected) {
    float output=123.0f;
    const bool okay=candidate::accumulate<Chunk,L,R>(carry,left,right,&output);
    require(okay?f32::bits(output)==expected:output==123.0f);
    accepted+=okay;rejected+=!okay;
}
template<unsigned L,unsigned R> void run() {
    constexpr unsigned cases=262144u;
    unsigned accepted=0u,rejected=0u,zero=0u,state=0x3958192u;
    auto random=[&]{state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;};
    original::Value previous{0u,-133,false};
    for(unsigned index=0u;index<cases;++index) {
        std::array<uint32_t,16u*L> left;std::array<uint32_t,16u*R> right;
        left.fill(0xdeadbeefu);right.fill(0xbeefdeadu);
        original::Value values[17];
        const unsigned mode=index%5u,common=64u+index%127u;
        const uint16_t constant=uint16_t((common<<7u)|(random()&127u));
        for(unsigned i=0u;i<16u;++i) {
            uint16_t a=uint16_t((random()&0x807fu)|((64u+random()%127u)<<7u));
            uint16_t b=uint16_t((random()&0x807fu)|((64u+random()%127u)<<7u));
            if(mode==1u){a=uint16_t((a&0x807fu)|(common<<7u));b=uint16_t((b&0x807fu)|(common<<7u));}
            if(mode==2u){a&=0x8000u;b&=0x8000u;}
            if(mode==3u){a=constant;b=uint16_t(constant|(i&1u?0x8000u:0u));}
            if(mode==4u)a=b=uint16_t((190u<<7u)|127u);
            left[i*L]=packed(a);right[i*R]=packed(b);
            values[i+1u]=original::multiply_bf16(a,b,-133);
        }
        const unsigned exponent=1u+random()%254u;
        values[0]={0x800000u|(random()&0x7fffffu),int16_t(int(exponent)-127),bool(random()&1u)};
        if(index%8u==0u)values[0]={0u,-133,false};
        if(index%8u==1u && previous.significand>=0x800000u && previous.exponent<=127)
            values[0]=previous;
        previous=original::group_sum<26,-133>(values,17u);
        const auto final=qrt_sm121_group16::finish_accumulator(previous);
        const uint32_t expected=f32::bits(original::value_to_float(final));
        const auto saved_left=left;
        const auto saved_right=right;
        const float carry=original::value_to_float(values[0]);
        compare<1u,L,R>(carry,left.data(),right.data(),expected,accepted,rejected);
        compare<2u,L,R>(carry,left.data(),right.data(),expected,accepted,rejected);
        compare<4u,L,R>(carry,left.data(),right.data(),expected,accepted,rejected);
        compare<8u,L,R>(carry,left.data(),right.data(),expected,accepted,rejected);
        require(left==saved_left && right==saved_right);zero+=expected==0u;
    }
    require(accepted && rejected && zero);
    std::printf("{\"kind\":\"two_pass_packed_group_host\",\"cases\":%u,\"chunk_sizes\":4,\"left_stride\":%u,\"right_stride\":%u,\"accepted\":%u,\"fallback\":%u,\"original_zero_results\":%u,\"raw_bit_mismatches\":0,\"immutable_operands\":true,\"rejected_output_unchanged\":true,\"hardware_executed\":false}\n",cases,L,R,accepted,rejected,zero);
}
int main(){run<1u,1u>();run<1u,16u>();run<3u,5u>();}
