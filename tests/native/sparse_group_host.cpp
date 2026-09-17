#include "../../native/providers/moe_accumulator/sm121_sparse_group.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace sparse=qrt_sm121_sparse_group;
namespace original=qrt_q1_moe_hawkeye;
using Value=original::Value;
void require(bool condition){if(!condition)throw std::runtime_error("sparse K16 host comparison");}
bool equal(Value a,Value b){return a.significand==b.significand && a.exponent==b.exponent && a.negative==b.negative;}
Value reference(Value carry,const uint16_t* a,const uint16_t* b){
    Value terms[17];terms[0]=carry;
    for(unsigned i=0u;i<16u;++i)terms[i+1u]=original::multiply_bf16(a[i],b[i],-133);
    return original::group_sum<26,-133>(terms,17u);
}
template<unsigned Stride>
Value candidate(Value carry,const uint16_t* a,const uint16_t* b,unsigned* active_count){
    uint32_t left[8]{},right[8u*Stride]{};uint32_t lm=0u,rm=0u;
    for(unsigned i=0u;i<16u;++i){
        left[i/2u]|=uint32_t(a[i])<<((i&1u)*16u);
        right[(i/2u)*Stride]|=uint32_t(b[i])<<((i&1u)*16u);
        if(a[i]&0x7fffu)lm|=1u<<i;if(b[i]&0x7fffu)rm|=1u<<i;
    }
    const auto saved_left=std::array<uint32_t,8>{left[0],left[1],left[2],left[3],left[4],left[5],left[6],left[7]};
    const unsigned al=sparse::prepare<1u>(left),ar=sparse::prepare<Stride>(right);
    require(al==lm && ar==rm && !std::memcmp(left,saved_left.data(),sizeof(left)));
    const unsigned mask=al&ar;*active_count=sparse::population(mask);
    if(!mask)return sparse::empty(carry);
    return *active_count<=4u?sparse::small<Stride>(carry,left,right,mask):reference(carry,a,b);
}
int main()try{
    uint32_t random=0x3958192u;auto next=[&](){random^=random<<13u;random^=random>>17u;random^=random<<5u;return random;};
    size_t checks=0u,empty=0u,short_groups=0u,dense=0u;
    auto test=[&](Value carry,const uint16_t* a,const uint16_t* b){
        const Value expected=reference(carry,a,b);unsigned n0,n1;
        const Value c0=candidate<1u>(carry,a,b,&n0),c1=candidate<8u>(carry,a,b,&n1);
        require(n0==n1 && equal(expected,c0) && equal(expected,c1));
        ++checks;empty+=n0==0u;short_groups+=n0>0u&&n0<=4u;dense+=n0>4u;return expected;
    };
    // Every original BF16 word, including signed zeros, subnormals and special
    // encodings, appears in an independently checked small-product group.
    for(unsigned raw=0u;raw<65536u;++raw){
        uint16_t a[16]{},b[16]{};
        for(unsigned i=0u;i<16u;++i){a[i]=uint16_t(next()&0x8000u);b[i]=uint16_t(next());}
        for(unsigned i=0u;i<4u;++i)a[(raw+i*3u)%16u]=uint16_t(raw+i);
        for(Value c:std::array<Value,5>{{{0u,-133,false},{0u,-133,true},{0x12345u,-126,true},{0xffffffu,127,false},{0x800001u,256,true}}})test(c,a,b);
    }
    // Enumerate every zero/nonzero support pattern with both operand layouts.
    for(unsigned mask=0u;mask<65536u;++mask){
        uint16_t a[16],b[16];for(unsigned i=0u;i<16u;++i){a[i]=mask&(1u<<i)?uint16_t(0x3f80u|(next()&0x807fu)):uint16_t(next()&0x8000u);b[i]=uint16_t(0x3f80u|(next()&0x807fu));}
        test({0xabcdefu,int16_t(int(mask%390u)-126),bool(mask&1u)},a,b);
    }
    // Chained raw states exercise cancellations, late exceptional operands,
    // extended carries, empty groups and repeated zero after underflow.
    for(unsigned dot=0u;dot<32768u;++dot){
        Value carry{0u,-133,false};
        for(unsigned group=0u;group<8u;++group){
            uint16_t a[16],b[16];const unsigned limit=(dot+group)%17u;
            for(unsigned i=0u;i<16u;++i){a[i]=i<limit?uint16_t(next()):uint16_t(next()&0x8000u);b[i]=uint16_t(next());if((dot+group)%11u==0u)b[i]&=0x8000u;}
            carry=test(carry,a,b);
        }
    }
    require(empty && short_groups && dense);
    std::printf("{\"kind\":\"sparse_group_host\",\"bf16_encodings\":65536,\"support_masks\":65536,\"strides\":[1,8],\"chained_dots\":32768,\"group_checks\":%zu,\"empty_groups\":%zu,\"short_groups\":%zu,\"dense_groups\":%zu,\"raw_state_mismatches\":0,\"mask_mismatches\":0,\"inference_acceptance\":false}\n",checks,empty,short_groups,dense);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
