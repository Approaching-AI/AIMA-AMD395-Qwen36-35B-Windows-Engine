#include "../../native/providers/moe_accumulator/sm121_dominant_half_products.h"
#include "dominant_half_cases.h"
#include <cassert>
#include <cstdio>
#include <cstring>
namespace candidate=qrt_sm121_dominant_half_products;
namespace half=qrt_sm121_scaled_half_products;
namespace original=qrt_q1_moe_hawkeye;
size_t checked=0u,accepted=0u,discarded=0u,partial=0u;
original::Value verify(original::Value carry,const uint16_t* left,const uint16_t* right) {
    const auto l=half::prepare(left),r=half::prepare(right);
    const bool valid=candidate::eligible(carry,l.control,r.control);
    assert(valid==dominant_half_cases::dominated(carry,left,right));
    original::Value terms[17];terms[0]=carry;
    int actual_maximum=std::max(-133,int(carry.exponent));
    for(unsigned i=0u;i<16u;++i) {
        assert(half::original(l,i)==left[i] && half::original(r,i)==right[i]);
        terms[i+1u]=original::multiply_bf16(left[i],right[i],-133);
        actual_maximum=std::max(actual_maximum,int(terms[i+1u].exponent));
    }
    const auto reference=original::group_sum<26,-133>(terms,17u);
    if(valid) {
        assert(actual_maximum==carry.exponent);
        const int power=25-carry.exponent+half::unit(l)+half::unit(r);
        assert(power<=-5);
        const auto sum=candidate::products<8u>(l.pairs,r.pairs,power);
        const auto actual=candidate::finish(carry,sum,((left[0]^right[0])&0x8000u)!=0u,carry.exponent);
        if(actual.significand!=reference.significand || actual.exponent!=reference.exponent || actual.negative!=reference.negative) {
            std::fprintf(stderr,"group %zu carry %u/%d/%u power %d got %u/%d/%u expected %u/%d/%u\n",checked,carry.significand,carry.exponent,carry.negative,power,actual.significand,actual.exponent,actual.negative,reference.significand,reference.exponent,reference.negative);
            std::abort();
        }
        ++accepted;discarded+=power<-126;partial+=((l.control&r.control)>>16u)!=65535u;
    }
    ++checked;return reference;
}
int main() {
    uint16_t left[16],right[16];
    for(unsigned bits=0u;bits<65536u;++bits) {
        std::fill_n(left,16u,uint16_t(bits));std::fill_n(right,16u,uint16_t(bits^0x8000u));
        for(int e:{-134,-133,0,128,254,300})for(unsigned sign=0u;sign<2u;++sign)
            verify({0x00ffffffu,int16_t(e),sign!=0u},left,right);
    }
    for(bool ordinary:{false,true})for(unsigned row=0u;row<2048u;++row) {
        original::Value carry{0u,-133,false};
        for(unsigned group=0u;group<129u;++group) {
            for(unsigned i=0u;i<16u;++i){const auto p=dominant_half_cases::input(row,group,i,ordinary);left[i]=p.x;right[i]=p.y;}
            carry=verify(carry,left,right);
        }
    }
    assert(accepted && accepted<checked && discarded && partial);
    std::printf("{\"groups\":%zu,\"dominated_groups\":%zu,\"underflow_scale_groups\":%zu,\"partial_nonzero_groups\":%zu,\"raw_value_mismatches\":0,\"all_bf16_payloads\":65536}\n",checked,accepted,discarded,partial);
}
