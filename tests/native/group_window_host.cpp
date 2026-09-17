#include "../../native/providers/moe_accumulator/sm121_group_window.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <array>
#include <cassert>
#include <cstdio>
namespace window=qrt_sm121_group_window;
namespace original=qrt_q1_moe_hawkeye;
uint32_t rng=0x395918u;
uint32_t random_word(){rng^=rng<<13u;rng^=rng>>17u;rng^=rng<<5u;return rng;}
original::Value step(original::Value carry,const uint32_t* a,const uint32_t* b,unsigned base){
    original::Value values[17];values[0]=carry;
    for(unsigned i=0u;i<16u;++i)values[i+1u]=original::multiply_bf16(
        uint16_t(a[(base+i)/2u]>>((i&1u)*16u)),uint16_t(b[(base+i)/2u]>>((i&1u)*16u)),-133);
    return original::group_sum<26,-133>(values,17u);
}
bool equal(original::Value a,original::Value b){return a.significand==b.significand&&a.exponent==b.exponent&&a.negative==b.negative;}
int main(){
    std::array<uint32_t,64u> a{},b{};std::array<uint32_t,512u> strided{};
    for(unsigned word=0u;word<65536u;++word){
        a.fill(0x80008000u);a[13]=word|0x80000000u;
        assert((window::prepare<8u>(a.data())==((word&0x7fffu)?2u:0u)));
    }
    for(unsigned x=0u;x<256u;++x)for(unsigned y=0u;y<256u;++y){
        a.fill(0x80008000u);b.fill(0u);strided.fill(0x80008000u);
        for(unsigned g=0u;g<8u;++g){a[g*8u]=(x&(1u<<g))?0x80003f80u:0x80008000u;
            b[g*8u+7u]=(y&(1u<<g))?0xbf800000u:0u;
            for(unsigned i=0u;i<8u;++i)strided[(g*8u+i)*8u+3u]=a[g*8u+i];}
        assert(window::prepare<8u>(a.data())==x&&window::prepare<8u>(b.data())==y);
        assert((window::prepare<8u,8u>(strided.data()+3u)==x));
        const unsigned mask=x&y,first=window::first(mask),last=window::end(mask);
        unsigned visited=0u,prior=0u;
        for(unsigned cursor=mask;cursor;cursor&=cursor-1u){const unsigned g=window::first(cursor);
            assert(g>=first&&g<last&&(!visited||g>prior));prior=g;++visited;}
        assert(visited==window::population(mask));
        assert((!mask&&first==0u&&last==0u)||(mask&&(mask&(1u<<first))&&(mask&(1u<<(last-1u)))));
    }
    size_t comparisons=0u,omitted_window=0u,omitted_set=0u;
    for(unsigned sample=0u;sample<32768u;++sample){
        const unsigned am=random_word()&255u,bm=random_word()&255u;
        for(unsigned i=0u;i<64u;++i){
            a[i]=random_word();b[i]=random_word();
            if(sample%3u==0u){a[i]=(a[i]&0x807f807fu)|0x3e803e80u;b[i]=(b[i]&0x807f807fu)|0x3f803f80u;}
            if(sample%5u==0u){a[i]&=0x807f807fu;b[i]=(b[i]&0x807f807fu)|0x7f007f00u;}
            if(!(am&(1u<<(i/8u))))a[i]&=0x80008000u;
            if(!(bm&(1u<<(i/8u))))b[i]&=0x80008000u;
        }
        const unsigned active=window::prepare<8u>(a.data())&window::prepare<8u>(b.data());
        const unsigned first=window::first(active),last=window::end(active);
        original::Value full{0u,-133,false},bounded=full,sparse=full;
        for(unsigned g=0u;g<8u;++g){
            full=step(full,a.data(),b.data(),g*16u);
            if(g>=first&&g<last)bounded=step(bounded,a.data(),b.data(),g*16u);else ++omitted_window;
            if(active&(1u<<g))sparse=step(sparse,a.data(),b.data(),g*16u);else ++omitted_set;
            assert(equal(full,bounded)&&equal(full,sparse));comparisons+=2u;
        }
    }
    std::printf("{\"kind\":\"group_window_host\",\"bf16_encodings\":65536,\"mask_pairs\":65536,\"strides\":[1,8],\"chains\":32768,\"raw_carry_comparisons\":%zu,\"window_omitted_groups\":%zu,\"set_omitted_groups\":%zu,\"raw_mismatches\":0,\"native_gpu_executed\":false,\"inference_acceptance\":false}\n",comparisons,omitted_window,omitted_set);
}
