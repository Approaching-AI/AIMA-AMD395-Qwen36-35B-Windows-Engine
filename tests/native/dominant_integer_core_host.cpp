#include "sm121_dominant_integer_core.h"
#include "float_alignment_cases.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>
namespace fast=qrt_sm121_dominant_integer_core;
namespace core=qrt_sm121_integer_core;
namespace original=qrt_q1_moe_hawkeye;
using Value=original::Value;
bool same(Value a,Value b) {return a.significand==b.significand && a.exponent==b.exponent && a.negative==b.negative;}
int main() {
    size_t cases=0u,accepted=0u,rejected=0u,compensated=0u,without_replays=0u,negative_results=0u;
    const Value marker{0xa5a5a5a5u,-17,true};
    for(unsigned row=0u;row<8192u;++row)for(unsigned group=0u;group<16u;++group) {
        core::Row a{},b{};Value terms[17];
        for(unsigned i=0u;i<16u;++i) {
            const auto p=qrt_float_alignment_cases::input(row,group,i);
            a.original[i]=p.left;b.original[i]=p.right;
            terms[i+1u]=original::multiply_bf16(p.left,p.right,-133);
        }
        core::prepare(a);core::prepare(b);const auto before_a=a,before_b=b;
        int32_t partials[4]{};int64_t mathematical=0;
        for(unsigned i=0u;i<16u;++i) {
            const int ac=core::signed_core(a.original[i],a.unit),bc=core::signed_core(b.original[i],b.unit);
            const int ah=int(uint16_t(ac)>>8u)-(ac<0?256:0),bh=int(uint16_t(bc)>>8u)-(bc<0?256:0);
            const int al=uint16_t(ac)&255u,bl=uint16_t(bc)&255u;
            partials[0]+=ah*bh;partials[1]+=ah*bl;partials[2]+=al*bh;partials[3]+=al*bl;
            mathematical+=int64_t(ac)*bc;
        }
        const auto product=fast::combine(partials[0],partials[1],partials[2],partials[3]);
        assert(fast::mathematical(product)==mathematical && product.lower>=-fast::lower_bound && product.lower<=fast::lower_bound);
        const int bound=a.maximum+b.maximum-254;
        for(int exponent:{bound-1,bound,bound+1,bound+3,bound+8,bound+13,bound+14,-133,-102,-101,127,128}) {
            const Value carry{exponent==-133?0u:0x800000u|((row*7919u+group*37u)&0x7fffffu),int16_t(exponent),bool((row+group)&1u)};
            terms[0]=carry;const auto expected=original::group_sum<26,-133>(terms,17u);
            Value actual=marker;unsigned pairs=0xa5a5a5a5u;
            const bool used=fast::accumulate(carry,a,b,product,&actual,&pairs);
            if(used) {
                if(!same(actual,expected)) {
                    std::fprintf(stderr,"dominant core mismatch row=%u group=%u exponent=%d units=%d,%d expected=%x/%d/%u actual=%x/%d/%u\n",row,group,exponent,a.unit,b.unit,expected.significand,expected.exponent,unsigned(expected.negative),actual.significand,actual.exponent,unsigned(actual.negative));return 1;
                }
                assert(pairs<=16u);++accepted;compensated+=pairs;without_replays+=pairs==0u;negative_results+=actual.negative;
            } else {assert(same(actual,marker) && pairs==0xa5a5a5a5u);++rejected;}
            assert(!fast::accumulate(carry,a,b,product,nullptr));
            Value invalid=carry;invalid.significand=0x1000000u;actual=marker;
            assert(!fast::accumulate(invalid,a,b,product,&actual) && same(actual,marker));
            ++cases;
        }
        assert(!std::memcmp(&a,&before_a,sizeof(a)) && !std::memcmp(&b,&before_b,sizeof(b)));
    }
    // Explicit maximum positive/negative modulo overlap, including the carry.
    for(unsigned negative=0u;negative<2u;++negative) {
        core::Row a{},b{};Value terms[17];
        for(unsigned i=0u;i<16u;++i){a.original[i]=uint16_t(0x3fffu|(negative?0x8000u:0u));b.original[i]=0x3fffu;terms[i+1u]=original::multiply_bf16(a.original[i],b.original[i],-133);}
        core::prepare(a);core::prepare(b);int32_t h=0,cross=0,l=0;
        for(unsigned i=0u;i<16u;++i){const int ac=core::signed_core(a.original[i],a.unit),bc=core::signed_core(b.original[i],b.unit);const int ah=int(uint16_t(ac)>>8u)-(ac<0?256:0),bh=int(uint16_t(bc)>>8u)-(bc<0?256:0);const int al=uint16_t(ac)&255u,bl=uint16_t(bc)&255u;h+=ah*bh;cross+=ah*bl+al*bh;l+=al*bl;}
        terms[0]={0xffffffu,0,bool(negative)};Value actual=marker;
        assert(fast::accumulate(terms[0],a,b,fast::combine(h,cross,0,l),&actual));
        assert(same(actual,original::group_sum<26,-133>(terms,17u)));
    }
    assert(accepted>100000u && rejected>100000u && compensated>10000u && without_replays>10000u && negative_results>10000u);
    std::printf("{\"kind\":\"dominant_integer_core_host\",\"cases\":%zu,\"accepted\":%zu,\"rejected\":%zu,\"compensated_pairs\":%zu,\"accepted_without_replay\":%zu,\"negative_results\":%zu,\"maximum_modulo_overlap_cases\":2,\"raw_mismatches\":0,\"rejected_outputs_unchanged\":true,\"immutable_rows\":true,\"prediction_required\":false,\"native_matrix_checked\":false}\n",cases,accepted,rejected,compensated,without_replays,negative_results);
}
