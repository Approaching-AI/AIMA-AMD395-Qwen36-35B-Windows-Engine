#include "../../native/providers/moe_accumulator/sm121_dyadic_carry_scan.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
namespace exact=qrt_sm121_dyadic_carry_scan;
namespace original=qrt_q1_moe_hawkeye;
using Value=exact::Value;
uint64_t random_state=0x39581927169ull;
uint64_t random_word(){random_state^=random_state<<13u;random_state^=random_state>>7u;return random_state^=random_state<<17u;}
uint64_t independent(exact::Function f,uint64_t x) {
    // Wide division avoids the implementation's bit-mask rounding and
    // handles addition overflow before taking the final modulo projection.
    const __uint128_t q=__uint128_t(1u)<<f.s;
    return uint64_t(((__uint128_t(x)+f.b)/q)*q+f.c);
}
void algebra() {
    size_t exhaustive=0;
    for(unsigned s1=0;s1<4;++s1)for(unsigned b1=0;b1<(1u<<s1);++b1)for(unsigned c1=0;c1<16;++c1)
    for(unsigned s2=0;s2<4;++s2)for(unsigned b2=0;b2<(1u<<s2);++b2)for(unsigned c2=0;c2<16;++c2)
    for(unsigned x=0;x<16;++x) {
        const exact::Function a{b1,c1,s1},b{b2,c2,s2};
        assert((exact::evaluate(exact::compose(a,b),x)&15u)==(independent(b,independent(a,x))&15u));++exhaustive;
    }
    for(unsigned i=0;i<200000;++i) {
        exact::Function a,b,c;
        for(auto* f:{&a,&b,&c}){f->s=unsigned(random_word()%64u);f->b=random_word()&exact::low(f->s);f->c=random_word();}
        const uint64_t x=random_word();
        assert(exact::evaluate(exact::compose(a,b),x)==independent(b,independent(a,x)));
        const auto left=exact::compose(exact::compose(a,b),c),right=exact::compose(a,exact::compose(b,c));
        assert(left.s==right.s&&left.b==right.b&&left.c==right.c);
        assert(exact::evaluate(left,x)==independent(c,independent(b,independent(a,x))));
    }
    assert(exhaustive==921600u);
    std::printf("{\"kind\":\"dyadic_algebra\",\"exhaustive_pairs\":%zu,\"random_pairs_and_triples\":200000,\"mismatches\":0}\n",exhaustive);
}
bool same(Value a,Value b){return a.significand==b.significand&&a.exponent==b.exponent&&(!a.significand||a.negative==b.negative);}
template<unsigned Groups>void generated() {
    size_t accepted=0,rejected=0,accepted_states=0,injected=0,changed=0;
    for(unsigned trial=0;trial<16384;++trial) {
        std::array<std::array<Value,16>,Groups> products;
        std::array<Value,Groups> reference,predicted;
        std::array<float,Groups> approximate,previous;
        std::array<exact::Function,Groups> functions,prior;
        std::array<int,Groups> maximum;
        std::array<int64_t,Groups> sums;
        Value entry{0u,-133,false};
        if(trial%3u)entry={0x800000u|uint32_t(random_word()&0x7fffffu),int16_t(int(trial%21u)-10),bool(trial&1u)};
        for(unsigned g=0;g<Groups;++g) {
            Value group[17];group[0]=g?reference[g-1]:entry;float partial=0;
            for(unsigned i=0;i<16;++i) {
                uint16_t a=uint16_t(((120u+random_word()%15u)<<7u)|(random_word()&0x807fu));
                uint16_t b=uint16_t(((120u+random_word()%15u)<<7u)|(random_word()&0x807fu));
                if(trial%11u==0u){a=uint16_t(random_word());b=uint16_t(random_word());}
                if(trial%17u==0u)a=0u;
                if(trial%19u==0u){a=1u;b=0x3f80u;}
                products[g][i]=group[i+1]=original::multiply_bf16(a,b,-133);
                const auto p=group[i+1];partial+=std::ldexp(float(p.significand)*(p.negative?-1.f:1.f),p.exponent-23);
            }
            approximate[g]=partial;reference[g]=original::group_sum<26,-133>(group,17u);
            changed+=reference[g].significand&&group[0].significand&&reference[g].exponent!=group[0].exponent;
        }
        for(unsigned step=1;step<Groups;step*=2u){previous=approximate;for(unsigned g=step;g<Groups;++g)approximate[g]=previous[g-step]+previous[g];}
        for(unsigned g=0;g<Groups;++g)predicted[g]=original::value_from_float(original::value_to_float(entry)+approximate[g],-133);
        if(trial%7u==0u){auto& p=predicted[trial%Groups];p.negative=!p.negative;if(p.significand)++p.exponent;++injected;}
        int base=10000;bool valid=exact::regular(entry);
        for(unsigned g=0;g<Groups;++g) {
            const auto incoming=g?predicted[g-1]:entry;int e=std::max(-133,int(incoming.exponent));
            for(auto p:products[g])e=std::max(e,int(p.exponent));
            int64_t sum=0;
            for(auto p:products[g]){const unsigned shift=unsigned(e-p.exponent);const uint64_t m=shift>=32?0u:(uint64_t(p.significand)<<2u)>>shift;sum+=p.negative?-int64_t(m):int64_t(m);}
            maximum[g]=e;sums[g]=sum;base=std::min(base,e-25);
            if(predicted[g].significand)base=std::min(base,int(predicted[g].exponent)-23);
        }
        if(entry.significand)base=std::min(base,int(entry.exponent)-23);
        for(unsigned g=0;g<Groups;++g)valid=exact::make_step(g?predicted[g-1]:entry,predicted[g],maximum[g],sums[g],base,&functions[g])&&valid;
        uint64_t initial=0;valid=exact::to_integer(entry,base,&initial)&&valid;
        if(!valid){++rejected;continue;}
        for(unsigned step=1;step<Groups;step*=2u){prior=functions;for(unsigned g=step;g<Groups;++g)functions[g]=exact::compose(prior[g-step],prior[g]);}
        std::array<Value,Groups> candidates;
        for(unsigned g=0;g<Groups;++g){const bool ok=exact::from_integer(exact::evaluate(functions[g],initial),base,&candidates[g]);valid=ok&&valid;if(ok)valid=exact::same_class(candidates[g],predicted[g])&&valid;}
        if(!valid){++rejected;continue;}
        ++accepted;
        for(unsigned g=0;g<Groups;++g){assert(same(candidates[g],reference[g]));++accepted_states;}
    }
    assert(accepted>8000u&&rejected>1000u&&injected>1000u&&changed>1000u);
    std::printf("{\"kind\":\"dyadic_generated_trajectory\",\"groups\":%u,\"trials\":16384,\"accepted\":%zu,\"rejected\":%zu,\"accepted_states\":%zu,\"injected_wrong_predictions\":%zu,\"binade_changes\":%zu,\"mismatches\":0}\n",Groups,accepted,rejected,accepted_states,injected,changed);
}
void invalid() {
    exact::Function f{3u,4u,2u};const auto saved=f;const Value normal{0x800000u,0,false};
    assert(!exact::make_step(normal,normal,0,0,-200,&f));assert(f.b==saved.b&&f.c==saved.c&&f.s==saved.s);
    assert(!exact::make_step(normal,normal,0,INT64_MAX,-25,&f));
    assert(!exact::make_step(normal,normal,-1,0,-25,&f));
    assert(!exact::make_step(normal,normal,0,0,-80,&f));
    assert(!exact::make_step(normal,normal,0,0,-25,nullptr));
    Value value{42u,11,true};assert(!exact::from_integer(1ull<<63u,0,&value));assert(value.significand==42u&&value.exponent==11&&value.negative);
    assert(!exact::from_integer((1ull<<40u)|1u,-40,&value));
    assert(!exact::from_integer(1u,-149,&value));
    assert(exact::from_integer(1u,0,&value)&&same(value,normal));
    uint64_t output=123;assert(!exact::to_integer(normal,1,&output)&&output==123);
    assert(!exact::to_integer({1u,-126,false},-149,&output)&&output==123);
}
int main(){algebra();invalid();generated<4>();generated<8>();generated<16>();generated<32>();generated<64>();}
