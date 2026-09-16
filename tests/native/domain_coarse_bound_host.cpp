#include "../../native/providers/moe_accumulator/sm121_domain_coarse_bound.h"
#include <cassert>
#include <cstdio>

namespace fast=qrt_sm121_domain_coarse_bound;
namespace base=qrt_sm121_coarse_projection_bound;
namespace scalar=base::scalar;
uint32_t random_word(){static uint32_t state=0x89157463u;state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
float random_value(unsigned maximum,bool signed_value){
    const unsigned exponent=random_word()%(maximum+1u),fraction=random_word()&0x7fffffu;
    return scalar::value((exponent<<23u)|fraction|(signed_value?(random_word()&0x80000000u):0u));
}
int main(){
    assert(!fast::width_supported(0u)&&!fast::width_supported(15u)&&!fast::width_supported(8208u));
    assert(fast::width_supported(16u)&&fast::width_supported(80u)&&fast::width_supported(8192u));
    const unsigned mantissas[]={0u,1u,2u,3u,0x3ffffeu,0x3fffffu,0x400000u,0x400001u,0x7ffffdu,0x7ffffeu,0x7fffffu};
    unsigned units=0u,uppers=0u;
    for(unsigned exponent=0u;exponent<=240u;++exponent)for(unsigned fraction:mantissas){
        const unsigned bits=(exponent<<23u)|fraction;const float x=scalar::value(bits);
        assert(scalar::bits(fast::upper(x))==scalar::bits(scalar::upper(x)));++uppers;
        if(bits){assert(scalar::bits(fast::positive_unit<23u>(x))==scalar::bits(base::unit(x,23u)));
            assert(scalar::bits(fast::positive_unit<25u>(x))==scalar::bits(base::unit(x,25u)));units+=2u;}
    }
    constexpr unsigned cases=2097152u;
    for(unsigned i=0u;i<cases;++i){
        // A larger state box than reachable C64 prefixes, including every
        // positive FP32 exponent through the proved per-field limits.
        const base::State before{random_value(237u,true),random_value(230u,false)};
        const float partial=random_value(229u,true),positive=random_value(229u,false);
        const auto expected=base::advance<4u,19u>(before,partial,positive);
        const auto actual=fast::advance(before,partial,positive);
        assert(scalar::bits(actual.center)==scalar::bits(expected.center));
        assert(scalar::bits(actual.error)==scalar::bits(expected.error));
        assert(scalar::finite(actual.center)&&scalar::finite(actual.error)&&actual.error>=0.0f);
    }
    // Zero/subnormal inputs and exact exponent transitions use a distinct
    // deterministic sweep instead of depending on random occurrence.
    unsigned edge_cases=0u;
    for(unsigned exponent=0u;exponent<=229u;++exponent)for(unsigned fraction:mantissas)
        for(unsigned sign=0u;sign<2u;++sign){
            const float p=scalar::value((sign<<31u)|(exponent<<23u)|fraction);
            base::State expected{},actual{};
            for(unsigned block=0u;block<128u;++block){
                const float partial=block%2u?-p:p,positive=scalar::absolute(p);
                expected=base::advance<4u,19u>(expected,partial,positive);
                actual=fast::advance(actual,partial,positive);
                assert(scalar::bits(actual.center)==scalar::bits(expected.center));
                assert(scalar::bits(actual.error)==scalar::bits(expected.error));++edge_cases;
            }
        }
    std::printf("{\"kind\":\"domain_coarse_bound_host\",\"random_states\":%u,\"ordered_edge_states\":%u,\"unit_comparisons\":%u,\"upper_comparisons\":%u,\"raw_mismatches\":0,\"width_fallback_checks\":true}\n",cases,edge_cases,units,uppers);
}
