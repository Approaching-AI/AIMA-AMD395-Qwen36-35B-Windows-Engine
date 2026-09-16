#pragma once
#include "strong_float_replay_cases.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
namespace qrt_pair_projection_reference {
struct Expected {unsigned recovered_pairs;};
inline Expected expected(qrt_q1_moe_hawkeye::Value carry,const uint16_t* a,const uint16_t* b) {
    int maximum=carry.exponent>-133?carry.exponent:-133;
    for(unsigned i=0;i<16;++i) {
        const auto p=qrt_q1_moe_hawkeye::multiply_bf16(a[i],b[i],-133);
        maximum=p.exponent>maximum?p.exponent:maximum;
    }
    unsigned count=0;
    for(unsigned i=0;i<16;i+=2) {
        unsigned units[2];bool valid=true;
        for(unsigned side=0;side<2;++side) {
            const auto* row=side?b:a;const unsigned e0=(row[i]>>7u)&255u,e1=(row[i+1u]>>7u)&255u;
            const unsigned lo=e0<e1?e0:e1,hi=e0>e1?e0:e1;
            units[side]=lo;valid&=lo>0u&&hi<255u&&hi-lo<=7u;
        }
        const int shift=maximum-int(units[0]+units[1])+243;
        count+=valid&&shift>=-11&&shift<=8;
    }
    return {count};
}
}
