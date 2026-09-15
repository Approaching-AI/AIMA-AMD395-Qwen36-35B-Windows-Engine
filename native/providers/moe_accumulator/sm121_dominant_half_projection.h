#pragma once
#include "sm121_dominant_half_products.h"
#include "sm121_staged_half_projection.h"

// Component candidate. Preparation, K16 order and the original fallback
// are shared with staged-half replay; no product dispatcher selects it.
namespace qrt_sm121_dominant_half_projection {
namespace staged=qrt_sm121_staged_half_projection;
namespace dominant=qrt_sm121_dominant_half_products;
using Row=staged::Row;
using Value=staged::Value;
using LaneOperands=staged::LaneOperands;
struct Stats { unsigned transformed=0u,original=0u,dominated=0u; };

template<bool AllNonzero>
__device__ __forceinline__ int original_maximum(Value carry,const LaneOperands& operands,unsigned active) {
    uint32_t paired_maximum=0u;
#pragma unroll
    for(unsigned i=0u;i<2u;++i) {
        uint32_t exponents=((operands.left[i]>>10u)&0x001f001fu)+((operands.right[i]>>10u)&0x001f001fu);
        if constexpr(!AllNonzero) {
            const unsigned bits=(active>>((threadIdx.x&3u)*4u+2u*i))&3u;
            exponents&=((bits&1u)|((bits&2u)<<15u))*65535u;
        }
        paired_maximum=qrt_sm121_prepared_integer_pairs::maximum_pair(paired_maximum,exponents);
    }
    int maximum=int((paired_maximum&65535u)>(paired_maximum>>16u)?paired_maximum&65535u:paired_maximum>>16u);
    maximum=qrt_sm121_lane_reduce::maximum<4u>(maximum)-30+
        int(int16_t(operands.left_control))+int(int16_t(operands.right_control));
    maximum=maximum>carry.exponent?maximum:carry.exponent;
    return maximum>-133?maximum:-133;
}
__device__ __forceinline__ Value accumulate(Value carry,const LaneOperands& operands,
    bool* transformed=nullptr,bool* dominated=nullptr) {
    const unsigned active=(operands.left_control&operands.right_control)>>16u;
    if(int16_t(operands.left_control)==-32768 || int16_t(operands.right_control)==-32768 || !active) {
        if(dominated)*dominated=false;
        return staged::accumulate(carry,operands,transformed);
    }
    const bool accepted=dominant::eligible(carry,operands.left_control,operands.right_control);
    if(dominated)*dominated=accepted;
    if(transformed)*transformed=true;
    int maximum=carry.exponent;
    if(!accepted)maximum=active==65535u?original_maximum<true>(carry,operands,active):original_maximum<false>(carry,operands,active);
    // Reconverge before multiplication and normalization: quads with and
    // without a dominating carry share the rest of the arithmetic body.
    const int power=25-maximum+int(int16_t(operands.left_control))+int(int16_t(operands.right_control));
    const uint32_t total=qrt_sm121_lane_reduce::sum<4u>(dominant::products<2u>(operands.left,operands.right,power));
    return dominant::finish(carry,total,((operands.left[0]^operands.right[0])&0x8000u)!=0u,maximum);
}
template<unsigned StagingGroups,bool Audit=false>
__device__ __forceinline__ float dot(const Row* left,const Row* right,unsigned width,
    uint32_t* raw_trace=nullptr,Stats* stats=nullptr) {
    static_assert(StagingGroups==2u || StagingGroups==4u);
    const unsigned lane=threadIdx.x&3u,groups=width/16u;Value carry{0u,-133,false};Stats counts;
#pragma unroll 1
    for(unsigned base=0u;base<groups;base+=StagingGroups) {
        LaneOperands operands[StagingGroups];
        const unsigned count=groups-base<StagingGroups?groups-base:StagingGroups;
#pragma unroll
        for(unsigned i=0u;i<StagingGroups;++i)if(i<count)operands[i]=staged::load(left[base+i],right[base+i]);
#pragma unroll
        for(unsigned i=0u;i<StagingGroups;++i)if(i<count) {
            bool transformed,dominated;
            carry=accumulate(carry,operands[i],Audit?&transformed:nullptr,Audit?&dominated:nullptr);
            if constexpr(Audit) {
                counts.transformed+=transformed;counts.original+=!transformed;counts.dominated+=dominated;
                if(!lane && raw_trace) {
                    raw_trace[3u*(base+i)]=carry.significand;
                    raw_trace[3u*(base+i)+1u]=uint32_t(int32_t(carry.exponent));
                    raw_trace[3u*(base+i)+2u]=unsigned(carry.negative);
                }
            }
        }
    }
    if constexpr(Audit)if(!lane && stats)*stats=counts;
    return lane?0.0f:qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_dominant_half_projection
