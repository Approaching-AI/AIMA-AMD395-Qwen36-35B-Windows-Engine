#pragma once
#include "sm121_staged_half_projection.h"
#include "sm121_dominant_half_products.h"
#include "sm121_carry_transfer.h"

// Component only. Candidate identities, prepared operands and four-lane
// output ownership match the selected staged projection implementation.
namespace qrt_sm121_transfer_projection {
namespace staged=qrt_sm121_staged_half_projection;
namespace dominant=qrt_sm121_dominant_half_products;
namespace transfer=qrt_sm121_carry_transfer;
using Row=staged::Row;using Value=staged::Value;
struct Stats{unsigned transformed=0u,original=0u,transfer_blocks=0u,reused_groups=0u,transition_replays=0u;};
__device__ __forceinline__ void record(uint32_t* trace,unsigned group,Value carry) {
    if(!(threadIdx.x&3u) && trace){trace[3u*group]=carry.significand;trace[3u*group+1u]=uint32_t(int32_t(carry.exponent));trace[3u*group+2u]=unsigned(carry.negative);}
}
template<unsigned Groups,bool Audit=false>
__device__ __forceinline__ float dot(const Row* left,const Row* right,unsigned width,
    uint32_t* trace=nullptr,Stats* statistics=nullptr) {
    static_assert(Groups==2u || Groups==4u || Groups==8u);
    Value carry{0u,-133,false};Stats stats;const unsigned group_count=width/16u;
#pragma unroll 1
    for(unsigned base=0u;base<group_count;base+=Groups) {
        staged::LaneOperands operands[Groups];
        const unsigned count=group_count-base<Groups?group_count-base:Groups;
#pragma unroll
        for(unsigned i=0u;i<Groups;++i)if(i<count)operands[i]=staged::load(left[base+i],right[base+i]);
        bool eligible=count==Groups && transfer::regular(carry);
#pragma unroll
        for(unsigned i=0u;i<Groups;++i)if(i<count)
            eligible=eligible && dominant::eligible(carry,operands[i].left_control,operands[i].right_control);
        if(eligible) {
            const Value entry=carry;uint32_t sums[Groups];
#pragma unroll
            for(unsigned i=0u;i<Groups;++i) {
                const int units=int(int16_t(operands[i].left_control))+int(int16_t(operands[i].right_control));
                const uint32_t total=dominant::products<2u>(operands[i].left,operands[i].right,25-entry.exponent+units);
                sums[i]=qrt_sm121_lane_reduce::sum<4u>(total);
            }
            const bool last_negative=((operands[Groups-1u].left[0]^operands[Groups-1u].right[0])&0x8000u)!=0u;
            Value result;
            if(transfer::apply(entry,sums,last_negative,&result)) {
                if constexpr(Audit) {
                    ++stats.transfer_blocks;stats.transformed+=Groups;stats.reused_groups+=Groups;
                    int32_t current=entry.negative?-int32_t(entry.significand):int32_t(entry.significand);
#pragma unroll
                    for(unsigned i=0u;i+1u<Groups;++i) {
                        current+=transfer::increment(sums[i],entry.negative);
                        record(trace,base+i,{uint32_t(current<0?-current:current),entry.exponent,entry.negative});
                    }
                    record(trace,base+Groups-1u,result);
                }
                carry=result;
            }else {
                // Prealigned sums remain exact whenever the actual carry
                // still has the entry exponent, even if its sign has changed.
                // Other groups use the unchanged original staged recurrence.
#pragma unroll
                for(unsigned i=0u;i<Groups;++i) {
                    if(carry.exponent==entry.exponent) {
                        const bool negative=((operands[i].left[0]^operands[i].right[0])&0x8000u)!=0u;
                        carry=dominant::finish(carry,sums[i],negative,entry.exponent);
                        if constexpr(Audit)++stats.reused_groups;
                    }else {
                        carry=staged::accumulate(carry,operands[i]);
                        if constexpr(Audit)++stats.transition_replays;
                    }
                    if constexpr(Audit){++stats.transformed;record(trace,base+i,carry);}
                }
            }
        }else {
#pragma unroll
            for(unsigned i=0u;i<Groups;++i)if(i<count) {
                bool transformed;carry=staged::accumulate(carry,operands[i],Audit?&transformed:nullptr);
                if constexpr(Audit){stats.transformed+=transformed;stats.original+=!transformed;record(trace,base+i,carry);}
            }
        }
    }
    if constexpr(Audit)if(!(threadIdx.x&3u) && statistics)*statistics=stats;
    return threadIdx.x&3u?0.0f:qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
}
