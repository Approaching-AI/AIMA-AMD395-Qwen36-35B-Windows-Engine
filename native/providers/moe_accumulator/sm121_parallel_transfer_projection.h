#pragma once
#include "sm121_transfer_projection.h"

// Isolated geometry replacement: each four-lane team prepares one K16 group,
// and 2/4/8 adjacent teams share an output. Exact fixed-binade increments are
// scanned across teams. Every interior carry is checked before publication.
// A declined window reuses its aligned sums only at the original exponent,
// otherwise it executes the unchanged staged K16 accumulator.
namespace qrt_sm121_parallel_transfer_projection {
namespace staged=qrt_sm121_staged_half_projection;
namespace dominant=qrt_sm121_dominant_half_products;
namespace transfer=qrt_sm121_carry_transfer;
using Row=staged::Row;
using Value=staged::Value;
using Stats=qrt_sm121_transfer_projection::Stats;

template<unsigned Width>
__device__ __forceinline__ bool all(bool value) {
    static_assert(Width==8u || Width==16u || Width==32u);
    constexpr unsigned mask=0xffffffffu>>(32u-Width);
    const unsigned offset=(threadIdx.x%32u)/Width*Width;
    return ((unsigned(__ballot(value))>>offset)&mask)==mask;
}

template<unsigned Width>
__device__ __forceinline__ staged::LaneOperands broadcast(
    const staged::LaneOperands& local,unsigned team) {
    const unsigned source=team*4u+(threadIdx.x&3u);
    staged::LaneOperands result;
#pragma unroll
    for(unsigned i=0u;i<2u;++i) {
        result.left[i]=__shfl(local.left[i],source,Width);
        result.right[i]=__shfl(local.right[i],source,Width);
    }
    result.left_control=__shfl(local.left_control,source,Width);
    result.right_control=__shfl(local.right_control,source,Width);
    return result;
}

template<unsigned Groups,bool Audit=false>
__device__ __forceinline__ float dot(const Row* left,const Row* right,unsigned width,
    uint32_t* trace=nullptr,Stats* statistics=nullptr) {
    static_assert(Groups==2u || Groups==4u || Groups==8u);
    constexpr unsigned lanes=Groups*4u;
    const unsigned lane=threadIdx.x%lanes,team=lane/4u,count_groups=width/16u;
    Value carry{0u,-133,false};Stats stats;
#pragma unroll 1
    for(unsigned base=0u;base<count_groups;base+=Groups) {
        const unsigned count=count_groups-base<Groups?count_groups-base:Groups;
        staged::LaneOperands operands{};
        if(team<count)operands=staged::load(left[base+team],right[base+team]);
        const Value entry=carry;
        const bool eligible=all<lanes>(count==Groups && transfer::regular(entry) &&
            dominant::eligible(entry,operands.left_control,operands.right_control));
        uint32_t total=0u;
        bool accepted=false;
        if(eligible) {
            const int units=int(int16_t(operands.left_control))+int(int16_t(operands.right_control));
            total=qrt_sm121_lane_reduce::sum<4u>(dominant::products<2u>(
                operands.left,operands.right,25-entry.exponent+units));
            const bool product_valid=transfer::product_sum(total);
            const int32_t step=team+1u<Groups && product_valid
                ?transfer::increment(total,entry.negative):0;
            const bool step_valid=step>-0x800000 && step<0x800000;
            // Invalid steps contribute zero to the diagnostic scan, preventing
            // signed overflow even for rejected windows. They cannot certify.
            int32_t prefix=step_valid?step:0;
#pragma unroll
            for(unsigned distance=1u;distance<Groups;distance*=2u) {
                const int32_t before=__shfl_up(prefix,distance*4u,lanes);
                if(team>=distance)prefix+=before;
            }
            const int32_t current=(entry.negative?-int32_t(entry.significand):int32_t(entry.significand))+prefix;
            const bool interior=team+1u==Groups || (entry.negative
                ?current<=-0x800000 && current>-0x1000000
                :current>=0x800000 && current<0x1000000);
            accepted=all<lanes>(product_valid && step_valid && interior);
            if(accepted) {
                const uint32_t last=__shfl(total,(Groups-1u)*4u,lanes);
                const int32_t last_entry=__shfl(current,(Groups-1u)*4u,lanes);
                const bool negative=__shfl((operands.left[0]^operands.right[0])&0x8000u,
                    (Groups-1u)*4u,lanes)!=0u;
                const auto sum=qrt_sm121_group16::decode_modulo_sum(uint32_t(last_entry)*4u+last,negative);
                carry=qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,entry.exponent);
                if constexpr(Audit) {
                    ++stats.transfer_blocks;stats.transformed+=Groups;stats.reused_groups+=Groups;
                    if(!(lane&3u) && trace) {
                        const Value state=team+1u==Groups?carry:Value{
                            uint32_t(current<0?-current:current),entry.exponent,entry.negative};
                        trace[3u*(base+team)]=state.significand;
                        trace[3u*(base+team)+1u]=uint32_t(int32_t(state.exponent));
                        trace[3u*(base+team)+2u]=unsigned(state.negative);
                    }
                }
            }
        }
        if(!accepted) {
#pragma unroll
            for(unsigned group=0u;group<Groups;++group)if(group<count) {
                const auto selected=broadcast<lanes>(operands,group);
                bool transformed=true;
                if(eligible && carry.exponent==entry.exponent) {
                    const uint32_t selected_sum=__shfl(total,group*4u,lanes);
                    const bool negative=((selected.left[0]^selected.right[0])&0x8000u)!=0u;
                    carry=dominant::finish(carry,selected_sum,negative,entry.exponent);
                    if constexpr(Audit)++stats.reused_groups;
                }else {
                    carry=staged::accumulate(carry,selected,Audit?&transformed:nullptr);
                    if constexpr(Audit)stats.transition_replays+=eligible;
                }
                if constexpr(Audit) {
                    stats.transformed+=transformed;stats.original+=!transformed;
                    if(!lane && trace) {
                        trace[3u*(base+group)]=carry.significand;
                        trace[3u*(base+group)+1u]=uint32_t(int32_t(carry.exponent));
                        trace[3u*(base+group)+2u]=unsigned(carry.negative);
                    }
                }
            }
        }
    }
    if constexpr(Audit)if(!lane && statistics)*statistics=stats;
    return lane?0.0f:qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_parallel_transfer_projection
