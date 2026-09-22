#pragma once
#include "sm121_scaled_half_projection.h"
#include <cstring>

// A lane holds only its four operands from each side of a K16 group.
// Prefetch several compact groups before computing their ordered carries.
// The caller owns preparation and bounded replay dispatch.
namespace qrt_sm121_staged_half_projection {
namespace half=qrt_sm121_scaled_half_products;
using Row=half::Row;
using Value=qrt_q1_moe_hawkeye::Value;
using Stats=qrt_sm121_scaled_half_projection::Stats;
struct LaneOperands { uint32_t left[2],right[2],left_control,right_control; };
static_assert(sizeof(LaneOperands)==24u);

__device__ __forceinline__ LaneOperands load(const Row& left,const Row& right) {
    const unsigned pair=(threadIdx.x&3u)*2u;LaneOperands operands;
    __builtin_memcpy(operands.left,left.pairs+pair,8u);
    __builtin_memcpy(operands.right,right.pairs+pair,8u);
    operands.left_control=left.control;operands.right_control=right.control;
    return operands;
}
__device__ __forceinline__ uint16_t original(const uint32_t* pairs,uint32_t control,unsigned i) {
    const uint16_t x=uint16_t(pairs[i/2u]>>(i%2u*16u));const int unit=int(int16_t(control));
    if(unit==-32768)return x;
    if(!(x&0x7fffu))return uint16_t(x&0x8000u);
    return uint16_t((x&0x8000u)|(unsigned(int((x>>10u)&31u)+112+unit)<<7u)|((x&1023u)>>3u));
}
__device__ __forceinline__ Value original_group(Value carry,const LaneOperands& operands) {
    uint32_t products[4];
#pragma unroll
    for(unsigned i=0u;i<4u;++i)products[i]=qrt_sm121_group16::pack_product(
        qrt_q1_moe_hawkeye::multiply_bf16(original(operands.left,operands.left_control,i),
            original(operands.right,operands.right_control,i),-133));
    return qrt_sm121_subgroup::accumulate_products<4u>(carry,products);
}
template<bool AllNonzero>
__device__ __forceinline__ Value transformed_group(Value carry,const LaneOperands& operands,unsigned active) {
    const unsigned lane=threadIdx.x&3u;float products[4];uint32_t paired_maximum=0u;
#pragma unroll
    for(unsigned i=0u;i<2u;++i) {
        const uint32_t a=operands.left[i],b=operands.right[i];
        products[2u*i]=half::product<false>(a,b);products[2u*i+1u]=half::product<true>(a,b);
        uint32_t exponents=((a>>10u)&0x001f001fu)+((b>>10u)&0x001f001fu);
        if constexpr(!AllNonzero) {
            const unsigned bits=(active>>(lane*4u+2u*i))&3u;
            exponents&=((bits&1u)|((bits&2u)<<15u))*65535u;
        }
        paired_maximum=qrt_sm121_prepared_integer_pairs::maximum_pair(paired_maximum,exponents);
    }
    int maximum=int((paired_maximum&65535u)>(paired_maximum>>16u)?paired_maximum&65535u:paired_maximum>>16u);
    maximum=qrt_sm121_lane_reduce::maximum<4u>(maximum);
    const int combined_unit=int(int16_t(operands.left_control))+int(int16_t(operands.right_control));
    maximum=maximum-30+combined_unit;
    maximum=maximum>carry.exponent?maximum:carry.exponent;maximum=maximum>-133?maximum:-133;
    const int power=25-maximum+combined_unit;uint32_t total=0u;
    if(power>=-126) {
        const float scale=qrt_sm121_float_alignment::from_bits(unsigned(127+power)<<23u);
#pragma unroll
        for(unsigned i=0u;i<4u;++i)total+=uint32_t(int32_t(products[i]*scale));
    }
    total=qrt_sm121_lane_reduce::sum<4u>(total);
    const unsigned shift=unsigned(maximum-carry.exponent);
    const uint32_t aligned=shift>=32u?0u:(carry.significand<<2u)>>shift;
    total+=carry.negative?0u-aligned:aligned;
    const bool negative=((operands.left[0]^operands.right[0])&0x8000u)!=0u;
    const auto sum=qrt_sm121_group16::decode_modulo_sum(total,negative);
    return qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,maximum);
}
__device__ __forceinline__ Value accumulate(Value carry,const LaneOperands& operands,bool* transformed=nullptr) {
    if(int16_t(operands.left_control)==-32768 || int16_t(operands.right_control)==-32768) {
        if(transformed)*transformed=false;
        return original_group(carry,operands);
    }
    if(transformed)*transformed=true;
    const unsigned active=(operands.left_control&operands.right_control)>>16u;
    if(!active) {
        const int maximum=carry.exponent>-133?carry.exponent:-133;
        const unsigned shift=unsigned(maximum-carry.exponent);
        const uint32_t magnitude=shift>=32u?0u:(carry.significand<<2u)>>shift;
        return qrt_sm121_canonical::normalize(magnitude,magnitude && carry.negative,maximum);
    }
    return active==65535u?transformed_group<true>(carry,operands,active):transformed_group<false>(carry,operands,active);
}
template<unsigned StagingGroups,bool Audit=false,bool StridedRight=false>
__device__ __forceinline__ float dot(const Row* left,const Row* right,unsigned width,
    uint32_t* raw_trace=nullptr,Stats* stats=nullptr,unsigned right_stride=1u) {
    static_assert(StagingGroups==1u || StagingGroups==2u || StagingGroups==4u || StagingGroups==8u);
    const unsigned lane=threadIdx.x&3u,groups=width/16u;Value carry{0u,-133,false};Stats counts;
#pragma unroll 1
    for(unsigned base=0u;base<groups;base+=StagingGroups) {
        LaneOperands operands[StagingGroups];
        const unsigned count=groups-base<StagingGroups?groups-base:StagingGroups;
#pragma unroll
        for(unsigned i=0u;i<StagingGroups;++i)if(i<count)operands[i]=load(
            left[base+i],right[std::size_t(base+i)*(StridedRight?right_stride:1u)]);
#pragma unroll
        for(unsigned i=0u;i<StagingGroups;++i)if(i<count) {
            bool transformed;carry=accumulate(carry,operands[i],Audit?&transformed:nullptr);
            if constexpr(Audit) {
                counts.transformed+=transformed;counts.original+=!transformed;
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
} // namespace qrt_sm121_staged_half_projection
