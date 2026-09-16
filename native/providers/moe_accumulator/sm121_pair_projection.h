#pragma once
#include "sm121_pair_projection_products.h"
#include "sm121_lane_reduce.h"
namespace qrt_sm121_pair_projection {
namespace products=qrt_sm121_pair_projection_products;
using Row=products::Row;
using Pair=products::Pair;
using Value=products::Value;
struct Stats { unsigned recovered=0u,fallback=0u; };
struct LaneOperands {Pair a[2],b[2];};
__global__ void prepare_rows(const uint16_t* input,Row* output,unsigned rows,unsigned width) {
    const size_t group=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(group<size_t(rows)*(width/16u))output[group]=products::prepare(input+group*16u);
}
__device__ __forceinline__ LaneOperands load(const Row& a,const Row& b) {
    const unsigned offset=(threadIdx.x&3u)*2u;
    return {{a.pairs[offset],a.pairs[offset+1u]},{b.pairs[offset],b.pairs[offset+1u]}};
}
template<bool Audit>
__device__ __forceinline__ Value accumulate(Value carry,const LaneOperands& p,Stats& stats) {
    const int e0=products::maximum(p.a[0],p.b[0]),e1=products::maximum(p.a[1],p.b[1]);
    int exponent=qrt_sm121_lane_reduce::maximum<4u>(e0>e1?e0:e1);
    exponent=exponent>carry.exponent?exponent:carry.exponent;exponent=exponent>-133?exponent:-133;
    uint32_t total=0u;
#pragma unroll
    for(unsigned i=0;i<2;++i) {
        bool used;total+=products::aligned(p.a[i],p.b[i],exponent,Audit?&used:nullptr);
        if constexpr(Audit){stats.recovered+=used;stats.fallback+=!used;}
    }
    total=qrt_sm121_lane_reduce::sum<4u>(total)+products::aligned_original(carry,exponent);
    const bool negative=((p.a[0].half^p.b[0].half)&32768u)!=0u;
    const auto sum=qrt_sm121_group16::decode_modulo_sum(total,negative);
    return qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,exponent);
}
template<unsigned StagingGroups,bool Audit=false>
__device__ __forceinline__ float dot(const Row* a,const Row* b,unsigned width,
    uint32_t* trace=nullptr,Stats* output=nullptr) {
    static_assert(StagingGroups==1u||StagingGroups==2u||StagingGroups==4u||StagingGroups==8u);
    const unsigned lane=threadIdx.x&3u,groups=width/16u;Value carry{0u,-133,false};Stats stats;
#pragma unroll 1
    for(unsigned base=0;base<groups;base+=StagingGroups) {
        LaneOperands operands[StagingGroups];const unsigned count=groups-base<StagingGroups?groups-base:StagingGroups;
#pragma unroll
        for(unsigned i=0;i<StagingGroups;++i)if(i<count)operands[i]=load(a[base+i],b[base+i]);
#pragma unroll
        for(unsigned i=0;i<StagingGroups;++i)if(i<count) {
            carry=qrt_sm121_pair_projection::accumulate<Audit>(carry,operands[i],stats);
            if constexpr(Audit)if(!lane&&trace){trace[3u*(base+i)]=carry.significand;trace[3u*(base+i)+1u]=uint32_t(int32_t(carry.exponent));trace[3u*(base+i)+2u]=carry.negative;}
        }
    }
    if constexpr(Audit) {
        stats.recovered=qrt_sm121_lane_reduce::sum<4u>(stats.recovered);stats.fallback=qrt_sm121_lane_reduce::sum<4u>(stats.fallback);
        if(!lane&&output)*output=stats;
    }
    return lane?0.0f:qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
}
