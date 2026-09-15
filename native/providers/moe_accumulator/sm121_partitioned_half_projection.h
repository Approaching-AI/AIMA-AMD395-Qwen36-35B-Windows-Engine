#pragma once
#include "sm121_staged_half_projection.h"

// Isolated replay specialization. Preparation classifies whole rows before
// dispatch, so fast kernels never contain the extended original-group path.
// Class 0 requires unchanged staged replay, 1 allows zeros, 3 is all nonzero.
// A pair's class is the bitwise AND of its two independently checked rows.
namespace qrt_sm121_partitioned_half_projection {
namespace staged=qrt_sm121_staged_half_projection;
using Row=staged::Row;
using Value=staged::Value;
using Stats=staged::Stats;

__global__ void classify_rows(const Row* input,unsigned* flags,unsigned rows,unsigned width) {
    const unsigned row=blockIdx.x;if(row>=rows)return;
    __shared__ unsigned waves[8];
    unsigned classification=3u;
    for(unsigned group=threadIdx.x;group<width/16u;group+=256u) {
        const uint32_t control=input[size_t(row)*(width/16u)+group].control;
        classification&=int16_t(control)==-32768?0u:(control>>16u)==65535u?3u:1u;
    }
#pragma unroll
    for(unsigned offset=16u;offset;offset>>=1u)
        classification&=__shfl_xor(classification,offset,32u);
    if(!(threadIdx.x&31u))waves[threadIdx.x/32u]=classification;
    __syncthreads();
    if(!threadIdx.x) {
        unsigned result=3u;
#pragma unroll
        for(unsigned wave=0u;wave<8u;++wave)result&=waves[wave];
        flags[row]=result;
    }
}

template<bool AllNonzero>
__device__ __forceinline__ Value accumulate(Value carry,const staged::LaneOperands& operands) {
    if constexpr(AllNonzero)return staged::transformed_group<true>(carry,operands,65535u);
    const unsigned active=(operands.left_control&operands.right_control)>>16u;
    if(!active) {
        const int maximum=carry.exponent>-133?carry.exponent:-133;
        const unsigned shift=unsigned(maximum-carry.exponent);
        const uint32_t magnitude=shift>=32u?0u:(carry.significand<<2u)>>shift;
        return qrt_sm121_canonical::normalize(magnitude,magnitude&&carry.negative,maximum);
    }
    return active==65535u?staged::transformed_group<true>(carry,operands,active)
        :staged::transformed_group<false>(carry,operands,active);
}

// The caller must check pair class before entry: nonzero for the general
// variant, exactly 3 for AllNonzero. Rejected pairs use staged::dot<2u>.
template<bool AllNonzero,bool Audit=false>
__device__ __forceinline__ float dot(const Row* left,const Row* right,unsigned width,
    uint32_t* raw_trace=nullptr,Stats* stats=nullptr) {
    const unsigned lane=threadIdx.x&3u,groups=width/16u;
    Value carry{0u,-133,false};
#pragma unroll 1
    for(unsigned base=0u;base<groups;base+=2u) {
        staged::LaneOperands operands[2];
        const unsigned count=groups-base<2u?groups-base:2u;
#pragma unroll
        for(unsigned i=0u;i<2u;++i)if(i<count)operands[i]=staged::load(left[base+i],right[base+i]);
#pragma unroll
        for(unsigned i=0u;i<2u;++i)if(i<count) {
            carry=accumulate<AllNonzero>(carry,operands[i]);
            if constexpr(Audit)if(!lane&&raw_trace) {
                raw_trace[3u*(base+i)]=carry.significand;
                raw_trace[3u*(base+i)+1u]=uint32_t(int32_t(carry.exponent));
                raw_trace[3u*(base+i)+2u]=unsigned(carry.negative);
            }
        }
    }
    if constexpr(Audit)if(!lane&&stats)*stats=Stats{groups,0u};
    return lane?0.0f:qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_partitioned_half_projection
