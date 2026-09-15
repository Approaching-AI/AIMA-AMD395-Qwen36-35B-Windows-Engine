#pragma once
#include "sm121_staged_half_projection.h"
#include "sm121_embedded_half_operands.h"

// Component candidate: replace nine-dword prepared rows with eight dwords,
// retaining the established four-lane arithmetic and per-group fallback.
namespace qrt_sm121_embedded_half_projection {
namespace compact=qrt_sm121_embedded_half;
namespace staged=qrt_sm121_staged_half_projection;
using Row=compact::Row;
using Value=qrt_q1_moe_hawkeye::Value;
struct Stats { unsigned compact_groups=0u,original_groups=0u; };

// A CTA owns one row and its bitmap. Flags[0] is the complete-row fast-path
// certificate; the remaining words identify individual supported K16 groups.
__global__ void prepare_rows(const uint16_t* input,Row* output,unsigned* flags,
    unsigned rows,unsigned width) {
    const unsigned row=blockIdx.x,tid=threadIdx.x,groups=width/16u;
    if(row>=rows)return;
    __shared__ unsigned supported[16],invalid;
    if(tid<16u)supported[tid]=0u;
    if(!tid)invalid=0u;
    __syncthreads();
    for(unsigned group=tid;group<groups;group+=blockDim.x) {
        Row packed;
        const bool valid=compact::prepare(input+size_t(row)*width+group*16u,&packed);
        output[size_t(row)*groups+group]=packed;
        if(valid)atomicOr(&supported[group/32u],1u<<(group&31u));
        else atomicOr(&invalid,1u);
    }
    __syncthreads();
    const unsigned words=compact::flag_words(width);
    if(!tid)flags[size_t(row)*words]=invalid==0u;
    if(tid+1u<words)flags[size_t(row)*words+1u+tid]=supported[tid];
}
inline hipError_t prepare(const uint16_t* input,size_t input_words,
    Row* output,size_t output_groups,unsigned* flags,size_t flags_capacity,
    unsigned rows,unsigned width,hipStream_t stream) {
    if(!input || !output || !flags || !rows || rows>16384u ||
        !width || width>8192u || width%16u ||
        input_words<size_t(rows)*width || output_groups<size_t(rows)*(width/16u) ||
        flags_capacity<size_t(rows)*compact::flag_words(width))return hipErrorInvalidValue;
    hipLaunchKernelGGL(prepare_rows,dim3(rows),dim3(256u),0u,stream,input,output,flags,rows,width);
    return hipGetLastError();
}

struct LaneOperands { staged::LaneOperands operands; unsigned left_meta,right_meta; };
__device__ __forceinline__ LaneOperands load(const Row& left,const Row& right) {
    const unsigned offset=(threadIdx.x&3u)*2u;
    LaneOperands p;
    __builtin_memcpy(p.operands.left,left.pairs+offset,8u);
    __builtin_memcpy(p.operands.right,right.pairs+offset,8u);
    p.left_meta=compact::metadata(p.operands.left[0],p.operands.left[1]);
    p.right_meta=compact::metadata(p.operands.right[0],p.operands.right[1]);
    p.operands.left_control=uint16_t(int(p.left_meta&255u)-142);
    p.operands.right_control=uint16_t(int(p.right_meta&255u)-142);
    for(unsigned i=0u;i<2u;++i) {
        p.operands.left[i]&=compact::payload_mask;
        p.operands.right[i]&=compact::payload_mask;
    }
    return p;
}
__device__ __forceinline__ Value accumulate(Value carry,LaneOperands p) {
    if((p.left_meta&p.right_meta)&256u) {
        p.operands.left_control|=0xffff0000u;p.operands.right_control|=0xffff0000u;
    }else{
        unsigned local=0u;
#pragma unroll
        for(unsigned i=0u;i<4u;++i) {
            const unsigned shift=(i&1u)*16u;
            const bool live=((p.operands.left[i/2u]>>shift)&0x7fffu) &&
                ((p.operands.right[i/2u]>>shift)&0x7fffu);
            local|=unsigned(live)<<i;
        }
        // Disjoint four-bit fields make this unsigned sum an exact OR.
        const unsigned active=qrt_sm121_lane_reduce::sum<4u>(local<<((threadIdx.x&3u)*4u));
        p.operands.left_control|=active<<16u;p.operands.right_control|=active<<16u;
    }
    return staged::accumulate(carry,p.operands);
}

template<unsigned StagingGroups,bool AllGroups,bool Audit>
__device__ __forceinline__ float dot_body(const Row* left,const Row* right,
    const uint16_t* original_left,const uint16_t* original_right,
    const unsigned* left_flags,const unsigned* right_flags,unsigned width,
    uint32_t* trace,Stats* stats) {
    static_assert(StagingGroups==2u || StagingGroups==4u);
    const unsigned groups=width/16u,lane=threadIdx.x&3u;
    Value carry{0u,-133,false};Stats counts;
#pragma unroll 1
    for(unsigned base=0u;base<groups;base+=StagingGroups) {
        LaneOperands operands[StagingGroups];bool valid[StagingGroups];
        const unsigned count=groups-base<StagingGroups?groups-base:StagingGroups;
#pragma unroll
        for(unsigned i=0u;i<StagingGroups;++i)if(i<count) {
            if constexpr(AllGroups)valid[i]=true;
            else valid[i]=((left_flags[1u+(base+i)/32u]&right_flags[1u+(base+i)/32u])>>((base+i)&31u))&1u;
            if(valid[i])operands[i]=load(left[base+i],right[base+i]);
        }
#pragma unroll
        for(unsigned i=0u;i<StagingGroups;++i)if(i<count) {
            if(valid[i])carry=accumulate(carry,operands[i]);
            else carry=qrt_sm121_subgroup::accumulate<4u>(carry,
                original_left+(base+i)*16u,original_right+(base+i)*16u);
            if constexpr(Audit) {
                counts.compact_groups+=valid[i];counts.original_groups+=!valid[i];
                if(!lane && trace) {
                    trace[(base+i)*3u]=carry.significand;
                    trace[(base+i)*3u+1u]=uint32_t(int32_t(carry.exponent));
                    trace[(base+i)*3u+2u]=unsigned(carry.negative);
                }
            }
        }
    }
    if constexpr(Audit)if(!lane && stats)*stats=counts;
    return lane?0.0f:qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
template<unsigned StagingGroups,bool Audit=false>
__device__ __forceinline__ float dot(const Row* left,const Row* right,
    const uint16_t* original_left,const uint16_t* original_right,
    const unsigned* left_flags,const unsigned* right_flags,unsigned width,
    uint32_t* trace=nullptr,Stats* stats=nullptr) {
    if(left_flags[0] && right_flags[0])return dot_body<StagingGroups,true,Audit>(
        left,right,original_left,original_right,left_flags,right_flags,width,trace,stats);
    return dot_body<StagingGroups,false,Audit>(left,right,original_left,original_right,
        left_flags,right_flags,width,trace,stats);
}
} // namespace qrt_sm121_embedded_half_projection
