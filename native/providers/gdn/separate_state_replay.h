#pragma once
#include "blackwell_lifetime_matrices.h"
#include "../moe_accumulator/sm121_narrow_f32_carry.h"

// Isolated candidate: no runtime dispatch. The narrow fast kernel has no
// compiled integer fallback. A separate kernel replays rejected CTAs using
// the original arithmetic. Each CTA owns eight complete value columns;
// global recurrent state is published only after every chunk succeeds.
namespace qrt_fla_separate_state {
namespace scalar=qrt_fla_blackwell_scalar;
namespace narrow=qrt_sm121_narrow_f32_carry;
using scalar::from_bf16;using scalar::to_bf16;using scalar::exponential;
using scalar::pack;using scalar::threads;using scalar::columns;

__device__ __forceinline__ bool admitted(uint16_t a,uint16_t b){
    return narrow::eligible(a)&&narrow::eligible(b);
}
template<unsigned Width,unsigned Columns>
__device__ __forceinline__ float narrow_dot(const uint32_t* left,
    const uint32_t (&right)[Width/2u][Columns],unsigned column){
    static_assert(Width==64u||Width==128u);
    float carry=0.0f;
    for(unsigned base=0u;base<Width;base+=16u){
        qrt_sm121_float_alignment::Group group;
#pragma unroll
        for(unsigned i=0u;i<16u;i+=2u){
            const uint32_t a=left[(base+i)/2u],b=right[(base+i)/2u][column];
            group.set(i,uint16_t(a),uint16_t(b));
            group.set(i+1u,uint16_t(a>>16u),uint16_t(b>>16u));
        }
        carry=narrow::accumulate(carry,group);
    }
    return carry;
}

template<unsigned Columns>
__global__ void fast_kernel(const uint16_t* k,const uint16_t* u,const uint16_t* w,
    const float* g,uint16_t* h,uint16_t* v_new,float* state,unsigned count,
    const unsigned char* table,unsigned* completed){
    static_assert(Columns==4u||Columns==8u);
    __shared__ float current[Columns][128],decay[64],segment_decay;
    __shared__ uint32_t rounded[64][Columns],residual[32][Columns];
    __shared__ uint16_t residual_words[64][Columns];
    union Operands {uint32_t weights[64][65];uint32_t keys[128][33];};
    __shared__ Operands operands;
    __shared__ unsigned accepted;
    const unsigned tid=threadIdx.x,head=blockIdx.y,first_column=blockIdx.x*Columns;
    const unsigned receipt=head*(128u/Columns)+blockIdx.x;
    if(!tid)completed[receipt]=0u;
    for(unsigned cell=tid;cell<Columns*128u;cell+=threads)
        current[cell/128u][cell%128u]=state[(head*128u+first_column+cell/128u)*128u+cell%128u];
    __syncthreads();
    for(unsigned offset=0u;offset<count;offset+=64u){
        const unsigned valid=min(64u,count-offset);
        if(!tid){accepted=1u;segment_decay=exponential(g[size_t(offset+valid-1u)*32u+head],table);}
        if(tid<valid)decay[tid]=exponential(g[size_t(offset+valid-1u)*32u+head]-g[size_t(offset+tid)*32u+head],table);
        __syncthreads();
        unsigned local_ok=1u;
        for(unsigned cell=tid;cell<64u*Columns;cell+=threads){
            const unsigned pair=cell/Columns,column=cell%Columns;
            const uint16_t a=to_bf16(current[column][pair*2u]),b=to_bf16(current[column][pair*2u+1u]);
            rounded[pair][column]=pack(a,b);
            const size_t index=size_t(offset/64u)*524288u+(head*128u+first_column+column)*128u+pair*2u;
            h[index]=a;h[index+1u]=b;
            if(!admitted(a,b))local_ok=0u;
        }
        for(unsigned cell=tid;cell<64u*64u;cell+=threads){
            const unsigned row=cell/64u,pair=cell%64u;
            uint16_t a=0u,b=0u;
            if(row<valid){const size_t index=(size_t(offset+row)*32u+head)*128u+pair*2u;a=w[index];b=w[index+1u];}
            operands.weights[row][pair]=pack(a,b);
            if(!admitted(a,b))local_ok=0u;
        }
        if(!local_ok)atomicAnd(&accepted,0u);
        __syncthreads();
        if(!accepted)return; // Uniform CTA exit; global state is still original.
        for(unsigned cell=tid;cell<64u*Columns;cell+=threads){
            const unsigned row=cell/Columns,column=cell%Columns;
            uint16_t value=0u;
            if(row<valid){
                const float sum=narrow_dot<128u,Columns>(operands.weights[row],rounded,column);
                const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;
                const float difference=from_bf16(u[index])-sum;
                v_new[index]=to_bf16(difference);
                value=to_bf16(difference*decay[row]);
            }
            residual_words[row][column]=value;
        }
        __syncthreads();
        local_ok=1u;
        for(unsigned cell=tid;cell<128u*32u;cell+=threads){
            const unsigned feature=cell/32u,pair=cell%32u;
            uint16_t a=0u,b=0u;
            if(pair*2u<valid)a=k[(size_t(offset+pair*2u)*16u+head/2u)*128u+feature];
            if(pair*2u+1u<valid)b=k[(size_t(offset+pair*2u+1u)*16u+head/2u)*128u+feature];
            operands.keys[feature][pair]=pack(a,b);
            if(!admitted(a,b))local_ok=0u;
        }
        for(unsigned cell=tid;cell<32u*Columns;cell+=threads){
            const unsigned pair=cell/Columns,column=cell%Columns;
            const uint16_t a=residual_words[pair*2u][column],b=residual_words[pair*2u+1u][column];
            residual[pair][column]=pack(a,b);
            if(!admitted(a,b))local_ok=0u;
        }
        if(!local_ok)atomicAnd(&accepted,0u);
        __syncthreads();
        if(!accepted)return;
        for(unsigned cell=tid;cell<128u*Columns;cell+=threads){
            const unsigned feature=cell/Columns,column=cell%Columns;
            const float sum=narrow_dot<64u,Columns>(operands.keys[feature],residual,column);
            current[column][feature]=fmaf(current[column][feature],segment_decay,sum);
        }
        __syncthreads();
    }
    for(unsigned cell=tid;cell<Columns*128u;cell+=threads)
        state[(head*128u+first_column+cell/128u)*128u+cell%128u]=current[cell/128u][cell%128u];
    __syncthreads();
    if(!tid)completed[receipt]=1u;
}

// The replay body below is the retained shared-arena state kernel, with only
// a uniform receipt check added before any shared memory or tensor access.
// It overwrites every H and v_new cell owned by a rejected CTA before the
// caller can launch output consumers. Successful CTAs are never replayed.
// Keep this isolated copy byte-checked against the retained kernel in tests.
using scalar::eligible;using scalar::dot;
template<unsigned Columns>
__global__ void replay_kernel(const uint16_t* k,const uint16_t* u,const uint16_t* w,
    const float* g,uint16_t* h,uint16_t* v_new,float* state,unsigned count,
    const unsigned char* table,const unsigned* completed) {
    static_assert(Columns==4u || Columns==8u);
    if(completed[blockIdx.y*(128u/Columns)+blockIdx.x])return;
    __shared__ float current[Columns][128],decay[64],segment_decay;
    __shared__ uint32_t rounded[64][Columns],residual[32][Columns];
    __shared__ uint16_t residual_words[64][Columns];
    union Operands { uint32_t weights[64][65]; uint32_t keys[128][33]; };
    __shared__ Operands operands;
    __shared__ unsigned key_ok[128],weight_ok[64],state_ok[Columns],residual_ok[Columns];
    const unsigned tid=threadIdx.x,head=blockIdx.y,first_column=blockIdx.x*Columns;
    for (unsigned cell=tid;cell<Columns*128u;cell+=threads)
        current[cell/128u][cell%128u]=state[(head*128u+first_column+cell/128u)*128u+cell%128u];
    __syncthreads();
    for (unsigned offset=0u;offset<count;offset+=64u) {
        const unsigned valid=min(64u,count-offset);
        if (tid<128u) key_ok[tid]=1u;
        if (tid<64u) weight_ok[tid]=1u;
        if (tid<Columns) {state_ok[tid]=1u;residual_ok[tid]=1u;}
        if (tid<valid) decay[tid]=exponential(g[size_t(offset+valid-1u)*32u+head]-g[size_t(offset+tid)*32u+head],table);
        if (!tid) segment_decay=exponential(g[size_t(offset+valid-1u)*32u+head],table);
        __syncthreads();
        for (unsigned cell=tid;cell<64u*Columns;cell+=threads) {
            const unsigned pair=cell/Columns,column=cell%Columns;
            const uint16_t a=to_bf16(current[column][pair*2u]),b=to_bf16(current[column][pair*2u+1u]);
            rounded[pair][column]=pack(a,b);
            const size_t index=size_t(offset/64u)*524288u+(head*128u+first_column+column)*128u+pair*2u;
            h[index]=a;h[index+1u]=b;
            if (!eligible(a,b)) atomicAnd(&state_ok[column],0u);
        }
        for (unsigned cell=tid;cell<64u*64u;cell+=threads) {
            const unsigned row=cell/64u,pair=cell%64u;
            uint16_t a=0u,b=0u;
            if (row<valid) {
                const size_t index=(size_t(offset+row)*32u+head)*128u+pair*2u;
                a=w[index];b=w[index+1u];
            }
            operands.weights[row][pair]=pack(a,b);
            if (!eligible(a,b)) atomicAnd(&weight_ok[row],0u);
        }
        __syncthreads();
        for (unsigned cell=tid;cell<64u*Columns;cell+=threads) {
            const unsigned row=cell/Columns,column=cell%Columns;
            uint16_t value=0u;
            if (row<valid) {
                const float sum=dot<128u,Columns>(operands.weights[row],rounded,column,weight_ok[row]&&state_ok[column]);
                const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;
                const float difference=from_bf16(u[index])-sum;
                v_new[index]=to_bf16(difference);
                value=to_bf16(difference*decay[row]);
            }
            residual_words[row][column]=value;
        }
        __syncthreads();
        // Every W/H dot has completed before this barrier. Reusing the
        // operand arena for K cannot change the already produced residuals.
        for (unsigned cell=tid;cell<128u*32u;cell+=threads) {
            const unsigned feature=cell/32u,pair=cell%32u;
            uint16_t a=0u,b=0u;
            if (pair*2u<valid) a=k[(size_t(offset+pair*2u)*16u+head/2u)*128u+feature];
            if (pair*2u+1u<valid) b=k[(size_t(offset+pair*2u+1u)*16u+head/2u)*128u+feature];
            operands.keys[feature][pair]=pack(a,b);
            if (!eligible(a,b)) atomicAnd(&key_ok[feature],0u);
        }
        for (unsigned cell=tid;cell<32u*Columns;cell+=threads) {
            const unsigned pair=cell/Columns,column=cell%Columns;
            const uint16_t a=residual_words[pair*2u][column],b=residual_words[pair*2u+1u][column];
            residual[pair][column]=pack(a,b);
            if (!eligible(a,b)) atomicAnd(&residual_ok[column],0u);
        }
        __syncthreads();
        for (unsigned cell=tid;cell<128u*Columns;cell+=threads) {
            const unsigned feature=cell/Columns,column=cell%Columns;
            const float sum=dot<64u,Columns>(operands.keys[feature],residual,column,key_ok[feature]&&residual_ok[column]);
            current[column][feature]=fmaf(current[column][feature],segment_decay,sum);
        }
        __syncthreads();
    }
    for (unsigned cell=tid;cell<Columns*128u;cell+=threads)
        state[(head*128u+first_column+cell/128u)*128u+cell%128u]=current[cell/128u][cell%128u];
}

} // namespace qrt_fla_separate_state
