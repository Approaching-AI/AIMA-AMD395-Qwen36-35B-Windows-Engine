#pragma once
#include "blackwell_scalar_state.h"

// Shared arena lifetime change. The original scalar dot and all numerical
// operations remain unchanged; only nonoverlapping shared arrays reuse space.
namespace qrt_fla_lifetime {
namespace scalar=qrt_fla_blackwell_scalar;
using scalar::from_bf16;using scalar::to_bf16;using scalar::exponential;
using scalar::eligible;using scalar::pack;using scalar::dot;
using scalar::threads;using scalar::columns;
template<unsigned Columns>
__global__ void state_kernel(const uint16_t* k,const uint16_t* u,const uint16_t* w,
    const float* g,uint16_t* h,uint16_t* v_new,float* state,unsigned count,
    const unsigned char* table) {
    static_assert(Columns==4u || Columns==8u);
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

// Query/checkpoint reads finish for every CTA thread before the same arenas
// receive scores/values. Two original Q/H results remain in private registers.
__global__ void output_kernel(const uint16_t* q,const uint16_t* v,const uint16_t* h,
    const float* g,const uint16_t* scores,float* output,unsigned count,
    const unsigned char* table) {
    union Left { uint32_t queries[64][65]; uint32_t scores[64][33]; };
    union Right { uint32_t checkpoint[64][columns]; uint32_t values[32][columns]; };
    __shared__ Left left;__shared__ Right right;
    __shared__ unsigned q_ok[64],score_ok[64],v_ok[columns],h_ok[columns];
    const unsigned tid=threadIdx.x,offset=blockIdx.z*64u,head=blockIdx.y;
    const unsigned first_column=blockIdx.x*columns,valid=min(64u,count-offset);
    if(tid<64u){q_ok[tid]=1u;score_ok[tid]=1u;}
    if(tid<columns){v_ok[tid]=1u;h_ok[tid]=1u;}
    __syncthreads();
    for(unsigned cell=tid;cell<64u*64u;cell+=threads){
        const unsigned row=cell/64u,pair=cell%64u;uint16_t a=0u,b=0u;
        if(row<valid){const size_t index=(size_t(offset+row)*16u+head/2u)*128u+pair*2u;a=q[index];b=q[index+1u];}
        left.queries[row][pair]=pack(a,b);
        if(!eligible(a,b))atomicAnd(&q_ok[row],0u);
    }
    for(unsigned cell=tid;cell<64u*columns;cell+=threads){
        const unsigned pair=cell/columns,column=cell%columns;
        const size_t index=size_t(blockIdx.z)*524288u+(head*128u+first_column+column)*128u+pair*2u;
        const uint16_t a=h[index],b=h[index+1u];right.checkpoint[pair][column]=pack(a,b);
        if(!eligible(a,b))atomicAnd(&h_ok[column],0u);
    }
    __syncthreads();
    float old[2]{};
#pragma unroll
    for(unsigned slot=0u;slot<2u;++slot){
        const unsigned cell=tid+slot*threads,row=cell/columns,column=cell%columns;
        if(row<valid)old[slot]=dot<128u>(left.queries[row],right.checkpoint,column,q_ok[row]&&h_ok[column]);
    }
    // This is a whole-CTA barrier, including threads with no valid output.
    __syncthreads();
    for(unsigned cell=tid;cell<64u*32u;cell+=threads){
        const unsigned row=cell/32u,pair=cell%32u;uint16_t a=0u,b=0u;
        if(row<valid){const size_t index=(size_t(offset+row)*32u+head)*64u+pair*2u;a=scores[index];b=scores[index+1u];}
        left.scores[row][pair]=pack(a,b);
        if(!eligible(a,b))atomicAnd(&score_ok[row],0u);
    }
    for(unsigned cell=tid;cell<32u*columns;cell+=threads){
        const unsigned pair=cell/columns,column=cell%columns;uint16_t words[2]{};
#pragma unroll
        for(unsigned half=0u;half<2u;++half)
            if(pair*2u+half<valid)words[half]=v[(size_t(offset+pair*2u+half)*32u+head)*128u+first_column+column];
        right.values[pair][column]=pack(words[0],words[1]);
        if(!eligible(words[0],words[1]))atomicAnd(&v_ok[column],0u);
    }
    __syncthreads();
#pragma unroll
    for(unsigned slot=0u;slot<2u;++slot){
        const unsigned cell=tid+slot*threads,row=cell/columns,column=cell%columns;
        if(row<valid){
            const float local=dot<64u>(left.scores[row],right.values,column,score_ok[row]&&v_ok[column]);
            constexpr float scale=0.08838834764831845f;
            const float prior=old[slot]*exponential(g[size_t(offset+row)*32u+head],table);
            output[(size_t(offset+row)*32u+head)*128u+first_column+column]=from_bf16(to_bf16(fmaf(local,scale,prior*scale)));
        }
    }
}
} // namespace qrt_fla_lifetime
