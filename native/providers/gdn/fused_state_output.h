#ifndef QRT_FLA_FUSED_STATE_OUTPUT_H
#define QRT_FLA_FUSED_STATE_OUTPUT_H
#include "blackwell_scalar_state.h"

// Component candidate. Retain current columns across the segment and reuse
// old rounded state and unscaled residuals for output before the next chunk.
// W/query and K/score storage share lifetimes without changing dot, decay,
// BF16 conversion, or FMA order. Capture mode preserves all former H/V writes.
namespace qrt_fla_fused_state_output {
using namespace qrt_fla_blackwell_scalar;
template<unsigned Columns,bool Capture>
__global__ void kernel(const uint16_t* q,const uint16_t* k,const uint16_t* u,const uint16_t* w,
    const float* g,const uint16_t* scores,float* output,uint16_t* h,uint16_t* v_new,float* state,unsigned count,
    const unsigned char* table) {
    static_assert(Columns==4u || Columns==8u);
    __shared__ float current[Columns][128],decay[64],segment_decay;
    __shared__ uint32_t rounded[64][Columns],residual[32][Columns];
    __shared__ uint16_t residual_words[64][Columns],updated_words[64][Columns];
    __shared__ uint32_t keys[128][33],weights[64][65];
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
            if constexpr(Capture){h[index]=a;h[index+1u]=b;}
            if (!eligible(a,b)) atomicAnd(&state_ok[column],0u);
        }
        for (unsigned cell=tid;cell<64u*64u;cell+=threads) {
            const unsigned row=cell/64u,pair=cell%64u;
            uint16_t a=0u,b=0u;
            if (row<valid) {
                const size_t index=(size_t(offset+row)*32u+head)*128u+pair*2u;
                a=w[index];b=w[index+1u];
            }
            weights[row][pair]=pack(a,b);
            if (!eligible(a,b)) atomicAnd(&weight_ok[row],0u);
        }
        for (unsigned cell=tid;cell<128u*32u;cell+=threads) {
            const unsigned feature=cell/32u,pair=cell%32u;
            uint16_t a=0u,b=0u;
            if (pair*2u<valid) a=k[(size_t(offset+pair*2u)*16u+head/2u)*128u+feature];
            if (pair*2u+1u<valid) b=k[(size_t(offset+pair*2u+1u)*16u+head/2u)*128u+feature];
            keys[feature][pair]=pack(a,b);
            if (!eligible(a,b)) atomicAnd(&key_ok[feature],0u);
        }
        __syncthreads();
        for (unsigned cell=tid;cell<64u*Columns;cell+=threads) {
            const unsigned row=cell/Columns,column=cell%Columns;
            uint16_t value=0u,updated=0u;
            if (row<valid) {
                const float sum=dot<128u,Columns>(weights[row],rounded,column,weight_ok[row]&&state_ok[column]);
                const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;
                const float difference=from_bf16(u[index])-sum;
                updated=to_bf16(difference);
                if constexpr(Capture)v_new[index]=updated;
                value=to_bf16(difference*decay[row]);
            }
            residual_words[row][column]=value;updated_words[row][column]=updated;
        }
        __syncthreads();
        for (unsigned cell=tid;cell<32u*Columns;cell+=threads) {
            const unsigned pair=cell/Columns,column=cell%Columns;
            const uint16_t a=residual_words[pair*2u][column],b=residual_words[pair*2u+1u][column];
            residual[pair][column]=pack(a,b);
            if (!eligible(a,b)) atomicAnd(&residual_ok[column],0u);
        }
        __syncthreads();
        for (unsigned cell=tid;cell<128u*Columns;cell+=threads) {
            const unsigned feature=cell/Columns,column=cell%Columns;
            const float sum=dot<64u,Columns>(keys[feature],residual,column,key_ok[feature]&&residual_ok[column]);
            current[column][feature]=fmaf(current[column][feature],segment_decay,sum);
        }
        __syncthreads();
        // State update has consumed W and K. Their shared storage now holds
        // queries and scores, while rounded still contains the old state.
        if(tid<64u){weight_ok[tid]=1u;key_ok[tid]=1u;}
        if(tid<Columns)residual_ok[tid]=1u;
        __syncthreads();
        for(unsigned cell=tid;cell<64u*64u;cell+=threads){
            const unsigned row=cell/64u,pair=cell%64u;
            uint16_t a=0u,b=0u;
            if(row<valid){const size_t at=(size_t(offset+row)*16u+head/2u)*128u+pair*2u;a=q[at];b=q[at+1u];}
            weights[row][pair]=pack(a,b);
            if(!eligible(a,b))atomicAnd(&weight_ok[row],0u);
        }
        for(unsigned cell=tid;cell<64u*32u;cell+=threads){
            const unsigned row=cell/32u,pair=cell%32u;
            uint16_t a=0u,b=0u;
            if(row<valid){const size_t at=(size_t(offset+row)*32u+head)*64u+pair*2u;a=scores[at];b=scores[at+1u];}
            keys[row][pair]=pack(a,b);
            if(!eligible(a,b))atomicAnd(&key_ok[row],0u);
        }
        for(unsigned cell=tid;cell<32u*Columns;cell+=threads){
            const unsigned pair=cell/Columns,column=cell%Columns;
            const uint16_t a=updated_words[pair*2u][column],b=updated_words[pair*2u+1u][column];
            residual[pair][column]=pack(a,b);
            if(!eligible(a,b))atomicAnd(&residual_ok[column],0u);
        }
        __syncthreads();
        for(unsigned cell=tid;cell<valid*Columns;cell+=threads){
            const unsigned row=cell/Columns,column=cell%Columns;
            const float old=dot<128u,Columns>(weights[row],rounded,column,weight_ok[row]&&state_ok[column]);
            const float local=dot<64u,Columns>(keys[row],residual,column,key_ok[row]&&residual_ok[column]);
            constexpr float scale=0.08838834764831845f;
            const float prior=old*exponential(g[size_t(offset+row)*32u+head],table);
            output[(size_t(offset+row)*32u+head)*128u+first_column+column]=
                from_bf16(to_bf16(fmaf(local,scale,prior*scale)));
        }
        __syncthreads();
    }
    for (unsigned cell=tid;cell<Columns*128u;cell+=threads)
        state[(head*128u+first_column+cell/128u)*128u+cell%128u]=current[cell/128u][cell%128u];
}
} // namespace qrt_fla_fused_state_output
#endif
