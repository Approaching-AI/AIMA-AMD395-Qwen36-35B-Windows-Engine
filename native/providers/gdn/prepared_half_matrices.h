#ifndef QRT_FLA_PREPARED_HALF_MATRICES_H
#define QRT_FLA_PREPARED_HALF_MATRICES_H
#include "blackwell_scalar_state.h"
#include "../moe_accumulator/sm121_staged_half_projection.h"

// Isolated shared-operand replacement for all three scalar GDN surfaces.
// Original BF16 values, K16 order, checkpoint layout and final FMAs remain.
// Unsupported scaled-half rows retain their original words and exact fallback.
namespace qrt_fla_prepared_half {
namespace scalar=qrt_fla_blackwell_scalar;
namespace half=qrt_sm121_scaled_half_products;
using Row=half::Row;
using scalar::from_bf16;using scalar::to_bf16;using scalar::exponential;
constexpr unsigned threads=256u,columns=8u;
template<unsigned Width,unsigned Lanes>
__device__ __forceinline__ float dot(const Row* left,const Row* right) {
    static_assert(Lanes==1u || Lanes==4u);
    if constexpr(Lanes==4u)return qrt_sm121_staged_half_projection::dot<2u>(left,right,Width);
    else {
        qrt_q1_moe_hawkeye::Value carry{0u,-133,false};
        for(unsigned group=0u;group<Width/16u;++group)carry=half::accumulate<true>(carry,left[group],right[group]);
        return qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
    }
}

// A CTA captures all V values in its eight owned columns before any U write.
// This preserves the production U=V alias. Per-token decay/beta are reused by
// every column without changing their evaluation or rounding.
template<unsigned Lanes>
__global__ void wu_kernel(const uint16_t* k,const uint16_t* v,const uint16_t* beta,
    const uint16_t* inverse,const float* g,uint16_t* w,uint16_t* u,unsigned count,
    const unsigned char* table) {
    __shared__ Row inv[64][4],keys[columns][4],values[columns][4];
    __shared__ float decay[64],scale[64];
    const unsigned tid=threadIdx.x,offset=blockIdx.z*64u,head=blockIdx.y;
    const unsigned first_column=blockIdx.x*columns,valid=min(64u,count-offset);
    if(tid<valid){decay[tid]=exponential(g[size_t(offset+tid)*32u+head],table);scale[tid]=from_bf16(beta[size_t(offset+tid)*32u+head]);}
    for(unsigned item=tid;item<64u*4u;item+=threads) {
        const unsigned row=item/4u,group=item%4u;uint16_t words[16]{};
        for(unsigned i=0u;i<16u;++i)if(row<valid && group*16u+i<valid)
            words[i]=inverse[(size_t(offset+row)*32u+head)*64u+group*16u+i];
        inv[row][group]=half::prepare(words);
    }
    __syncthreads();
    for(unsigned item=tid;item<columns*4u;item+=threads) {
        const unsigned column=item/4u,group=item%4u;uint16_t ks[16]{},vs[16]{};
        for(unsigned i=0u;i<16u;++i) {
            const unsigned row=group*16u+i;
            if(row<valid) {
                const size_t token=offset+row;
                const uint16_t scaled=to_bf16(from_bf16(k[(token*16u+head/2u)*128u+first_column+column])*scale[row]);
                ks[i]=to_bf16(from_bf16(scaled)*decay[row]);
                vs[i]=to_bf16(from_bf16(v[(token*32u+head)*128u+first_column+column])*scale[row]);
            }
        }
        keys[column][group]=half::prepare(ks);values[column][group]=half::prepare(vs);
    }
    __syncthreads();
    for(unsigned cell=tid/Lanes;cell<valid*columns;cell+=threads/Lanes) {
        const unsigned row=cell/columns,column=cell%columns;
        const float sw=dot<64u,Lanes>(inv[row],keys[column]),su=dot<64u,Lanes>(inv[row],values[column]);
        if(tid%Lanes==0u){const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;w[index]=to_bf16(sw);u[index]=to_bf16(su);}
    }
}

template<unsigned Lanes>
__global__ void output_kernel(const uint16_t* q,const uint16_t* v,const uint16_t* h,
    const float* g,const uint16_t* scores,float* output,unsigned count,
    const unsigned char* table) {
    __shared__ Row queries[64][8],score_rows[64][4],values[columns][4],checkpoint[columns][8];
    __shared__ float decay[64];
    const unsigned tid=threadIdx.x,offset=blockIdx.z*64u,head=blockIdx.y;
    const unsigned first_column=blockIdx.x*columns,valid=min(64u,count-offset);
    if(tid<valid)decay[tid]=exponential(g[size_t(offset+tid)*32u+head],table);
    for(unsigned item=tid;item<64u*8u;item+=threads) {
        const unsigned row=item/8u,group=item%8u;uint16_t words[16]{};
        for(unsigned i=0u;i<16u;++i)if(row<valid)words[i]=q[(size_t(offset+row)*16u+head/2u)*128u+group*16u+i];
        queries[row][group]=half::prepare(words);
    }
    for(unsigned item=tid;item<64u*4u;item+=threads) {
        const unsigned row=item/4u,group=item%4u;uint16_t words[16]{};
        for(unsigned i=0u;i<16u;++i)if(row<valid)words[i]=scores[(size_t(offset+row)*32u+head)*64u+group*16u+i];
        score_rows[row][group]=half::prepare(words);
    }
    for(unsigned item=tid;item<columns*8u;item+=threads) {
        const unsigned column=item/8u,group=item%8u;uint16_t words[16];
        for(unsigned i=0u;i<16u;++i)words[i]=h[size_t(blockIdx.z)*524288u+(head*128u+first_column+column)*128u+group*16u+i];
        checkpoint[column][group]=half::prepare(words);
    }
    for(unsigned item=tid;item<columns*4u;item+=threads) {
        const unsigned column=item/4u,group=item%4u;uint16_t words[16]{};
        for(unsigned i=0u;i<16u;++i)if(group*16u+i<valid)words[i]=v[(size_t(offset+group*16u+i)*32u+head)*128u+first_column+column];
        values[column][group]=half::prepare(words);
    }
    __syncthreads();
    for(unsigned cell=tid/Lanes;cell<valid*columns;cell+=threads/Lanes) {
        const unsigned row=cell/columns,column=cell%columns;
        const float old=dot<128u,Lanes>(queries[row],checkpoint[column]),local=dot<64u,Lanes>(score_rows[row],values[column]);
        if(tid%Lanes==0u) {
            constexpr float scale=0.08838834764831845f;
            const float prior=old*decay[row];
            output[(size_t(offset+row)*32u+head)*128u+first_column+column]=from_bf16(to_bf16(fmaf(local,scale,prior*scale)));
        }
    }
}

template<unsigned Lanes>
__global__ void state_kernel(const uint16_t* k,const uint16_t* u,const uint16_t* w,
    const float* g,uint16_t* h,uint16_t* v_new,float* state,unsigned count,
    const unsigned char* table) {
    __shared__ float current[columns][128],decay[64],segment_decay;
    __shared__ Row rounded[columns][8],residual[columns][4],keys[128][4],weights[64][8];
    __shared__ uint16_t residual_words[64][columns];
    const unsigned tid=threadIdx.x,head=blockIdx.y,first_column=blockIdx.x*columns;
    for(unsigned cell=tid;cell<columns*128u;cell+=threads)current[cell/128u][cell%128u]=state[(head*128u+first_column+cell/128u)*128u+cell%128u];
    __syncthreads();
    for(unsigned offset=0u;offset<count;offset+=64u) {
        const unsigned valid=min(64u,count-offset);
        if(tid<valid)decay[tid]=exponential(g[size_t(offset+valid-1u)*32u+head]-g[size_t(offset+tid)*32u+head],table);
        if(!tid)segment_decay=exponential(g[size_t(offset+valid-1u)*32u+head],table);
        for(unsigned item=tid;item<columns*8u;item+=threads) {
            const unsigned column=item/8u,group=item%8u;uint16_t words[16];
            for(unsigned i=0u;i<16u;++i){words[i]=to_bf16(current[column][group*16u+i]);h[size_t(offset/64u)*524288u+(head*128u+first_column+column)*128u+group*16u+i]=words[i];}
            rounded[column][group]=half::prepare(words);
        }
        for(unsigned item=tid;item<64u*8u;item+=threads) {
            const unsigned row=item/8u,group=item%8u;uint16_t words[16]{};
            for(unsigned i=0u;i<16u;++i)if(row<valid)words[i]=w[(size_t(offset+row)*32u+head)*128u+group*16u+i];
            weights[row][group]=half::prepare(words);
        }
        for(unsigned item=tid;item<128u*4u;item+=threads) {
            const unsigned feature=item/4u,group=item%4u;uint16_t words[16]{};
            for(unsigned i=0u;i<16u;++i)if(group*16u+i<valid)words[i]=k[(size_t(offset+group*16u+i)*16u+head/2u)*128u+feature];
            keys[feature][group]=half::prepare(words);
        }
        __syncthreads();
        for(unsigned cell=tid/Lanes;cell<64u*columns;cell+=threads/Lanes) {
            const unsigned row=cell/columns,column=cell%columns;uint16_t value=0u;
            if(row<valid) {
                const float sum=dot<128u,Lanes>(weights[row],rounded[column]);
                if(tid%Lanes==0u) {
                    const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;
                    const float difference=from_bf16(u[index])-sum;v_new[index]=to_bf16(difference);value=to_bf16(difference*decay[row]);
                }
            }
            if(tid%Lanes==0u)residual_words[row][column]=value;
        }
        __syncthreads();
        for(unsigned item=tid;item<columns*4u;item+=threads) {
            const unsigned column=item/4u,group=item%4u;uint16_t words[16];
            for(unsigned i=0u;i<16u;++i)words[i]=residual_words[group*16u+i][column];
            residual[column][group]=half::prepare(words);
        }
        __syncthreads();
        for(unsigned cell=tid/Lanes;cell<128u*columns;cell+=threads/Lanes) {
            const unsigned feature=cell/columns,column=cell%columns;
            const float sum=dot<64u,Lanes>(keys[feature],residual[column]);
            if(tid%Lanes==0u)current[column][feature]=fmaf(current[column][feature],segment_decay,sum);
        }
        __syncthreads();
    }
    for(unsigned cell=tid;cell<columns*128u;cell+=threads)state[(head*128u+first_column+cell/128u)*128u+cell%128u]=current[cell/128u][cell%128u];
}
}
#endif
