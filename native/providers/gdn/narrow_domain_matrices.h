#pragma once
#include "blackwell_lifetime_matrices.h"
#include "paired_score_matrices.h"
#include "../moe_accumulator/sm121_narrow_f32_carry.h"

// Isolated complete-GDN candidate. Original row-valid flags gain a second
// bit for the proved narrow input domain. The actual BF16 operands, including
// freshly rounded recurrent state and residuals, are checked before each dot.
// Shared arena lifetimes, U=V ownership, gates, BF16 endpoints and FMAs remain.
namespace qrt_fla_narrow_domain {
namespace scalar=qrt_fla_blackwell_scalar;
namespace narrow=qrt_sm121_narrow_f32_carry;
using scalar::from_bf16;using scalar::to_bf16;using scalar::exponential;
using scalar::pack;using scalar::threads;using scalar::columns;
__device__ __forceinline__ unsigned admission(uint16_t a,uint16_t b){
    return unsigned(scalar::eligible(a,b)) |
        ((narrow::eligible(a)&&narrow::eligible(b))?2u:0u);
}
template<unsigned Width,unsigned RightColumns=columns>
__device__ __forceinline__ float dot(const uint32_t* left,
    const uint32_t (&right)[Width/2u][RightColumns],unsigned column,unsigned admitted){
    static_assert(Width==64u||Width==128u);
    if(!(admitted&2u))return scalar::dot<Width,RightColumns>(left,right,column,(admitted&1u)!=0u);
    // The K256 proof also covers these shorter zero-padded dots. Original
    // ordered groups are retained; no exceptional group can occur here.
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

// A CTA owns every row of its eight V columns and captures all V before
// any U write. The production U=V alias therefore keeps its original owner.
__global__ void wu_kernel(const uint16_t* k,const uint16_t* v,const uint16_t* beta,
    const uint16_t* inverse,const float* g,uint16_t* w,uint16_t* u,unsigned count,
    const unsigned char* table) {
    __shared__ uint32_t inv[64][33],keys[32][columns],values[32][columns];
    __shared__ unsigned inv_ok[64],key_ok[columns],value_ok[columns];
    const unsigned tid=threadIdx.x,offset=blockIdx.z*64u,head=blockIdx.y;
    const unsigned first_column=blockIdx.x*columns,valid=min(64u,count-offset);
    if (tid<64u) inv_ok[tid]=3u;
    if (tid<columns) {key_ok[tid]=3u;value_ok[tid]=3u;}
    __syncthreads();
    for (unsigned cell=tid;cell<64u*32u;cell+=threads) {
        const unsigned row=cell/32u,pair=cell%32u;
        uint16_t a=0u,b=0u;
        if (row<valid) {
            const size_t index=((size_t(offset+row)*32u+head)*64u)+pair*2u;
            if (pair*2u<valid) a=inverse[index];
            if (pair*2u+1u<valid) b=inverse[index+1u];
        }
        inv[row][pair]=pack(a,b);
        {const unsigned mask=admission(a,b);if(mask!=3u)atomicAnd(&inv_ok[row],mask);}
    }
    for (unsigned cell=tid;cell<32u*columns;cell+=threads) {
        const unsigned pair=cell/columns,column=cell%columns;
        uint16_t ks[2]{},vs[2]{};
#pragma unroll
        for (unsigned half=0u;half<2u;++half) {
            const unsigned row=pair*2u+half;
            if (row<valid) {
                const size_t token=offset+row;
                const float scale=from_bf16(beta[token*32u+head]);
                const uint16_t scaled=to_bf16(from_bf16(k[(token*16u+head/2u)*128u+first_column+column])*scale);
                ks[half]=to_bf16(from_bf16(scaled)*exponential(g[token*32u+head],table));
                vs[half]=to_bf16(from_bf16(v[(token*32u+head)*128u+first_column+column])*scale);
            }
        }
        keys[pair][column]=pack(ks[0],ks[1]);values[pair][column]=pack(vs[0],vs[1]);
        {const unsigned mask=admission(ks[0],ks[1]);if(mask!=3u)atomicAnd(&key_ok[column],mask);}
        {const unsigned mask=admission(vs[0],vs[1]);if(mask!=3u)atomicAnd(&value_ok[column],mask);}
    }
    __syncthreads();
    for (unsigned cell=tid;cell<valid*columns;cell+=threads) {
        const unsigned row=cell/columns,column=cell%columns;
        const float sw=dot<64u>(inv[row],keys,column,inv_ok[row]&key_ok[column]);
        const float su=dot<64u>(inv[row],values,column,inv_ok[row]&value_ok[column]);
        const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;
        w[index]=to_bf16(sw);u[index]=to_bf16(su);
    }
}

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
        if (tid<128u) key_ok[tid]=3u;
        if (tid<64u) weight_ok[tid]=3u;
        if (tid<Columns) {state_ok[tid]=3u;residual_ok[tid]=3u;}
        if (tid<valid) decay[tid]=exponential(g[size_t(offset+valid-1u)*32u+head]-g[size_t(offset+tid)*32u+head],table);
        if (!tid) segment_decay=exponential(g[size_t(offset+valid-1u)*32u+head],table);
        __syncthreads();
        for (unsigned cell=tid;cell<64u*Columns;cell+=threads) {
            const unsigned pair=cell/Columns,column=cell%Columns;
            const uint16_t a=to_bf16(current[column][pair*2u]),b=to_bf16(current[column][pair*2u+1u]);
            rounded[pair][column]=pack(a,b);
            const size_t index=size_t(offset/64u)*524288u+(head*128u+first_column+column)*128u+pair*2u;
            h[index]=a;h[index+1u]=b;
            {const unsigned mask=admission(a,b);if(mask!=3u)atomicAnd(&state_ok[column],mask);}
        }
        for (unsigned cell=tid;cell<64u*64u;cell+=threads) {
            const unsigned row=cell/64u,pair=cell%64u;
            uint16_t a=0u,b=0u;
            if (row<valid) {
                const size_t index=(size_t(offset+row)*32u+head)*128u+pair*2u;
                a=w[index];b=w[index+1u];
            }
            operands.weights[row][pair]=pack(a,b);
            {const unsigned mask=admission(a,b);if(mask!=3u)atomicAnd(&weight_ok[row],mask);}
        }
        __syncthreads();
        for (unsigned cell=tid;cell<64u*Columns;cell+=threads) {
            const unsigned row=cell/Columns,column=cell%Columns;
            uint16_t value=0u;
            if (row<valid) {
                const float sum=dot<128u,Columns>(operands.weights[row],rounded,column,weight_ok[row]&state_ok[column]);
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
            {const unsigned mask=admission(a,b);if(mask!=3u)atomicAnd(&key_ok[feature],mask);}
        }
        for (unsigned cell=tid;cell<32u*Columns;cell+=threads) {
            const unsigned pair=cell/Columns,column=cell%Columns;
            const uint16_t a=residual_words[pair*2u][column],b=residual_words[pair*2u+1u][column];
            residual[pair][column]=pack(a,b);
            {const unsigned mask=admission(a,b);if(mask!=3u)atomicAnd(&residual_ok[column],mask);}
        }
        __syncthreads();
        for (unsigned cell=tid;cell<128u*Columns;cell+=threads) {
            const unsigned feature=cell/Columns,column=cell%Columns;
            const float sum=dot<64u,Columns>(operands.keys[feature],residual,column,key_ok[feature]&residual_ok[column]);
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
    if(tid<64u){q_ok[tid]=3u;score_ok[tid]=3u;}
    if(tid<columns){v_ok[tid]=3u;h_ok[tid]=3u;}
    __syncthreads();
    for(unsigned cell=tid;cell<64u*64u;cell+=threads){
        const unsigned row=cell/64u,pair=cell%64u;uint16_t a=0u,b=0u;
        if(row<valid){const size_t index=(size_t(offset+row)*16u+head/2u)*128u+pair*2u;a=q[index];b=q[index+1u];}
        left.queries[row][pair]=pack(a,b);
        {const unsigned mask=admission(a,b);if(mask!=3u)atomicAnd(&q_ok[row],mask);}
    }
    for(unsigned cell=tid;cell<64u*columns;cell+=threads){
        const unsigned pair=cell/columns,column=cell%columns;
        const size_t index=size_t(blockIdx.z)*524288u+(head*128u+first_column+column)*128u+pair*2u;
        const uint16_t a=h[index],b=h[index+1u];right.checkpoint[pair][column]=pack(a,b);
        {const unsigned mask=admission(a,b);if(mask!=3u)atomicAnd(&h_ok[column],mask);}
    }
    __syncthreads();
    float old[2]{};
#pragma unroll
    for(unsigned slot=0u;slot<2u;++slot){
        const unsigned cell=tid+slot*threads,row=cell/columns,column=cell%columns;
        if(row<valid)old[slot]=dot<128u>(left.queries[row],right.checkpoint,column,q_ok[row]&h_ok[column]);
    }
    // This is a whole-CTA barrier, including threads with no valid output.
    __syncthreads();
    for(unsigned cell=tid;cell<64u*32u;cell+=threads){
        const unsigned row=cell/32u,pair=cell%32u;uint16_t a=0u,b=0u;
        if(row<valid){const size_t index=(size_t(offset+row)*32u+head)*64u+pair*2u;a=scores[index];b=scores[index+1u];}
        left.scores[row][pair]=pack(a,b);
        {const unsigned mask=admission(a,b);if(mask!=3u)atomicAnd(&score_ok[row],mask);}
    }
    for(unsigned cell=tid;cell<32u*columns;cell+=threads){
        const unsigned pair=cell/columns,column=cell%columns;uint16_t words[2]{};
#pragma unroll
        for(unsigned half=0u;half<2u;++half)
            if(pair*2u+half<valid)words[half]=v[(size_t(offset+pair*2u+half)*32u+head)*128u+first_column+column];
        right.values[pair][column]=pack(words[0],words[1]);
        {const unsigned mask=admission(words[0],words[1]);if(mask!=3u)atomicAnd(&v_ok[column],mask);}
    }
    __syncthreads();
#pragma unroll
    for(unsigned slot=0u;slot<2u;++slot){
        const unsigned cell=tid+slot*threads,row=cell/columns,column=cell%columns;
        if(row<valid){
            const float local=dot<64u>(left.scores[row],right.values,column,score_ok[row]&v_ok[column]);
            constexpr float scale=0.08838834764831845f;
            const float prior=old[slot]*exponential(g[size_t(offset+row)*32u+head],table);
            output[(size_t(offset+row)*32u+head)*128u+first_column+column]=from_bf16(to_bf16(fmaf(local,scale,prior*scale)));
        }
    }
}

constexpr unsigned score_rows=8u,score_columns=32u,pairs=64u;
__global__ void score_kernel(const uint16_t* q,const uint16_t* k,const float* g,
    uint16_t* scores,unsigned count,const unsigned char* table) {
    __shared__ uint32_t queries[score_rows][65],keys[pairs][score_columns];
    __shared__ unsigned admitted;
    const unsigned chunk=blockIdx.z/(64u/score_rows),first_row=(blockIdx.z%(64u/score_rows))*score_rows;
    const unsigned offset=chunk*64u,valid=min(64u,count-offset);
    const unsigned first_column=blockIdx.x*score_columns,qk_head=blockIdx.y;
    const unsigned row=first_row+threadIdx.x/score_columns,column=first_column+threadIdx.x%score_columns;
    if(first_column>first_row+score_rows-1u||first_row>=valid){
        if(row>=valid)return;
#pragma unroll
        for(unsigned half=0u;half<2u;++half)
            scores[(size_t(offset+row)*32u+qk_head*2u+half)*64u+column]=0u;
        return;
    }
    if(!threadIdx.x)admitted=3u;
    __syncthreads();
    unsigned mask=3u;
    for(unsigned cell=threadIdx.x;cell<score_rows*pairs;cell+=threads){
        const unsigned r=cell/pairs,pair=cell%pairs;uint16_t a=0u,b=0u;
        if(first_row+r<valid){
            const size_t index=(size_t(offset+first_row+r)*16u+qk_head)*128u+pair*2u;
            a=q[index];b=q[index+1u];
        }
        queries[r][pair]=scalar::pack(a,b);mask &= admission(a,b);
    }
    for(unsigned cell=threadIdx.x;cell<pairs*score_columns;cell+=threads){
        const unsigned pair=cell/score_columns,c=cell%score_columns;uint16_t a=0u,b=0u;
        if(first_column+c<valid){
            const size_t index=(size_t(offset+first_column+c)*16u+qk_head)*128u+pair*2u;
            a=k[index];b=k[index+1u];
        }
        keys[pair][c]=scalar::pack(a,b);mask &= admission(a,b);
    }
    if(mask!=3u)atomicAnd(&admitted,mask);
    __syncthreads();
    if(row>=valid)return;
    if(column>row){
#pragma unroll
        for(unsigned half=0u;half<2u;++half)
            scores[(size_t(offset+row)*32u+qk_head*2u+half)*64u+column]=0u;
        return;
    }
    const float sum=dot<128u,score_columns>(queries[threadIdx.x/score_columns],keys,
        threadIdx.x%score_columns,admitted);
#pragma unroll
    for(unsigned half=0u;half<2u;++half){
        const unsigned head=qk_head*2u+half;
        const float difference=g[size_t(offset+row)*32u+head]-g[size_t(offset+column)*32u+head];
        scores[(size_t(offset+row)*32u+head)*64u+column]=scalar::to_bf16(sum*scalar::exponential(difference,table));
    }
}
} // namespace qrt_fla_narrow_domain
