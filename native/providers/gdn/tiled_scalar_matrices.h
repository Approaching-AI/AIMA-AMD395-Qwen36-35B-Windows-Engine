#pragma once
#include "blackwell_scalar_state.h"
#include "../moe_accumulator/sm121_packed_dot_tile.h"

// Isolated complete GDN dataflow replacement. Shared operands feed two/four
// independent ordered dots per lane. Every original intermediate, U=V owner,
// state checkpoint, BF16 conversion and final FMA remains in the same order.
// No production dispatcher includes this header.
namespace qrt_fla_tiled_scalar {
namespace scalar=qrt_fla_blackwell_scalar;
namespace tile=qrt_sm121_packed_dot_tile;
using scalar::from_bf16;using scalar::to_bf16;using scalar::exponential;
using scalar::eligible;using scalar::pack;
constexpr unsigned columns=scalar::columns;
// A CTA owns every row of its eight V columns and captures all V before
// any U write. The production U=V alias therefore keeps its original owner.
template<unsigned TileRows,unsigned TileColumns,unsigned Threads>
__global__ void wu_kernel(const uint16_t* k,const uint16_t* v,const uint16_t* beta,
    const uint16_t* inverse,const float* g,uint16_t* w,uint16_t* u,unsigned count,
    const unsigned char* table) {
    static_assert((TileRows==1u || TileRows==2u) && TileColumns==2u);
    static_assert(Threads==128u || Threads==256u);
    constexpr unsigned threads=Threads;
    __shared__ uint32_t inv[64][33],keys[32][columns],values[32][columns];
    __shared__ unsigned inv_ok[64],key_ok[columns],value_ok[columns];
    const unsigned tid=threadIdx.x,offset=blockIdx.z*64u,head=blockIdx.y;
    const unsigned first_column=blockIdx.x*columns,valid=min(64u,count-offset);
    if (tid<64u) inv_ok[tid]=1u;
    if (tid<columns) {key_ok[tid]=1u;value_ok[tid]=1u;}
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
        if (!eligible(a,b)) atomicAnd(&inv_ok[row],0u);
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
        if (!eligible(ks[0],ks[1])) atomicAnd(&key_ok[column],0u);
        if (!eligible(vs[0],vs[1])) atomicAnd(&value_ok[column],0u);
    }
    __syncthreads();
    for (unsigned cell=tid;cell<((valid+TileRows-1u)/TileRows)*(columns/TileColumns);cell+=threads) {
        const unsigned row=cell/(columns/TileColumns)*TileRows,column=cell%(columns/TileColumns)*TileColumns;
        const auto sw=tile::dot<64u,33u,columns,TileRows,TileColumns>(&inv[row][0],&keys[0][column],inv_ok+row,key_ok+column);
        const auto su=tile::dot<64u,33u,columns,TileRows,TileColumns>(&inv[row][0],&values[0][column],inv_ok+row,value_ok+column);
#pragma unroll
        for(unsigned r=0u;r<TileRows;++r)if(row+r<valid){
#pragma unroll
            for(unsigned c=0u;c<TileColumns;++c){
                const size_t index=(size_t(offset+row+r)*32u+head)*128u+first_column+column+c;
                w[index]=to_bf16(sw.values[r][c]);u[index]=to_bf16(su.values[r][c]);
            }
        }
    }
}

// Packed feature-major right operands avoid repeated row-stride LDS bank
// collisions. Padding the left row pitch separates adjacent query banks.
template<unsigned TileRows,unsigned TileColumns,unsigned Threads>
__global__ void output_kernel(const uint16_t* q,const uint16_t* v,const uint16_t* h,
    const float* g,const uint16_t* scores,float* output,unsigned count,
    const unsigned char* table) {
    static_assert((TileRows==1u || TileRows==2u) && TileColumns==2u);
    static_assert(Threads==128u || Threads==256u);
    constexpr unsigned threads=Threads;
    __shared__ uint32_t queries[64][65],score_rows[64][33];
    __shared__ uint32_t values[32][columns],checkpoint[64][columns];
    __shared__ unsigned q_ok[64],score_ok[64],v_ok[columns],h_ok[columns];
    const unsigned tid=threadIdx.x,offset=blockIdx.z*64u,head=blockIdx.y;
    const unsigned first_column=blockIdx.x*columns,valid=min(64u,count-offset);
    if (tid<64u) {q_ok[tid]=1u;score_ok[tid]=1u;}
    if (tid<columns) {v_ok[tid]=1u;h_ok[tid]=1u;}
    __syncthreads();
    for (unsigned cell=tid;cell<64u*64u;cell+=threads) {
        const unsigned row=cell/64u,pair=cell%64u;
        uint16_t a=0u,b=0u;
        if (row<valid) {
            const size_t index=(size_t(offset+row)*16u+head/2u)*128u+pair*2u;
            a=q[index];b=q[index+1u];
        }
        queries[row][pair]=pack(a,b);
        if (!eligible(a,b)) atomicAnd(&q_ok[row],0u);
    }
    for (unsigned cell=tid;cell<64u*32u;cell+=threads) {
        const unsigned row=cell/32u,pair=cell%32u;
        uint16_t a=0u,b=0u;
        if (row<valid) {
            const size_t index=(size_t(offset+row)*32u+head)*64u+pair*2u;
            a=scores[index];b=scores[index+1u];
        }
        score_rows[row][pair]=pack(a,b);
        if (!eligible(a,b)) atomicAnd(&score_ok[row],0u);
    }
    for (unsigned cell=tid;cell<64u*columns;cell+=threads) {
        const unsigned pair=cell/columns,column=cell%columns;
        const size_t index=size_t(blockIdx.z)*524288u+(head*128u+first_column+column)*128u+pair*2u;
        const uint16_t a=h[index],b=h[index+1u];
        checkpoint[pair][column]=pack(a,b);
        if (!eligible(a,b)) atomicAnd(&h_ok[column],0u);
    }
    for (unsigned cell=tid;cell<32u*columns;cell+=threads) {
        const unsigned pair=cell/columns,column=cell%columns;
        uint16_t words[2]{};
#pragma unroll
        for (unsigned half=0u;half<2u;++half)
            if (pair*2u+half<valid) words[half]=v[(size_t(offset+pair*2u+half)*32u+head)*128u+first_column+column];
        values[pair][column]=pack(words[0],words[1]);
        if (!eligible(words[0],words[1])) atomicAnd(&v_ok[column],0u);
    }
    __syncthreads();
    for (unsigned cell=tid;cell<((valid+TileRows-1u)/TileRows)*(columns/TileColumns);cell+=threads) {
        const unsigned row=cell/(columns/TileColumns)*TileRows,column=cell%(columns/TileColumns)*TileColumns;
        const auto old=tile::dot<128u,65u,columns,TileRows,TileColumns>(&queries[row][0],&checkpoint[0][column],q_ok+row,h_ok+column);
        const auto local=tile::dot<64u,33u,columns,TileRows,TileColumns>(&score_rows[row][0],&values[0][column],score_ok+row,v_ok+column);
        constexpr float scale=0.08838834764831845f;
#pragma unroll
        for(unsigned r=0u;r<TileRows;++r)if(row+r<valid){
            const float gate=exponential(g[size_t(offset+row+r)*32u+head],table);
#pragma unroll
            for(unsigned c=0u;c<TileColumns;++c){
                const float prior=old.values[r][c]*gate;
                output[(size_t(offset+row+r)*32u+head)*128u+first_column+column+c]=
                    from_bf16(to_bf16(fmaf(local.values[r][c],scale,prior*scale)));
            }
        }
    }
}
// Retain complete state columns throughout one bounded segment. Calls that
// capture prefix checkpoints continue to use their original implementation.
template<unsigned Columns,unsigned TileRows,unsigned TileColumns,unsigned Threads>
__global__ void state_kernel(const uint16_t* k,const uint16_t* u,const uint16_t* w,
    const float* g,uint16_t* h,uint16_t* v_new,float* state,unsigned count,
    const unsigned char* table) {
    static_assert(Columns==4u || Columns==8u);
    static_assert((TileRows==1u || TileRows==2u) && TileColumns==2u);
    static_assert(Threads==128u || Threads==256u);
    constexpr unsigned threads=Threads;
    __shared__ float current[Columns][128],decay[64],segment_decay;
    __shared__ uint32_t rounded[64][Columns],residual[32][Columns];
    __shared__ uint16_t residual_words[64][Columns];
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
        for (unsigned cell=tid;cell<(64u/TileRows)*(Columns/TileColumns);cell+=threads) {
            const unsigned row=cell/(Columns/TileColumns)*TileRows,column=cell%(Columns/TileColumns)*TileColumns;
            const auto sums=tile::dot<128u,65u,Columns,TileRows,TileColumns>(&weights[row][0],&rounded[0][column],weight_ok+row,state_ok+column);
#pragma unroll
            for(unsigned r=0u;r<TileRows;++r){
#pragma unroll
                for(unsigned c=0u;c<TileColumns;++c){
                    uint16_t value=0u;
                    if(row+r<valid){
                        const size_t index=(size_t(offset+row+r)*32u+head)*128u+first_column+column+c;
                        const float difference=from_bf16(u[index])-sums.values[r][c];
                        v_new[index]=to_bf16(difference);
                        value=to_bf16(difference*decay[row+r]);
                    }
                    residual_words[row+r][column+c]=value;
                }
            }
        }
        __syncthreads();
        for (unsigned cell=tid;cell<32u*Columns;cell+=threads) {
            const unsigned pair=cell/Columns,column=cell%Columns;
            const uint16_t a=residual_words[pair*2u][column],b=residual_words[pair*2u+1u][column];
            residual[pair][column]=pack(a,b);
            if (!eligible(a,b)) atomicAnd(&residual_ok[column],0u);
        }
        __syncthreads();
        for (unsigned cell=tid;cell<(128u/TileRows)*(Columns/TileColumns);cell+=threads) {
            const unsigned feature=cell/(Columns/TileColumns)*TileRows,column=cell%(Columns/TileColumns)*TileColumns;
            const auto sums=tile::dot<64u,33u,Columns,TileRows,TileColumns>(&keys[feature][0],&residual[0][column],key_ok+feature,residual_ok+column);
#pragma unroll
            for(unsigned r=0u;r<TileRows;++r){
#pragma unroll
                for(unsigned c=0u;c<TileColumns;++c)
                    current[column+c][feature+r]=fmaf(current[column+c][feature+r],segment_decay,sums.values[r][c]);
            }
        }
        __syncthreads();
    }
    for (unsigned cell=tid;cell<Columns*128u;cell+=threads)
        state[(head*128u+first_column+cell/128u)*128u+cell%128u]=current[cell/128u][cell%128u];
}
}} // namespace qrt_fla_tiled_scalar
