#pragma once
#include "sm121_byte_exponents.h"
#include "sm121_staged_half_projection.h"
#include "sm121_tiled_projection.h"

// Isolated candidate replay. A sparse square tile reuses original BF16
// operands and exact byte metadata. Dense tiles and unsupported operand rows
// keep the qualified four-lane staged-half dot, with identical candidate IDs.
namespace qrt_sm121_sparse_byte_projection {
namespace byte=qrt_sm121_byte_exponents;
namespace staged=qrt_sm121_staged_half_projection;
using Metadata=byte::Metadata;
constexpr unsigned threads=256u,capacity=256u,window=64u;
inline bool supported_width(unsigned width){return width&&width<=8192u&&width%16u==0u;}

// Extending the original K256 proof to K8192 changes only the upper bound:
// absolute products sum to less than 8192*2^66=2^79. Canonical signed
// alignment and normalization do not increase absolute growth. The smallest
// nonzero aligned quantum remains 2^-89. Thus every nonzero carry is normal,
// every alignment scale is finite, and the original integer conversions fit.
// Complete rows must still pass exponent95:159 and each K16 span<=31.
__global__ void prepare(const uint16_t* values,Metadata* metadata,unsigned* flags,
    unsigned rows,unsigned width){
    const unsigned row=blockIdx.x,lane=threadIdx.x,groups=width/16u;
    unsigned valid=1u;
    for(unsigned group=lane;group<groups;group+=32u){
        const auto m=byte::prepare(values+size_t(row)*width+group*16u);
        metadata[size_t(row)*groups+group]=m;valid&=unsigned(m.maximum>=0);
    }
    const unsigned all=__ballot(valid!=0u);
    if(!lane&&row<rows)flags[row]=all==0xffffffffu;
}

template<unsigned Tile>
__global__ __launch_bounds__(threads) void replay(const uint16_t* weights,const uint16_t* inputs,
    const staged::Row* prepared_weights,const staged::Row* prepared_inputs,
    const Metadata* weight_metadata,const Metadata* input_metadata,
    const unsigned* weight_flags,const unsigned* input_flags,const unsigned* mask,
    float* output,unsigned* statistics,unsigned rows,unsigned tokens,unsigned width){
    static_assert(Tile==32u||Tile==64u);
    constexpr unsigned groups=window/16u,sides=2u*Tile;
    __shared__ unsigned count;
    __shared__ uint16_t candidates[capacity];
    __shared__ uint32_t values[sides][window/2u+1u];
    __shared__ uint32_t meta[groups][5u][sides];
    const unsigned tid=threadIdx.x,first_row=blockIdx.x*Tile,first_token=blockIdx.y*Tile;
    const unsigned total_groups=width/16u;
    if(!tid)count=0u;
    __syncthreads();
    for(unsigned i=tid;i<Tile*Tile;i+=threads){
        const unsigned row=first_row+i%Tile,token=first_token+i/Tile;
        if(qrt_sm121_tiled_projection::selected(mask,rows,tokens,row,token)){
            const unsigned slot=atomicAdd(&count,1u);
            if(slot<capacity)candidates[slot]=uint16_t(i);
        }
    }
    __syncthreads();
    if(!count)return;
    if(count>capacity){
        if(!tid){atomicAdd(statistics+1u,count);atomicAdd(statistics+3u,1u);}
        for(unsigned i=tid/4u;i<Tile*Tile;i+=threads/4u){
            const unsigned row=first_row+i%Tile,token=first_token+i/Tile;
            if(!qrt_sm121_tiled_projection::selected(mask,rows,tokens,row,token))continue;
            const float result=staged::dot<2u>(prepared_inputs+size_t(token)*total_groups,
                prepared_weights+size_t(row)*total_groups,width);
            if(!(tid&3u))output[size_t(token)*rows+row]=result;
        }
        return;
    }
    if(!tid)atomicAdd(statistics+2u,1u);
    const bool active=tid<count;
    const unsigned local=active?candidates[tid]:0u,wr=local%Tile,ir=local/Tile;
    const bool fast=active&&weight_flags[first_row+wr]&&input_flags[first_token+ir];
    const unsigned fast_mask=__ballot(fast),slow_mask=__ballot(active&&!fast);
    if(!(tid&31u)){atomicAdd(statistics,__popc(fast_mask));atomicAdd(statistics+1u,__popc(slow_mask));}
    float carry=0.0f;
    for(unsigned base=0u;base<width;base+=window){
        for(unsigned i=tid;i<sides*(window/2u);i+=threads){
            const unsigned side=i/(window/2u),pair=i%(window/2u),row=side<Tile?first_row+side:first_token+side-Tile;
            uint32_t word=0u;
            if(row<(side<Tile?rows:tokens)&&base+pair*2u<width)
                __builtin_memcpy(&word,(side<Tile?weights:inputs)+size_t(row)*width+base+pair*2u,4u);
            values[side][pair]=word;
        }
        for(unsigned i=tid;i<groups*5u*sides;i+=threads){
            const unsigned side=i%sides,word=(i/sides)%5u,group=i/(sides*5u);
            const unsigned row=side<Tile?first_row+side:first_token+side-Tile;
            uint32_t value=0u;
            if(row<(side<Tile?rows:tokens)&&base/16u+group<total_groups){
                const auto* m=(side<Tile?weight_metadata:input_metadata)+size_t(row)*total_groups+base/16u+group;
                __builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(m)+word*4u,4u);
            }
            meta[group][word][side]=value;
        }
        __syncthreads();
        if(fast){
#pragma unroll
            for(unsigned group=0u;group<groups;++group)if(base+group*16u<width){
                Metadata a,b;
#pragma unroll
                for(unsigned i=0u;i<4u;++i){a.deficits[i]=meta[group][i][Tile+ir];b.deficits[i]=meta[group][i][wr];}
                a.maximum=int(meta[group][4u][Tile+ir]);b.maximum=int(meta[group][4u][wr]);
                const int maximum=byte::alignment(carry,byte::product_maximum(a,b));
                const float scale=byte::f32::alignment::from_bits(uint32_t(152-maximum)<<23u);
                uint32_t modulo=uint32_t(int32_t(carry*scale));bool negative=false;
#pragma unroll
                for(unsigned pair=0u;pair<8u;++pair){
                    const uint32_t x=values[Tile+ir][group*8u+pair],w=values[wr][group*8u+pair];
                    const float lo=byte::f32::alignment::from_bits(x<<16u)*byte::f32::alignment::from_bits(w<<16u);
                    const float hi=byte::f32::alignment::from_bits(x&0xffff0000u)*byte::f32::alignment::from_bits(w&0xffff0000u);
                    modulo+=uint32_t(int32_t(lo*scale));modulo+=uint32_t(int32_t(hi*scale));
                    if(!pair)negative=((x^w)&0x8000u)!=0u;
                }
                carry=byte::finish(modulo,negative,maximum);
            }
        }
        __syncthreads();
    }
    if(fast)output[size_t(first_token+ir)*rows+first_row+wr]=carry;
    // Whole-row admission is checked before fast arithmetic. Each rejected
    // candidate keeps the original complete recurrence, including tiny rows.
    for(unsigned slot=tid/4u;slot<count;slot+=threads/4u){
        const unsigned cell=candidates[slot],row=first_row+cell%Tile,token=first_token+cell/Tile;
        if(weight_flags[row]&&input_flags[token])continue;
        const float result=staged::dot<2u>(prepared_inputs+size_t(token)*total_groups,
            prepared_weights+size_t(row)*total_groups,width);
        if(!(tid&3u))output[size_t(token)*rows+row]=result;
    }
}

inline hipError_t launch(const uint16_t* weights,const uint16_t* inputs,
    const staged::Row* pw,size_t weight_records,const staged::Row* pi,size_t input_records,
    const Metadata* wm,const Metadata* im,const unsigned* wf,const unsigned* inf,
    const unsigned* mask,size_t mask_words,float* output,size_t output_cells,unsigned* statistics,
    unsigned rows,unsigned tokens,unsigned width,unsigned tile,hipStream_t stream){
    const size_t cells=size_t(rows)*tokens;
    if(!weights||!inputs||!pw||!pi||!wm||!im||!wf||!inf||!mask||!output||!statistics||
        !rows||rows>16384u||!tokens||tokens>8192u||!supported_width(width)||
        weight_records<size_t(rows)*(width/16u)||input_records<size_t(tokens)*(width/16u)||
        mask_words<(cells+31u)/32u||output_cells<cells||(tile!=32u&&tile!=64u))return hipErrorInvalidValue;
    const dim3 grid((rows+tile-1u)/tile,(tokens+tile-1u)/tile);
    if(tile==32u)hipLaunchKernelGGL((replay<32u>),grid,dim3(threads),0u,stream,
        weights,inputs,pw,pi,wm,im,wf,inf,mask,output,statistics,rows,tokens,width);
    else hipLaunchKernelGGL((replay<64u>),grid,dim3(threads),0u,stream,
        weights,inputs,pw,pi,wm,im,wf,inf,mask,output,statistics,rows,tokens,width);
    return hipGetLastError();
}
} // namespace qrt_sm121_sparse_byte_projection
