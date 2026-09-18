#pragma once
#include "narrow_domain_qk_workspace.h"
#include "../moe_accumulator/sm121_byte_exponents.h"

namespace qrt_byte_exponent_qk {
namespace core=qrt_sm121_byte_exponents;
struct Workspace {
    qrt_narrow_domain_qk::Workspace original;
    const core::Metadata *query_metadata,*key_metadata;
    const unsigned *query_flags,*key_flags;
};
template<bool Key>
__global__ void prepare(const uint16_t* input,core::Metadata* output,
    unsigned* flags,unsigned tokens) {
    constexpr unsigned heads=Key?2u:16u;
    const unsigned row=blockIdx.x,group=threadIdx.x;
    __shared__ unsigned rejected;
    if(!group)rejected=0u;
    __syncthreads();
    if(group<16u){
        uint16_t values[16];
#pragma unroll
        for(unsigned i=0u;i<16u;++i)values[i]=input[size_t(row)*256u+group*16u+i];
        const auto metadata=core::prepare(values);
        output[Key?(size_t(row%heads)*16u+group)*tokens+row/heads:size_t(row)*16u+group]=metadata;
        if(metadata.maximum<0)atomicOr(&rejected,1u);
    }
    __syncthreads();
    if(!group)flags[row]=!rejected;
}

// All accepted cells use the same proved narrow K256 arithmetic. Packed
// exponent minima establish each K16 scale before any product is formed.
template<unsigned QueryCells,unsigned KeyCells=4u,unsigned Window=64u>
__global__ void scores(const uint16_t* query,const uint16_t* transposed_key,
    Workspace w,float* output,unsigned start,unsigned count,unsigned stride) {
    static_assert(QueryCells==2u||QueryCells==4u);
    static_assert(KeyCells==4u&&Window==64u);
    constexpr unsigned rows=QueryCells*16u,columns=KeyCells*16u,groups=Window/16u;
    __shared__ uint32_t qvalues[rows][Window/2u+1u],kvalues[Window/2u][columns];
    __shared__ uint32_t qm[groups][5u][rows],km[groups][5u][columns];
    const unsigned head=blockIdx.y,kv=head/8u,qr=threadIdx.x/16u,kc=threadIdx.x%16u;
    const unsigned query_tile=blockIdx.z*rows,key_tile=blockIdx.x*columns,tokens=w.original.tokens;
    const bool interior=query_tile+rows<=count&&key_tile+columns<=stride&&
        key_tile+columns-1u<=start+query_tile;
    if(!interior)return;
    auto* rejected=&qvalues[0][0];
    if(!threadIdx.x)*rejected=0u;
    __syncthreads();
    if(threadIdx.x<rows&&!w.query_flags[(start+query_tile+threadIdx.x)*16u+head])atomicOr(rejected,1u);
    if(threadIdx.x<columns&&!w.key_flags[(key_tile+threadIdx.x)*2u+kv])atomicOr(rejected,1u);
    __syncthreads();
    const bool accepted=!*rejected;
    __syncthreads();
    if(!accepted)return;
    if(!threadIdx.x)atomicAdd(w.original.tile_counts+1u,1u);
    float carry[QueryCells][KeyCells]{};
    for(unsigned window=0u;window<256u;window+=Window){
        for(unsigned i=threadIdx.x;i<rows*(Window/2u);i+=256u){
            const unsigned r=i/(Window/2u),p=i%(Window/2u);
            uint32_t words;__builtin_memcpy(&words,query+(size_t(start+query_tile+r)*16u+head)*256u+window+p*2u,4u);
            qvalues[r][p]=words;
        }
        for(unsigned i=threadIdx.x;i<(Window/2u)*columns;i+=256u){
            const unsigned p=i/columns,c=i%columns;
            const auto* source=transposed_key+(size_t(kv)*256u+window+p*2u)*tokens+key_tile+c;
            kvalues[p][c]=uint32_t(source[0])|(uint32_t(source[tokens])<<16u);
        }
        for(unsigned i=threadIdx.x;i<groups*(rows+columns)*5u;i+=256u){
            const unsigned word=i%5u,position=(i/5u)%(rows+columns),group=i/(5u*(rows+columns));
            const bool is_query=position<rows;
            const unsigned local=is_query?position:position-rows;
            const auto* source=is_query
                ?w.query_metadata+(size_t(start+query_tile+local)*16u+head)*16u+window/16u+group
                :w.key_metadata+(size_t(kv)*16u+window/16u+group)*tokens+key_tile+local;
            uint32_t value;__builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(source)+word*4u,4u);
            if(is_query)qm[group][word][local]=value;else km[group][word][local]=value;
        }
        __syncthreads();
        for(unsigned group=0u;group<groups;++group){
#pragma unroll
            for(unsigned q=0u;q<QueryCells;++q){
                core::Metadata left;
#pragma unroll
                for(unsigned i=0u;i<4u;++i)left.deficits[i]=qm[group][i][qr+q*16u];
                left.maximum=int(qm[group][4u][qr+q*16u]);
                int maximum[KeyCells];float scale[KeyCells];uint32_t modulo[KeyCells];bool negative[KeyCells];
#pragma unroll
                for(unsigned k=0u;k<KeyCells;++k){
                    core::Metadata right;
#pragma unroll
                    for(unsigned i=0u;i<4u;++i)right.deficits[i]=km[group][i][kc+k*16u];
                    right.maximum=int(km[group][4u][kc+k*16u]);
                    maximum[k]=core::alignment(carry[q][k],core::product_maximum(left,right));
                    scale[k]=core::f32::alignment::from_bits(uint32_t(152-maximum[k])<<23u);
                    modulo[k]=uint32_t(int32_t(carry[q][k]*scale[k]));
                }
#pragma unroll
                for(unsigned pair=0u;pair<8u;++pair){
                    const uint32_t a=qvalues[qr+q*16u][group*8u+pair];
                    const float alo=core::f32::alignment::from_bits(a<<16u);
                    const float ahi=core::f32::alignment::from_bits(a&0xffff0000u);
#pragma unroll
                    for(unsigned k=0u;k<KeyCells;++k){
                        const uint32_t b=kvalues[group*8u+pair][kc+k*16u];
                        const float lo=alo*core::f32::alignment::from_bits(b<<16u);
                        const float hi=ahi*core::f32::alignment::from_bits(b&0xffff0000u);
                        modulo[k]+=uint32_t(int32_t(lo*scale[k]));
                        modulo[k]+=uint32_t(int32_t(hi*scale[k]));
                        if(!pair)negative[k]=((a^b)&0x8000u)!=0u;
                    }
                }
#pragma unroll
                for(unsigned k=0u;k<KeyCells;++k)carry[q][k]=core::finish(modulo[k],negative[k],maximum[k]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(unsigned q=0u;q<QueryCells;++q){
#pragma unroll
        for(unsigned k=0u;k<KeyCells;++k)
            output[(size_t(query_tile+qr+q*16u)*16u+head)*stride+key_tile+kc+k*16u]=
                carry[q][k]*qrt_blackwell_attention::kExactScale;
    }
}

template<unsigned Queries>
inline int launch(const void* state,const uint16_t* query,const uint16_t* transposed_key,
    float* output,hipStream_t stream,unsigned start,unsigned count,unsigned stride,unsigned key_stride){
    if(!state||!query||!transposed_key||!output)return int(hipErrorInvalidValue);
    const auto& w=*static_cast<const Workspace*>(state);const auto& o=w.original;
    if(!qrt_narrow_domain_qk::valid(o)||!w.query_metadata||!w.key_metadata||!w.query_flags||!w.key_flags||
        !count||count>128u||start>=o.tokens||count>o.tokens-start||stride!=start+count||key_stride!=o.tokens)
        return int(hipErrorInvalidValue);
    const dim3 grid((stride+63u)/64u,16u,(count+Queries*16u-1u)/(Queries*16u));
    hipLaunchKernelGGL((scores<Queries>),grid,dim3(256u),0u,stream,query,transposed_key,w,output,start,count,stride);
    auto status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL((qrt_narrow_domain_qk::scores<false,Queries,4u,64u>),grid,dim3(256u),0u,stream,
        o.query,o.key,o.query_flags,o.key_flags,w.query_flags,w.key_flags,output,o.tile_counts,
        start,count,stride,key_stride);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,
        dim3((size_t(count)*16u*stride+255u)/256u),dim3(256u),0u,stream,
        query,transposed_key,output,start,count,stride,key_stride);
    return int(hipGetLastError());
}
} // namespace qrt_byte_exponent_qk
