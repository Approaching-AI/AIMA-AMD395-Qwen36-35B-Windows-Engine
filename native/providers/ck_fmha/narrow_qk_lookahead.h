#pragma once
#include "byte_exponent_qk.h"
#include "../moe_accumulator/sm121_narrow_qk_plan.h"

namespace qrt_narrow_qk_lookahead {
namespace core=qrt_sm121_byte_exponents;
namespace planned=qrt_sm121_narrow_qk_plan;
using B16=uint16_t __attribute__((ext_vector_type(16)));
using F8=float __attribute__((ext_vector_type(8)));
struct Workspace { qrt_byte_exponent_qk::Workspace operands; unsigned* rejected_scores; };

// All accepted cells use the same proved narrow K256 arithmetic. Packed
// exponent minima establish each K16 scale before any product is formed.
template<unsigned Groups,bool ForceMismatch=false>
__global__ void scores(const uint16_t* query,const uint16_t* transposed_key,
    Workspace owner,float* output,unsigned start,unsigned count,unsigned stride) {
    static_assert(Groups==2u||Groups==4u);
    constexpr unsigned QueryCells=2u,KeyCells=2u,Window=Groups*16u;
    constexpr unsigned rows=32u,columns=32u,groups=Groups;
    const auto& w=owner.operands;
    __shared__ uint32_t qvalues[rows][Window/2u+1u],kvalues[Window/2u][columns];
    __shared__ uint32_t qm[groups][5u][rows],km[groups][5u][columns];
    __shared__ float partials[Groups][rows*columns];
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
    float carry[QueryCells][KeyCells]{},predictor[QueryCells][KeyCells]{};
    bool fallback[QueryCells][KeyCells]{};
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
        const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u,source=lane%16u;
        for(unsigned job=wave;job<Groups*4u;job+=8u){
            const unsigned group=job/4u,subtile=job%4u,rbase=(subtile/2u)*16u,cbase=(subtile%2u)*16u;
            B16 left{},right{};
#pragma unroll
            for(unsigned i=0u;i<16u;++i){
                left[i]=uint16_t(qvalues[rbase+source][group*8u+i/2u]>>((i%2u)*16u));
                right[i]=uint16_t(kvalues[group*8u+i/2u][cbase+source]>>((i%2u)*16u));
            }
            const F8 zero{};
            const F8 values=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(left,right,zero);
#pragma unroll
            for(unsigned i=0u;i<8u;++i)
                partials[group][(rbase+2u*i+lane/16u)*columns+cbase+source]=values[i];
        }
        __syncthreads();
        planned::Plan plans[QueryCells][KeyCells][Groups];
#pragma unroll
        for(unsigned group=0u;group<groups;++group){
#pragma unroll
            for(unsigned q=0u;q<QueryCells;++q){
                core::Metadata left;
#pragma unroll
                for(unsigned i=0u;i<4u;++i)left.deficits[i]=qm[group][i][qr+q*16u];
                left.maximum=int(qm[group][4u][qr+q*16u]);
                int maximum[KeyCells],product_maximum[KeyCells];
                float scale[KeyCells];uint32_t modulo[KeyCells]{};bool negative[KeyCells];
#pragma unroll
                for(unsigned k=0u;k<KeyCells;++k){
                    core::Metadata right;
#pragma unroll
                    for(unsigned i=0u;i<4u;++i)right.deficits[i]=km[group][i][kc+k*16u];
                    right.maximum=int(km[group][4u][kc+k*16u]);
                    product_maximum[k]=core::product_maximum(left,right);
                    const int predicted=planned::predicted_exponent(predictor[q][k]);
                    predictor[q][k]+=partials[group][(qr+q*16u)*columns+kc+k*16u];
                    maximum[k]=planned::maximum(product_maximum[k],predicted);
                    if constexpr(ForceMismatch)maximum[k]=(product_maximum[k]>-89?product_maximum[k]:-89)+1;
                    scale[k]=maximum[k]==512?0.0f:core::f32::alignment::from_bits(uint32_t(152-maximum[k])<<23u);
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
                for(unsigned k=0u;k<KeyCells;++k)
                    plans[q][k][group]=maximum[k]==512?planned::Plan{}:
                        planned::plan::encode(modulo[k],maximum[k],product_maximum[k],negative[k]);
            }
        }
#pragma unroll
        for(unsigned q=0u;q<QueryCells;++q){
#pragma unroll
            for(unsigned k=0u;k<KeyCells;++k){
#pragma unroll
                for(unsigned group=0u;group<Groups;++group)if(!fallback[q][k]){
                    float next;
                    if(planned::apply(carry[q][k],plans[q][k][group],&next))carry[q][k]=next;
                    else fallback[q][k]=true;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(unsigned q=0u;q<QueryCells;++q){
#pragma unroll
        for(unsigned k=0u;k<KeyCells;++k){
            if(fallback[q][k])atomicAdd(owner.rejected_scores,1u);
            output[(size_t(query_tile+qr+q*16u)*16u+head)*stride+key_tile+kc+k*16u]=
                fallback[q][k]?core::f32::alignment::from_bits(qrt_deferred_qk_fallback::deferred_bits):
                carry[q][k]*qrt_blackwell_attention::kExactScale;
        }
    }
}

template<unsigned Groups,bool ForceMismatch=false>
inline int launch(const void* state,const uint16_t* query,const uint16_t* transposed_key,
    float* output,hipStream_t stream,unsigned start,unsigned count,unsigned stride,unsigned key_stride){
    if(!state||!query||!transposed_key||!output)return int(hipErrorInvalidValue);
    const auto& owner=*static_cast<const Workspace*>(state);
    const auto& w=owner.operands;const auto& o=w.original;
    if(!qrt_narrow_domain_qk::valid(o)||!w.query_metadata||!w.key_metadata||!w.query_flags||!w.key_flags||!owner.rejected_scores||
        !count||count>128u||start>=o.tokens||count>o.tokens-start||stride!=start+count||key_stride!=o.tokens)
        return int(hipErrorInvalidValue);
    const dim3 grid((stride+31u)/32u,16u,(count+31u)/32u);
    hipLaunchKernelGGL((scores<Groups,ForceMismatch>),grid,dim3(256u),0u,stream,query,transposed_key,owner,output,start,count,stride);
    auto status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL((qrt_narrow_domain_qk::scores<false,2u,2u,64u>),grid,dim3(256u),0u,stream,
        o.query,o.key,o.query_flags,o.key_flags,w.query_flags,w.key_flags,output,o.tile_counts,
        start,count,stride,key_stride);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,
        dim3((size_t(count)*16u*stride+255u)/256u),dim3(256u),0u,stream,
        query,transposed_key,output,start,count,stride,key_stride);
    return int(hipGetLastError());
}
} // namespace qrt_narrow_qk_lookahead
