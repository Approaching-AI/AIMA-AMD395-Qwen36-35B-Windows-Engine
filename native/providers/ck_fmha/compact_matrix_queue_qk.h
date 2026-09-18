#pragma once
#include "narrow_domain_qk_workspace.h"
#include "../moe_accumulator/sm121_compact_matrix_group.h"

// Isolated complete-QK producer: compact integer matrices prepare two groups
// ahead; original narrow groups are assigned to consecutive wave lanes through
// ballots and register shuffles. Four ordered carry values remain with each
// output owner. No carry or work queue is spilled to shared memory.
namespace qrt_compact_matrix_queue_qk {
namespace group=qrt_sm121_compact_matrix_group;
using Row=group::Row;
using I4=int __attribute__((ext_vector_type(4)));
using I8=int __attribute__((ext_vector_type(8)));
struct Workspace { qrt_narrow_domain_qk::Workspace original;const Row *query,*key; };

template<bool Key>
__global__ void prepare(const uint16_t* input,Row* output,unsigned tokens){
    constexpr unsigned heads=Key?2u:16u;
    const size_t item=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(item>=size_t(tokens)*heads*16u)return;
    const unsigned row=unsigned(item/16u),g=unsigned(item%16u);uint16_t raw[16];
#pragma unroll
    for(unsigned i=0u;i<16u;++i)raw[i]=input[item*16u+i];
    output[Key?(size_t(row%heads)*16u+g)*tokens+row/heads:item]=group::prepare(raw);
}

__device__ __forceinline__ float original_group(float carry,const uint32_t* a,const uint32_t* b){
    qrt_sm121_float_alignment::Group values;
    values.maximum=-133;
#pragma unroll
    for(unsigned pair=0u;pair<8u;++pair){
        const uint32_t x=a[pair],y=b[pair];
#pragma unroll
        for(unsigned j=0u;j<2u;++j){
            const uint16_t left=uint16_t(x>>(16u*j)),right=uint16_t(y>>(16u*j));
            values.products[2u*pair+j]=group::f32::alignment::from_bits(uint32_t(left)<<16u)*
                group::f32::alignment::from_bits(uint32_t(right)<<16u);
            const int exponent=(left&0x7fffu)&&(right&0x7fffu)?
                int((left>>7u)&255u)+int((right>>7u)&255u)-254:-133;
            values.maximum=values.maximum>exponent?values.maximum:exponent;
        }
        if(!pair)values.first_negative=((x^y)&0x8000u)!=0u;
    }
    return qrt_sm121_narrow_f32_carry::accumulate(carry,values);
}

template<bool Force>
__device__ __forceinline__ bool matrix_step(float& carry,const Row* rows,const int64_t* products,unsigned cell){
    if constexpr(Force)return false;
    return group::accumulate(carry,rows[cell/32u],rows[32u+cell%32u],products[cell],&carry);
}

template<bool Compact,bool Force=false>
__global__ __launch_bounds__(256) void scores(Workspace w,float* output,
    unsigned start,unsigned count,unsigned stride){
    constexpr unsigned rows=32u,columns=32u,groups=2u,words=sizeof(Row)/4u;
    __shared__ Row operands[groups][rows+columns];
    __shared__ uint32_t raw[groups][rows+columns][8];
    __shared__ int64_t products[groups][rows*columns];
    const unsigned tid=threadIdx.x,lane=tid%32u,wave=tid/32u;
    const unsigned head=blockIdx.y,kv=head/8u,qt=blockIdx.z*rows,kt=blockIdx.x*columns;
    const unsigned tokens=w.original.tokens;
    const bool interior=qt+rows<=count&&kt+columns<=stride&&kt+columns-1u<=start+qt;
    if(!interior)return;
    // Reuse a dead product word for the tile's complete-domain decision.
    auto* rejected=reinterpret_cast<unsigned*>(&products[0][0]);
    if(!tid)*rejected=0u;__syncthreads();
    if(tid<rows&&!w.original.query_domain[(start+qt+tid)*16u+head])atomicOr(rejected,1u);
    if(tid<columns&&!w.original.key_domain[(kt+tid)*2u+kv])atomicOr(rejected,1u);
    __syncthreads();const bool admitted=!*rejected;__syncthreads();
    if(!admitted)return;
    if(!tid)atomicAdd(w.original.tile_counts+1u,1u);
    float carry0=0.0f,carry1=0.0f,carry2=0.0f,carry3=0.0f;
    for(unsigned base=0u;base<16u;base+=groups){
        for(unsigned i=tid;i<groups*(rows+columns)*words;i+=256u){
            const unsigned word=i%words,position=(i/words)%(rows+columns),g=i/(words*(rows+columns));
            const bool query=position<rows;const unsigned local=query?position:position-rows;
            const Row* source=query?w.query+(size_t(start+qt+local)*16u+head)*16u+base+g:
                w.key+(size_t(kv)*16u+base+g)*tokens+kt+local;
            uint32_t value;__builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(source)+word*4u,4u);
            __builtin_memcpy(reinterpret_cast<unsigned char*>(&operands[g][position])+word*4u,&value,4u);
        }
        __syncthreads();
        for(unsigned i=tid;i<groups*(rows+columns);i+=256u){
            const unsigned g=i/(rows+columns),row=i%(rows+columns);
#pragma unroll
            for(unsigned pair=0u;pair<8u;++pair)raw[g][row][pair]=
                uint32_t(group::compact::original(operands[g][row].encoded,pair*2u))|
                (uint32_t(group::compact::original(operands[g][row].encoded,pair*2u+1u))<<16u);
        }
        if constexpr(!Force){
            const unsigned g=wave/4u,tile=wave%4u;
            const auto& a=operands[g][(tile/2u)*16u+lane%16u].encoded;
            const auto& b=operands[g][32u+(tile%2u)*16u+lane%16u].encoded;
            I4 ah{},al{},bh{},bl{};
#pragma unroll
            for(unsigned i=0u;i<4u;++i){
                const uint32_t a0=a.pairs[2u*i],a1=a.pairs[2u*i+1u],b0=b.pairs[2u*i],b1=b.pairs[2u*i+1u];
                al[i]=int((a0&0x00ff00ffu)|((a1&0x00ff00ffu)<<8u));
                ah[i]=int(((a0>>8u)&0x00ff00ffu)|(a1&0xff00ff00u));
                bl[i]=int((b0&0x00ff00ffu)|((b1&0x00ff00ffu)<<8u));
                bh[i]=int(((b0>>8u)&0x00ff00ffu)|(b1&0xff00ff00u));
            }
            const I8 zero{};
            const auto hh=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,ah,true,bh,zero,false);
            const auto hl=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,ah,false,bl,zero,false);
            const auto lh=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,al,true,bh,zero,false);
            const auto ll=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,al,false,bl,zero,false);
#pragma unroll
            for(unsigned i=0u;i<8u;++i){
                const unsigned row=(tile/2u)*16u+2u*i+lane/16u,column=(tile%2u)*16u+lane%16u;
                products[g][row*32u+column]=int64_t(hh[i])*65536+(int64_t(hl[i])+lh[i])*256+ll[i];
            }
        }
        // Both expanded original words and all matrix products are complete.
        __syncthreads();
#pragma unroll 1
        for(unsigned g=0u;g<groups;++g){
            const bool r0=!matrix_step<Force>(carry0,operands[g],products[g],tid);
            const bool r1=!matrix_step<Force>(carry1,operands[g],products[g],tid+256u);
            const bool r2=!matrix_step<Force>(carry2,operands[g],products[g],tid+512u);
            const bool r3=!matrix_step<Force>(carry3,operands[g],products[g],tid+768u);
            if constexpr(!Compact){
                if(r0)carry0=original_group(carry0,raw[g][tid/32u],raw[g][32u+tid%32u]);
                if(r1)carry1=original_group(carry1,raw[g][tid/32u+8u],raw[g][32u+tid%32u]);
                if(r2)carry2=original_group(carry2,raw[g][tid/32u+16u],raw[g][32u+tid%32u]);
                if(r3)carry3=original_group(carry3,raw[g][tid/32u+24u],raw[g][32u+tid%32u]);
            }else{
                const unsigned m0=__ballot(r0),m1=__ballot(r1),m2=__ballot(r2),m3=__ballot(r3);
                const unsigned n0=__popc(m0),n1=__popc(m1),n2=__popc(m2),n3=__popc(m3),total=n0+n1+n2+n3;
                const unsigned before=(uint32_t(1u)<<lane)-1u;
                const unsigned rank0=__popc(m0&before),rank1=n0+__popc(m1&before);
                const unsigned rank2=n0+n1+__popc(m2&before),rank3=n0+n1+n2+__popc(m3&before);
                for(unsigned first=0u;first<total;first+=32u){
                    const unsigned job=first+lane;unsigned rank=job,item=0u,mask=m0;
                    if(rank>=n0){rank-=n0;item=1u;mask=m1;
                        if(rank>=n1){rank-=n1;item=2u;mask=m2;
                            if(rank>=n2){rank-=n2;item=3u;mask=m3;}}}
                    const unsigned source=job<total?group::select_bit(mask,rank):0u;
                    // Every lane executes all shuffles, even the final inactive
                    // consumers. Sources belong to this wave and remain live.
                    const float c0=__shfl(carry0,source),c1=__shfl(carry1,source);
                    const float c2=__shfl(carry2,source),c3=__shfl(carry3,source);
                    float result=0.0f;
                    if(job<total){
                        const unsigned cell=wave*32u+source+item*256u;
                        const float initial=item==0u?c0:item==1u?c1:item==2u?c2:c3;
                        result=original_group(initial,raw[g][cell/32u],raw[g][32u+cell%32u]);
                    }
                    const float result0=__shfl(result,rank0%32u),result1=__shfl(result,rank1%32u);
                    const float result2=__shfl(result,rank2%32u),result3=__shfl(result,rank3%32u);
                    if(r0&&rank0/32u==first/32u)carry0=result0;
                    if(r1&&rank1/32u==first/32u)carry1=result1;
                    if(r2&&rank2/32u==first/32u)carry2=result2;
                    if(r3&&rank3/32u==first/32u)carry3=result3;
                }
            }
        }
        __syncthreads();
    }
    const size_t first=(size_t(qt+tid/32u)*16u+head)*stride+kt+tid%32u;
    output[first]=carry0*qrt_blackwell_attention::kExactScale;
    output[first+size_t(8u)*16u*stride]=carry1*qrt_blackwell_attention::kExactScale;
    output[first+size_t(16u)*16u*stride]=carry2*qrt_blackwell_attention::kExactScale;
    output[first+size_t(24u)*16u*stride]=carry3*qrt_blackwell_attention::kExactScale;
}

template<bool Compact,bool Force=false>
inline int launch(const void* state,const uint16_t* query,const uint16_t* transposed_key,
    float* output,hipStream_t stream,unsigned start,unsigned count,unsigned stride,unsigned key_stride){
    if(!state||!query||!transposed_key||!output)return int(hipErrorInvalidValue);
    const auto& w=*static_cast<const Workspace*>(state);const auto& o=w.original;
    if(!qrt_narrow_domain_qk::valid(o)||!w.query||!w.key||!count||count>128u||
        start>=o.tokens||count>o.tokens-start||stride!=start+count||key_stride!=o.tokens)
        return int(hipErrorInvalidValue);
    const dim3 grid((stride+31u)/32u,16u,(count+31u)/32u);
    hipLaunchKernelGGL((scores<Compact,Force>),grid,dim3(256u),0u,stream,w,output,start,count,stride);
    auto status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL((qrt_narrow_domain_qk::scores<false,2u,2u,64u>),grid,dim3(256u),0u,stream,
        o.query,o.key,o.query_flags,o.key_flags,o.query_domain,o.key_domain,output,o.tile_counts,
        start,count,stride,key_stride);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,
        dim3((size_t(count)*16u*stride+255u)/256u),dim3(256u),0u,stream,
        query,transposed_key,output,start,count,stride,key_stride);
    return int(hipGetLastError());
}
} // namespace qrt_compact_matrix_queue_qk
