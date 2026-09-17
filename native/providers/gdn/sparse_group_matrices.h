#pragma once
#include "blackwell_scalar_state.h"
#include "../moe_accumulator/sm121_sparse_group.h"

// Isolated complete GDN route using exact operand sparsity. The original
// packed operands, ordered K16 carries, checkpoints, aliases and final FMAs
// remain canonical. Shared masks come from actual operands, never references.
namespace qrt_fla_sparse_group {
namespace scalar=qrt_fla_blackwell_scalar;
namespace masks=qrt_sm121_sparse_group;
using Metadata=uint32_t;
struct Counts { unsigned groups,empty,short_groups,short_products; };
static_assert(sizeof(Counts)==16u);
template<bool Audit>
__device__ __forceinline__ void publish(Counts value,Counts* output) {
 if constexpr(Audit) {
  __shared__ Counts sums[256];sums[threadIdx.x]=value;__syncthreads();
  for(unsigned width=128u;width;width/=2u) {
   if(threadIdx.x<width) {auto& a=sums[threadIdx.x];const auto b=sums[threadIdx.x+width];
    a.groups+=b.groups;a.empty+=b.empty;a.short_groups+=b.short_groups;a.short_products+=b.short_products;}
   __syncthreads();
  }
  if(!threadIdx.x)output[blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z)]=sums[0];
 }
}
using scalar::from_bf16;using scalar::to_bf16;using scalar::exponential;
using scalar::eligible;using scalar::pack;
constexpr unsigned threads=scalar::threads,columns=scalar::columns;

template<unsigned Rows,unsigned Pitch,unsigned Groups>
__device__ __forceinline__ void prepare_left(const uint32_t (&input)[Rows][Pitch],
    Metadata (&metadata)[Rows][Groups]) {
    static_assert(Pitch>=Groups*8u);
    for(unsigned cell=threadIdx.x;cell<Rows*Groups;cell+=threads)
        metadata[cell/Groups][cell%Groups]=masks::prepare<1u>(&input[cell/Groups][(cell%Groups)*8u]);
}
template<unsigned Pairs,unsigned Columns>
__device__ __forceinline__ void prepare_right(const uint32_t (&input)[Pairs][Columns],
    Metadata (&metadata)[Pairs/8u][Columns]) {
    static_assert(Pairs%8u==0u);
    for(unsigned cell=threadIdx.x;cell<(Pairs/8u)*Columns;cell+=threads)
        metadata[cell/Columns][cell%Columns]=masks::prepare<Columns>(&input[(cell/Columns)*8u][cell%Columns]);
}
template<unsigned Width,bool Sparse,bool Audit,unsigned RightColumns = columns>
__device__ __forceinline__ float dot(const uint32_t* left,
    const uint32_t (&right)[Width / 2u][RightColumns], unsigned column, bool valid,
    const Metadata* lm,const Metadata (&rm)[Width/16u][RightColumns],Counts& counts) {
    qrt_q1_moe_hawkeye::Value carry{0u,-133,false};
    for (unsigned base=0u;base<Width;base+=16u) {
        if constexpr(Audit)++counts.groups;
        const uint32_t active=lm[base/16u]&rm[base/16u][column];
        if(!active) {carry=masks::empty(carry);if constexpr(Audit)++counts.empty;continue;}
        if constexpr(Sparse) if(masks::population(active)<=4u) {
            carry=masks::small<RightColumns>(carry,left+base/2u,&right[base/2u][column],active);
            if constexpr(Audit){++counts.short_groups;counts.short_products+=masks::population(active);}
            continue;
        }
        qrt_sm121_group16::AlignedSum sum;bool accepted=false;
        if(valid) {
            qrt_sm121_float_alignment::Group group;
#pragma unroll
            for(unsigned i=0u;i<16u;i+=2u) {
                const uint32_t a=left[(base+i)/2u],b=right[(base+i)/2u][column];
                group.set(i,uint16_t(a),uint16_t(b));group.set(i+1u,uint16_t(a>>16u),uint16_t(b>>16u));
            }
            accepted=qrt_sm121_float_alignment::sum(carry,group,&sum);
        }
        if(!accepted) {
            uint32_t products[16];
#pragma unroll
            for(unsigned i=0u;i<16u;i+=2u) {
                const uint32_t a=left[(base+i)/2u],b=right[(base+i)/2u][column];
                products[i]=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(uint16_t(a),uint16_t(b),-133));
                products[i+1u]=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(uint16_t(a>>16u),uint16_t(b>>16u),-133));
            }
            sum=qrt_sm121_group16::sum_packed(carry,products);
        }
        carry=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
    }
    return qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}

// A CTA owns every row of its eight V columns and captures all V before
// any U write. The production U=V alias therefore keeps its original owner.
template<bool Sparse,bool Audit>
__global__ void wu_kernel(const uint16_t* k,const uint16_t* v,const uint16_t* beta,
    const uint16_t* inverse,const float* g,uint16_t* w,uint16_t* u,unsigned count,
    const unsigned char* table,Counts* statistics) {
    Counts counts{};
    __shared__ uint32_t inv[64][33],keys[32][columns],values[32][columns];
    __shared__ Metadata im[64][4],km[4][columns],vm[4][columns];
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
    prepare_left(inv,im);prepare_right(keys,km);prepare_right(values,vm);
    __syncthreads();
    for (unsigned cell=tid;cell<valid*columns;cell+=threads) {
        const unsigned row=cell/columns,column=cell%columns;
        const float sw=dot<64u,Sparse,Audit>(inv[row],keys,column,inv_ok[row]&&key_ok[column],im[row],km,counts);
        const float su=dot<64u,Sparse,Audit>(inv[row],values,column,inv_ok[row]&&value_ok[column],im[row],vm,counts);
        const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;
        w[index]=to_bf16(sw);u[index]=to_bf16(su);
    }
    publish<Audit>(counts,statistics);
}

// Packed feature-major right operands avoid repeated row-stride LDS bank
// collisions. Padding the left row pitch separates adjacent query banks.
template<bool Sparse,bool Audit>
__global__ void output_kernel(const uint16_t* q,const uint16_t* v,const uint16_t* h,
    const float* g,const uint16_t* scores,float* output,unsigned count,
    const unsigned char* table,Counts* statistics) {
    Counts counts{};
    constexpr unsigned Rows=64u;
    __shared__ uint32_t queries[Rows][65],score_rows[Rows][33];
    __shared__ Metadata qm[Rows][8],sm[Rows][4],vm[4][columns],hm[8][columns];
    __shared__ uint32_t values[32][columns],checkpoint[64][columns];
    __shared__ unsigned q_ok[Rows],score_ok[Rows],v_ok[columns],h_ok[columns];
    const unsigned tid=threadIdx.x,offset=blockIdx.z*64u,head=blockIdx.y;
    const unsigned first_column=(blockIdx.x%(128u/columns))*columns;
    const unsigned row_base=(blockIdx.x/(128u/columns))*Rows;
    if(offset+row_base>=count)return;
    const unsigned valid=min(Rows,count-offset-row_base);
    if (tid<Rows) {q_ok[tid]=1u;score_ok[tid]=1u;}
    if (tid<columns) {v_ok[tid]=1u;h_ok[tid]=1u;}
    __syncthreads();
    for (unsigned cell=tid;cell<Rows*64u;cell+=threads) {
        const unsigned row=cell/64u,pair=cell%64u;
        uint16_t a=0u,b=0u;
        if (row<valid) {
            const size_t index=(size_t(offset+row_base+row)*16u+head/2u)*128u+pair*2u;
            a=q[index];b=q[index+1u];
        }
        queries[row][pair]=pack(a,b);
        if (!eligible(a,b)) atomicAnd(&q_ok[row],0u);
    }
    for (unsigned cell=tid;cell<Rows*32u;cell+=threads) {
        const unsigned row=cell/32u,pair=cell%32u;
        uint16_t a=0u,b=0u;
        if (row<valid) {
            const size_t index=(size_t(offset+row_base+row)*32u+head)*64u+pair*2u;
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
            if (pair*2u+half<min(64u,count-offset)) words[half]=v[(size_t(offset+pair*2u+half)*32u+head)*128u+first_column+column];
        values[pair][column]=pack(words[0],words[1]);
        if (!eligible(words[0],words[1])) atomicAnd(&v_ok[column],0u);
    }
    __syncthreads();
    prepare_left(queries,qm);prepare_left(score_rows,sm);
    prepare_right(values,vm);prepare_right(checkpoint,hm);
    __syncthreads();
    for (unsigned cell=tid;cell<valid*columns;cell+=threads) {
        const unsigned row=cell/columns,column=cell%columns;
        const float old=dot<128u,Sparse,Audit>(queries[row],checkpoint,column,q_ok[row]&&h_ok[column],qm[row],hm,counts);
        const float local=dot<64u,Sparse,Audit>(score_rows[row],values,column,score_ok[row]&&v_ok[column],sm[row],vm,counts);
        constexpr float scale=0.08838834764831845f;
        const float prior=old*exponential(g[size_t(offset+row_base+row)*32u+head],table);
        output[(size_t(offset+row_base+row)*32u+head)*128u+first_column+column]=
            from_bf16(to_bf16(fmaf(local,scale,prior*scale)));
    }
    publish<Audit>(counts,statistics);
}
// Retain complete state columns throughout one bounded segment. Calls that
// capture prefix checkpoints continue to use their original implementation.
template<unsigned Columns,bool Sparse,bool Audit>
__global__ void state_kernel(const uint16_t* k,const uint16_t* u,const uint16_t* w,
    const float* g,uint16_t* h,uint16_t* v_new,float* state,unsigned count,
    const unsigned char* table,Counts* statistics) {
    Counts counts{};
    static_assert(Columns==4u || Columns==8u);
    __shared__ float current[Columns][128],decay[64],segment_decay;
    __shared__ uint32_t rounded[64][Columns],residual[32][Columns];
    __shared__ uint16_t residual_words[64][Columns];
    __shared__ uint32_t keys[128][33],weights[64][65];
    __shared__ Metadata km[128][4],wm[64][8],hm[8][Columns],vm[4][Columns];
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
        prepare_left(keys,km);prepare_left(weights,wm);prepare_right(rounded,hm);
        __syncthreads();
        for (unsigned cell=tid;cell<64u*Columns;cell+=threads) {
            const unsigned row=cell/Columns,column=cell%Columns;
            uint16_t value=0u;
            if (row<valid) {
                const float sum=dot<128u,Sparse,Audit,Columns>(weights[row],rounded,column,weight_ok[row]&&state_ok[column],wm[row],hm,counts);
                const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;
                const float difference=from_bf16(u[index])-sum;
                v_new[index]=to_bf16(difference);
                value=to_bf16(difference*decay[row]);
            }
            residual_words[row][column]=value;
        }
        __syncthreads();
        for (unsigned cell=tid;cell<32u*Columns;cell+=threads) {
            const unsigned pair=cell/Columns,column=cell%Columns;
            const uint16_t a=residual_words[pair*2u][column],b=residual_words[pair*2u+1u][column];
            residual[pair][column]=pack(a,b);
            if (!eligible(a,b)) atomicAnd(&residual_ok[column],0u);
        }
        __syncthreads();
        prepare_right(residual,vm);
        __syncthreads();
        for (unsigned cell=tid;cell<128u*Columns;cell+=threads) {
            const unsigned feature=cell/Columns,column=cell%Columns;
            const float sum=dot<64u,Sparse,Audit,Columns>(keys[feature],residual,column,key_ok[feature]&&residual_ok[column],km[feature],vm,counts);
            current[column][feature]=fmaf(current[column][feature],segment_decay,sum);
        }
        __syncthreads();
    }
    for (unsigned cell=tid;cell<Columns*128u;cell+=threads)
        state[(head*128u+first_column+cell/128u)*128u+cell%128u]=current[cell/128u][cell%128u];
    publish<Audit>(counts,statistics);
}
} // namespace qrt_fla_sparse_group
