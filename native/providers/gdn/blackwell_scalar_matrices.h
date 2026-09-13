#ifndef QRT_FLA_BLACKWELL_SCALAR_MATRICES_H
#define QRT_FLA_BLACKWELL_SCALAR_MATRICES_H
#include "blackwell_accumulator.h"
#include "sm121_exp2_table.h"
#include "../moe_accumulator/sm121_float_alignment.h"
#include <cstdlib>
#include <cstring>

namespace qrt_fla_blackwell_scalar {
using qrt_fla_blackwell::from_bf16;
using qrt_fla_blackwell::to_bf16;
constexpr unsigned threads = 256u, columns = 8u;
inline int mode() {
    const char* value = std::getenv("QRT_FLA_GDN_SCALAR_FLOAT_MATRICES");
    if (!value || !*value || !std::strcmp(value,"0")) return 0;
    return !std::strcmp(value,"1") ? 1 : -1;
}
__device__ __forceinline__ bool eligible(uint16_t a, uint16_t b) {
    return qrt_sm121_float_alignment::eligible(a) && qrt_sm121_float_alignment::eligible(b);
}
__device__ __forceinline__ uint32_t pack(uint16_t a, uint16_t b) {
    return uint32_t(a) | (uint32_t(b) << 16u);
}
__device__ __forceinline__ float exponential(float x, const unsigned char* table) {
    return qrt_sm121_exp2::evaluate(table, x * 1.4426950408889634074f);
}
template<unsigned Width, unsigned RightColumns = columns>
__device__ __forceinline__ float dot(const uint32_t* left,
    const uint32_t (&right)[Width / 2u][RightColumns], unsigned column, bool valid) {
    qrt_q1_moe_hawkeye::Value carry{0u,-133,false};
    for (unsigned base=0u;base<Width;base+=16u) {
        qrt_sm121_group16::AlignedSum sum;
        bool accepted=false;
        if (valid) {
            qrt_sm121_float_alignment::Group group;
#pragma unroll
            for (unsigned i=0u;i<16u;i+=2u) {
                const uint32_t a=left[(base+i)/2u],b=right[(base+i)/2u][column];
                group.set(i,uint16_t(a),uint16_t(b));
                group.set(i+1u,uint16_t(a>>16u),uint16_t(b>>16u));
            }
            accepted=qrt_sm121_float_alignment::sum(carry,group,&sum);
        }
        if (!accepted) {
            uint32_t products[16];
#pragma unroll
            for (unsigned i=0u;i<16u;i+=2u) {
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
__global__ void wu_kernel(const uint16_t* k,const uint16_t* v,const uint16_t* beta,
    const uint16_t* inverse,const float* g,uint16_t* w,uint16_t* u,unsigned count,
    const unsigned char* table) {
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
    for (unsigned cell=tid;cell<valid*columns;cell+=threads) {
        const unsigned row=cell/columns,column=cell%columns;
        const float sw=dot<64u>(inv[row],keys,column,inv_ok[row]&&key_ok[column]);
        const float su=dot<64u>(inv[row],values,column,inv_ok[row]&&value_ok[column]);
        const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;
        w[index]=to_bf16(sw);u[index]=to_bf16(su);
    }
}

// Packed feature-major right operands avoid repeated row-stride LDS bank
// collisions. Padding the left row pitch separates adjacent query banks.
__global__ void output_kernel(const uint16_t* q,const uint16_t* v,const uint16_t* h,
    const float* g,const uint16_t* scores,float* output,unsigned count,
    const unsigned char* table) {
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
    for (unsigned cell=tid;cell<valid*columns;cell+=threads) {
        const unsigned row=cell/columns,column=cell%columns;
        const float old=dot<128u>(queries[row],checkpoint,column,q_ok[row]&&h_ok[column]);
        const float local=dot<64u>(score_rows[row],values,column,score_ok[row]&&v_ok[column]);
        constexpr float scale=0.08838834764831845f;
        const float prior=old*exponential(g[size_t(offset+row)*32u+head],table);
        output[(size_t(offset+row)*32u+head)*128u+first_column+column]=
            from_bf16(to_bf16(fmaf(local,scale,prior*scale)));
    }
}
} // namespace qrt_fla_blackwell_scalar
#endif
