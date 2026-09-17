#pragma once
#include "blackwell_scalar_state.h"

// Isolated complete WU/state/output component. Gate values are computed once
// per token/head with the original SHA-bound table. The copied kernels below
// change only their exponential reads; ordered dots, aliases and rounding stay
// unchanged. Production dispatch does not include this header.
namespace qrt_fla_prepared_gate {
using qrt_fla_blackwell_scalar::from_bf16;
using qrt_fla_blackwell_scalar::to_bf16;
using qrt_fla_blackwell_scalar::eligible;
using qrt_fla_blackwell_scalar::pack;
using qrt_fla_blackwell_scalar::dot;
constexpr unsigned threads=256u,columns=8u;
struct GateView {const float* exponential;const float* remaining;};

__global__ void prepare_kernel(const float* g,float* exponential,float* remaining,
    unsigned count,const unsigned char* table) {
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=count*32u)return;
    const unsigned token=cell/32u,head=cell%32u;
    const unsigned last=min((token/64u+1u)*64u,count)-1u;
    exponential[cell]=qrt_fla_blackwell_scalar::exponential(g[cell],table);
    remaining[cell]=qrt_fla_blackwell_scalar::exponential(g[size_t(last)*32u+head]-g[cell],table);
}
inline hipError_t prepare(const float* g,float* exponential,float* remaining,
    unsigned count,size_t words,const unsigned char* table,hipStream_t stream) {
    if(!g||!exponential||!remaining||!table||!count||count>8192u||words<size_t(count)*32u)
        return hipErrorInvalidValue;
    const uintptr_t begin[]={reinterpret_cast<uintptr_t>(g),reinterpret_cast<uintptr_t>(exponential),reinterpret_cast<uintptr_t>(remaining)};
    const size_t bytes=size_t(count)*32u*sizeof(float);
    for(unsigned i=0u;i<3u;++i){
        if(begin[i]%alignof(float)||begin[i]>UINTPTR_MAX-bytes)return hipErrorInvalidValue;
        for(unsigned j=0u;j<i;++j)
            if(begin[i]<begin[j]+bytes&&begin[j]<begin[i]+bytes)return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(prepare_kernel,dim3((count*32u+255u)/256u),dim3(threads),0u,stream,
        g,exponential,remaining,count,table);
    return hipGetLastError();
}

__global__ void wu_kernel(const uint16_t* k,const uint16_t* v,const uint16_t* beta,
    const uint16_t* inverse,const float* g,uint16_t* w,uint16_t* u,unsigned count,
    GateView gates) {
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
                ks[half]=to_bf16(from_bf16(scaled)*gates.exponential[token*32u+head]);
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
    GateView gates) {
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
        const float prior=old*gates.exponential[size_t(offset+row)*32u+head];
        output[(size_t(offset+row)*32u+head)*128u+first_column+column]=
            from_bf16(to_bf16(fmaf(local,scale,prior*scale)));
    }
}

template<unsigned Columns>
__global__ void state_kernel(const uint16_t* k,const uint16_t* u,const uint16_t* w,
    const float* g,uint16_t* h,uint16_t* v_new,float* state,unsigned count,
    GateView gates) {
    static_assert(Columns==4u || Columns==8u);
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
        if (tid<valid) decay[tid]=gates.remaining[size_t(offset+tid)*32u+head];
        if (!tid) segment_decay=gates.exponential[size_t(offset+valid-1u)*32u+head];
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
        for (unsigned cell=tid;cell<64u*Columns;cell+=threads) {
            const unsigned row=cell/Columns,column=cell%Columns;
            uint16_t value=0u;
            if (row<valid) {
                const float sum=dot<128u,Columns>(weights[row],rounded,column,weight_ok[row]&&state_ok[column]);
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
        for (unsigned cell=tid;cell<128u*Columns;cell+=threads) {
            const unsigned feature=cell/Columns,column=cell%Columns;
            const float sum=dot<64u,Columns>(keys[feature],residual,column,key_ok[feature]&&residual_ok[column]);
            current[column][feature]=fmaf(current[column][feature],segment_decay,sum);
        }
        __syncthreads();
    }
    for (unsigned cell=tid;cell<Columns*128u;cell+=threads)
        state[(head*128u+first_column+cell/128u)*128u+cell%128u]=current[cell/128u][cell%128u];
}

} // namespace qrt_fla_prepared_gate
