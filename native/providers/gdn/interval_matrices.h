#ifndef QRT_FLA_INTERVAL_MATRICES_H
#define QRT_FLA_INTERVAL_MATRICES_H
#include "blackwell_scalar_state.h"
#include "consumer_interval.h"

// Isolated component route: original ordered K16 intervals and exact fallback
// at BF16 consumers. No production dispatcher includes this header.
namespace qrt_fla_interval {
namespace scalar=qrt_fla_blackwell_scalar;
namespace consumer=qrt_fla_consumer_interval;
namespace interval=qrt_sm121_projection_interval;
using scalar::from_bf16;using scalar::to_bf16;using scalar::exponential;
using scalar::eligible;using scalar::pack;using scalar::dot;
constexpr unsigned threads=256u,columns=16u;
using B16=unsigned short __attribute__((ext_vector_type(16)));
using F8=float __attribute__((ext_vector_type(8)));
struct Ranges{F8 lower,upper;};
template<unsigned Width,unsigned Rows,unsigned Pitch,unsigned Columns>
__device__ __forceinline__ Ranges matrix(const uint32_t (&left)[Rows][Pitch],
    const uint32_t (&right)[Width/2u][Columns],unsigned row_base,unsigned column_base) {
    const unsigned lane=threadIdx.x%32u;
    Ranges ranges{};
#pragma unroll 1
    for(unsigned base=0u;base<Width;base+=16u) {
        B16 a{},b{},aa{},bb{};interval::Row ar,br;
#pragma unroll
        for(unsigned k=0u;k<16u;++k) {
            const unsigned feature=base+k;
            a[k]=uint16_t(left[row_base+lane%16u][feature/2u]>>(feature%2u*16u));
            b[k]=uint16_t(right[feature/2u][column_base+lane%16u]>>(feature%2u*16u));
            aa[k]=a[k]&0x7fffu;bb[k]=b[k]&0x7fffu;
            interval::include(ar,a[k]);interval::include(br,b[k]);
        }
        const F8 zero{};
        const F8 products=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a,b,zero);
        const F8 absolute=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(aa,bb,zero);
#pragma unroll
        for(unsigned item=0u;item<8u;++item) {
            const unsigned source=2u*item+lane/16u;
            const interval::Row row{__shfl(ar.minimum,source),__shfl(ar.maximum,source),bool(__shfl(int(ar.valid),source))};
            const auto next=interval::group({ranges.lower[item],ranges.upper[item]},products[item],absolute[item],row,br);
            ranges.lower[item]=next.lower;ranges.upper[item]=next.upper;
        }
    }
    return ranges;
}
// Unique per-CTA counters avoid a contended global atomic. They are diagnostics
// included in the component clock, with unused capacity checked by the caller.
__device__ __forceinline__ void report(unsigned admitted,unsigned considered,unsigned* statistics) {
    __shared__ unsigned counts[8][2];const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u;
    for(unsigned offset=16u;offset;offset>>=1u){admitted+=__shfl_down(admitted,offset);considered+=__shfl_down(considered,offset);}
    if(!lane){counts[wave][0]=admitted;counts[wave][1]=considered;}
    __syncthreads();
    if(!threadIdx.x){unsigned a=0u,c=0u;for(unsigned i=0u;i<8u;++i){a+=counts[i][0];c+=counts[i][1];}
        const size_t block=(size_t(blockIdx.z)*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x;
        statistics[block*2u]=a;statistics[block*2u+1u]=c;}
}
// A CTA owns every row of its eight V columns and captures all V before
// any U write. The production U=V alias therefore keeps its original owner.
__global__ void wu_kernel(const uint16_t* k,const uint16_t* v,const uint16_t* beta,
    const uint16_t* inverse,const float* g,uint16_t* w,uint16_t* u,unsigned count,
    const unsigned char* table,unsigned* statistics) {
    __shared__ uint32_t inv[64][33],keys[32][columns],values[32][columns];
    __shared__ unsigned inv_ok[64],key_ok[columns],value_ok[columns];
    const unsigned tid=threadIdx.x,offset=blockIdx.z*64u,head=blockIdx.y;
    const unsigned first_column=blockIdx.x*columns,valid=min(64u,count-offset);
    __shared__ float cached_decay[64],cached_scale[64];
    if(tid<valid){cached_decay[tid]=exponential(g[size_t(offset+tid)*32u+head],table);cached_scale[tid]=from_bf16(beta[size_t(offset+tid)*32u+head]);}
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
                const float scale=cached_scale[row];
                const uint16_t scaled=to_bf16(from_bf16(k[(token*16u+head/2u)*128u+first_column+column])*scale);
                ks[half]=to_bf16(from_bf16(scaled)*cached_decay[row]);
                vs[half]=to_bf16(from_bf16(v[(token*32u+head)*128u+first_column+column])*scale);
            }
        }
        keys[pair][column]=pack(ks[0],ks[1]);values[pair][column]=pack(vs[0],vs[1]);
        if (!eligible(ks[0],ks[1])) atomicAnd(&key_ok[column],0u);
        if (!eligible(vs[0],vs[1])) atomicAnd(&value_ok[column],0u);
    }
    __syncthreads();
    const unsigned wave=tid/32u,lane=tid%32u,row_base=(wave%4u)*16u;
    const auto ranges=matrix<64u>(inv,wave<4u?keys:values,row_base,0u);
    unsigned admitted=0u,considered=0u;
    for(unsigned item=0u;item<8u;++item) {
        const unsigned row=row_base+2u*item+lane/16u,column=lane%16u;
        if(row<valid) {
            uint16_t result;
            const bool accepted=consumer::rounded({ranges.lower[item],ranges.upper[item]},&result);
            if(!accepted)result=to_bf16(dot<64u,columns>(inv[row],wave<4u?keys:values,column,inv_ok[row]&&(wave<4u?key_ok[column]:value_ok[column])));
            const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;
            (wave<4u?w:u)[index]=result;admitted+=unsigned(accepted);++considered;
        }
    }
    report(admitted,considered,statistics);
}

// Packed feature-major right operands avoid repeated row-stride LDS bank
// collisions. Padding the left row pitch separates adjacent query banks.
__global__ void output_kernel(const uint16_t* q,const uint16_t* v,const uint16_t* h,
    const float* g,const uint16_t* scores,float* output,unsigned count,
    const unsigned char* table,unsigned* statistics) {
    constexpr unsigned columns=32u;
    __shared__ uint32_t queries[64][65],score_rows[64][33];
    __shared__ uint32_t values[32][columns],checkpoint[64][columns];
    __shared__ unsigned q_ok[64],score_ok[64],v_ok[columns],h_ok[columns];
    const unsigned tid=threadIdx.x,offset=blockIdx.z*64u,head=blockIdx.y;
    const unsigned first_column=blockIdx.x*columns,valid=min(64u,count-offset);
    __shared__ float cached_decay[64];
    if(tid<valid)cached_decay[tid]=exponential(g[size_t(offset+tid)*32u+head],table);
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
    const unsigned wave=tid/32u,lane=tid%32u,row_base=(wave/2u)*16u,column_base=(wave%2u)*16u;
    const auto prior=matrix<128u>(queries,checkpoint,row_base,column_base);
    const auto local=matrix<64u>(score_rows,values,row_base,column_base);
    unsigned admitted=0u,considered=0u;
    for(unsigned item=0u;item<8u;++item) {
        const unsigned row=row_base+2u*item+lane/16u,column=column_base+lane%16u;
        if(row<valid) {
            uint16_t result;
            const bool accepted=consumer::output({prior.lower[item],prior.upper[item]},
                {local.lower[item],local.upper[item]},cached_decay[row],&result);
            if(!accepted) {
                const float old=dot<128u,columns>(queries[row],checkpoint,column,q_ok[row]&&h_ok[column]);
                const float value=dot<64u,columns>(score_rows[row],values,column,score_ok[row]&&v_ok[column]);
                constexpr float scale=0.08838834764831845f;
                result=to_bf16(fmaf(value,scale,(old*cached_decay[row])*scale));
            }
            output[(size_t(offset+row)*32u+head)*128u+first_column+column]=from_bf16(result);
            admitted+=unsigned(accepted);++considered;
        }
    }
    report(admitted,considered,statistics);
}
// Retain complete state columns throughout one bounded segment. Calls that
// capture prefix checkpoints continue to use their original implementation.
__global__ void state_kernel(const uint16_t* k,const uint16_t* u,const uint16_t* w,
    const float* g,uint16_t* h,uint16_t* v_new,float* state,unsigned count,
    const unsigned char* table,unsigned* statistics) {
    constexpr unsigned Columns=16u;
    __shared__ float current[Columns][128],decay[64],segment_decay;
    __shared__ uint32_t rounded[64][Columns],residual[32][Columns];
    __shared__ uint16_t residual_words[64][Columns];
    __shared__ uint32_t keys[128][33],weights[64][65];
    __shared__ unsigned key_ok[128],weight_ok[64],state_ok[Columns],residual_ok[Columns];
    const unsigned tid=threadIdx.x,head=blockIdx.y,first_column=blockIdx.x*Columns;
    for (unsigned cell=tid;cell<Columns*128u;cell+=threads)
        current[cell/128u][cell%128u]=state[(head*128u+first_column+cell/128u)*128u+cell%128u];
    __syncthreads();
    unsigned admitted=0u,considered=0u;
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
        const unsigned wave=tid/32u,lane=tid%32u;
        if(wave<4u) {
            const unsigned row_base=wave*16u;
            const auto ranges=matrix<128u>(weights,rounded,row_base,0u);
            for(unsigned item=0u;item<8u;++item) {
                const unsigned row=row_base+2u*item+lane/16u,column=lane%16u;
                uint16_t value=0u;
                if(row<valid) {
                    const size_t index=(size_t(offset+row)*32u+head)*128u+first_column+column;
                    uint16_t updated;
                    const bool accepted=consumer::residual({ranges.lower[item],ranges.upper[item]},
                        from_bf16(u[index]),decay[row],&updated,&value);
                    if(!accepted) {
                        const float sum=dot<128u,Columns>(weights[row],rounded,column,weight_ok[row]&&state_ok[column]);
                        const float difference=from_bf16(u[index])-sum;
                        updated=to_bf16(difference);value=to_bf16(difference*decay[row]);
                    }
                    v_new[index]=updated;admitted+=unsigned(accepted);++considered;
                }
                residual_words[row][column]=value;
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
        for (unsigned cell=tid;cell<128u*Columns;cell+=threads) {
            const unsigned feature=cell/Columns,column=cell%Columns;
            const float sum=dot<64u,Columns>(keys[feature],residual,column,key_ok[feature]&&residual_ok[column]);
            current[column][feature]=fmaf(current[column][feature],segment_decay,sum);
        }
        __syncthreads();
    }
    report(admitted,considered,statistics);
    for (unsigned cell=tid;cell<Columns*128u;cell+=threads)
        state[(head*128u+first_column+cell/128u)*128u+cell%128u]=current[cell/128u][cell%128u];
}
}
#endif
