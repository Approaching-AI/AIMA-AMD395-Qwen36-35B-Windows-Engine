#ifndef QRT_SM121_COARSE_PROJECTION_MATRIX_H
#define QRT_SM121_COARSE_PROJECTION_MATRIX_H
#include <hip/hip_runtime.h>
#include "sm121_coarse_projection_bound.h"
#include "sm121_wmma_operand_load.h"

// Isolated complete producer, with no product dispatcher. Original selected
// replay supplies ambiguous or unsupported BF16 endpoints in its test owner.
// Internal launch preconditions: positive rows/tokens, positive width divisible
// by16, complete operand/flag spans, and rows*tokens<=UINT32_MAX for compaction.
namespace qrt_sm121_coarse_projection_matrix {
namespace bound=qrt_sm121_coarse_projection_bound;
constexpr unsigned threads=256u,row_tile=128u;
using B16=uint16_t __attribute__((ext_vector_type(16)));
using F8=float __attribute__((ext_vector_type(8)));
__global__ void eligibility(const uint16_t* input,unsigned* flags,unsigned rows,unsigned width) {
    const unsigned row=blockIdx.x,lane=threadIdx.x;
    if(row>=rows)return;
    unsigned valid=1u;
    for(unsigned k=lane;k<width;k+=threads)valid&=unsigned(bound::eligible(input[size_t(row)*width+k]));
    __shared__ unsigned all[threads];all[lane]=valid;__syncthreads();
    for(unsigned stride=threads/2u;stride;stride>>=1u){if(lane<stride)all[lane]&=all[lane+stride];__syncthreads();}
    if(!lane)flags[row]=all[0];
}
template<unsigned Chunk,unsigned Fragments=1u,bool VectorLoads=false,unsigned NativeErrorBits=19u>
__global__ __launch_bounds__(threads) void produce(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_ok,const unsigned* input_ok,float* centers,float* errors,
    unsigned rows,unsigned tokens,unsigned width) {
    static_assert(Chunk==64u || Chunk==128u || Chunk==256u);
    static_assert(Fragments==1u || Fragments==2u);
    const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u,source=lane%16u;
    const unsigned row=blockIdx.x*row_tile+wave*16u+source;
    const unsigned first_token=blockIdx.y*(16u*Fragments);
    const bool valid_weight=row<rows && weight_ok[row];
    // This route's operand/eligibility spans are immutable for the launch.
    const uint16_t* vector_weight=nullptr;const uint16_t* vector_inputs[Fragments]{};
    bool vector_input_ok[Fragments]{};
    if constexpr(VectorLoads){
        if(valid_weight)vector_weight=weights+size_t(row)*width;
#pragma unroll
        for(unsigned fragment=0u;fragment<Fragments;++fragment){
            const unsigned token=first_token+fragment*16u+source;
            vector_input_ok[fragment]=token<tokens&&input_ok[token];
            if(vector_input_ok[fragment])vector_inputs[fragment]=inputs+size_t(token)*width;
        }
    }
    F8 centers_local[Fragments]{},errors_local[Fragments]{};
    for(unsigned coarse=0u;coarse<width;coarse+=Chunk) {
        F8 partial[Fragments]{},positive[Fragments]{};
#pragma unroll 1
        for(unsigned base=coarse;base<coarse+Chunk && base<width;base+=16u) {
            B16 w{},wa{};
            if constexpr(VectorLoads){
                w=qrt_sm121_wmma_operand_load::read<B16>(vector_weight,base,valid_weight);
#pragma unroll
                for(unsigned k=0u;k<16u;++k)wa[k]=w[k]&0x7fffu;
            }else{
#pragma unroll
                for(unsigned k=0u;k<16u;++k){w[k]=valid_weight?weights[size_t(row)*width+base+k]:0u;wa[k]=w[k]&0x7fffu;}
            }
#pragma unroll
            for(unsigned fragment=0u;fragment<Fragments;++fragment) {
                const unsigned token=first_token+fragment*16u+source;
                B16 x{},xa{};
                if constexpr(VectorLoads){
                    x=qrt_sm121_wmma_operand_load::read<B16>(vector_inputs[fragment],base,vector_input_ok[fragment]);
#pragma unroll
                    for(unsigned k=0u;k<16u;++k)xa[k]=x[k]&0x7fffu;
                }else{const bool valid_input=token<tokens&&input_ok[token];
#pragma unroll
                    for(unsigned k=0u;k<16u;++k){x[k]=valid_input?inputs[size_t(token)*width+base+k]:0u;xa[k]=x[k]&0x7fffu;}
                }
                const F8 zero{};
                // Explicit zero-C products followed by FP32 additions. Native
                // C accumulation is not substituted for this error model.
                const F8 signed_product=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x,w,zero);
                const F8 absolute_product=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(xa,wa,zero);
                partial[fragment]+=signed_product;positive[fragment]+=absolute_product;
            }
        }
#pragma unroll
        for(unsigned fragment=0u;fragment<Fragments;++fragment) {
#pragma unroll
            for(unsigned item=0u;item<8u;++item) {
                const auto next=bound::advance<Chunk/16u,NativeErrorBits>({centers_local[fragment][item],errors_local[fragment][item]},partial[fragment][item],positive[fragment][item]);
                centers_local[fragment][item]=next.center;errors_local[fragment][item]=next.error;
            }
        }
    }
    if(row>=rows)return;
#pragma unroll
    for(unsigned fragment=0u;fragment<Fragments;++fragment) {
#pragma unroll
        for(unsigned item=0u;item<8u;++item) {
            const unsigned token=first_token+fragment*16u+2u*item+lane/16u;
            if(token<tokens){const size_t cell=size_t(token)*rows+row;centers[cell]=centers_local[fragment][item];
                errors[cell]=valid_weight && input_ok[token]?errors_local[fragment][item]:bound::scalar::infinity();}
        }
    }
}
// One allocation per CTA, with stable lane ownership and no per-cell global
// atomic. The owner provides capacity for every output cell and a zero counter.
__global__ void compact(const float* centers,const float* errors,float* output,
    unsigned* indices,unsigned* count,unsigned cells) {
    const unsigned cell=blockIdx.x*threads+threadIdx.x,lane=threadIdx.x%32u,wave=threadIdx.x/32u;
    const bool selected=cell<cells && !bound::certified({centers[cell],errors[cell]});
    const unsigned mask=__ballot(selected);__shared__ unsigned offsets[8],first;
    if(!lane)offsets[wave]=__popc(mask);__syncthreads();
    if(!threadIdx.x){unsigned total=0u;for(unsigned i=0u;i<8u;++i){const unsigned n=offsets[i];offsets[i]=total;total+=n;}first=atomicAdd(count,total);}
    __syncthreads();
    if(cell<cells)output[cell]=centers[cell];
    if(selected)indices[first+offsets[wave]+__popc(mask&((1u<<lane)-1u))]=cell;
}
}
#endif
