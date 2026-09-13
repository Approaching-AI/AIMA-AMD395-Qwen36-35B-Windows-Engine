#ifndef QRT_SM121_SCALED_PROJECTION_H
#define QRT_SM121_SCALED_PROJECTION_H
#include "sm121_scalar_projection.h"
#include "sm121_scaled_significand.h"

namespace qrt_sm121_scaled_projection {
namespace scaled = qrt_sm121_scaled_significand;
using Value = qrt_q1_moe_hawkeye::Value;
constexpr unsigned threads = 256u, lanes = 4u;

__global__ void normal_rows_kernel(const uint16_t* input, unsigned* flags, unsigned rows, unsigned width) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    __shared__ unsigned invalid;
    if (!threadIdx.x) invalid = 0u;
    __syncthreads();
    bool bad = false;
    for (unsigned k = threadIdx.x; k < width; k += blockDim.x)
        bad |= !scaled::eligible(input[size_t(row) * width + k]);
    if (bad) atomicOr(&invalid, 1u);
    __syncthreads();
    if (!threadIdx.x) flags[row] = invalid == 0u;
}

template<bool Audit = false>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
    unsigned width, bool normal, uint32_t* trace = nullptr, uint32_t* stats = nullptr) {
    if constexpr (!Audit) if (!normal) return qrt_sm121_subgroup::dot<lanes>(left,right,width);
    const unsigned lane = threadIdx.x & 3u;
    Value carry{0u,-133,false};
#pragma unroll 1
    for (unsigned base = 0u; base < width; base += 16u) {
        uint64_t a,b;
        __builtin_memcpy(&a,left+base+lane*4u,sizeof(a));
        __builtin_memcpy(&b,right+base+lane*4u,sizeof(b));
        if (normal) {
            scaled::Pair pairs[4];
            int maximum = carry.exponent > -133 ? carry.exponent : -133;
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) {
                pairs[i] = scaled::prepare(uint16_t(a>>(i*16u)),uint16_t(b>>(i*16u)));
                maximum = pairs[i].exponent > maximum ? pairs[i].exponent : maximum;
            }
            maximum = qrt_sm121_lane_reduce::maximum<lanes>(maximum);
            uint32_t modulo = 0u;
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) modulo += scaled::aligned(pairs[i],maximum);
            modulo = qrt_sm121_lane_reduce::sum<lanes>(modulo);
            const unsigned shift = unsigned(maximum-carry.exponent);
            const uint32_t aligned = shift >= 32u ? 0u : (carry.significand<<2u)>>shift;
            modulo += carry.negative ? 0u-aligned : aligned;
            const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo,((a^b)&0x8000u)!=0u);
            carry = qrt_sm121_wave16::normalize(sum.magnitude,sum.negative,maximum);
        } else {
            uint32_t products[4];
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) products[i] = qrt_sm121_group16::pack_product(
                qrt_q1_moe_hawkeye::multiply_bf16(uint16_t(a>>(i*16u)),uint16_t(b>>(i*16u)),-133));
            carry = qrt_sm121_subgroup::accumulate_products<lanes>(carry,products);
        }
        if constexpr (Audit) if (!lane && trace) {
            trace[base/16u*3u] = carry.significand;
            trace[base/16u*3u+1u] = uint32_t(int32_t(carry.exponent));
            trace[base/16u*3u+2u] = unsigned(carry.negative);
        }
    }
    if constexpr (Audit) if (!lane && stats) {
        stats[0] = normal ? width/16u : 0u;
        stats[1] = normal ? 0u : width/16u;
        stats[2] = stats[3] = 0u;
    }
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}

template<bool Audit = false>
__global__ void replay_kernel(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_flags,const unsigned* input_flags,const unsigned* indices,
    unsigned count,float* output,unsigned rows,unsigned tokens,unsigned width,uint32_t* stats=nullptr) {
    const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/lanes;
    if(slot>=count) return;
    const unsigned cell=indices[slot];
    if(size_t(cell)>=size_t(rows)*tokens) return;
    const unsigned row=cell%rows,token=cell/rows;
    const float value=dot<Audit>(inputs+size_t(token)*width,weights+size_t(row)*width,width,
        weight_flags[row]&&input_flags[token],nullptr,stats?stats+size_t(slot)*4u:nullptr);
    if(!(threadIdx.x&3u)) {
        const uint32_t bits=__float_as_uint(value);
        output[cell]=qrt_sm121_float_alignment::from_bits((bits+0x7fffu+((bits>>16u)&1u))&0xffff0000u);
    }
}

inline hipError_t prepare(const uint16_t* input,unsigned* flags,size_t capacity,
    unsigned rows,unsigned width,hipStream_t stream) {
    if(!input || !flags || !rows || rows>16384u || !width || width>4096u || width%16u || capacity<rows)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(normal_rows_kernel,dim3(rows),dim3(threads),0u,stream,input,flags,rows,width);
    return hipGetLastError();
}

inline hipError_t launch(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_flags,size_t weight_capacity,const unsigned* input_flags,size_t input_capacity,
    const unsigned* indices,unsigned count,float* output,unsigned rows,unsigned tokens,unsigned width,hipStream_t stream) {
    if(!weights || !inputs || !weight_flags || !input_flags || !indices || !output ||
        !rows || rows>16384u || !tokens || tokens>8192u || !width || width>4096u || width%16u ||
        count>size_t(rows)*tokens || weight_capacity<rows || input_capacity<tokens) return hipErrorInvalidValue;
    if(!count) return hipSuccess;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(replay_kernel<false>),dim3((count*lanes+threads-1u)/threads),dim3(threads),0u,stream,
        weights,inputs,weight_flags,input_flags,indices,count,output,rows,tokens,width,nullptr);
    return hipGetLastError();
}
} // namespace qrt_sm121_scaled_projection
#endif
