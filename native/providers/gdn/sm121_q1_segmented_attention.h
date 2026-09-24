#pragma once
#include "sm121_q1_segmented_attention_math.h"
#include "../ck_fmha/blackwell_attention.h"

namespace qrt_sm121_q1_segmented_attention {
// A segment owns one head and an independent 16-token online softmax/PV
// chain. Prefix and decoded values remain in their actual separate owners.
__global__ void segment_kernel(const float* scores, const uint16_t* prefix_value,
    const uint16_t* tail_value, float* segment_output, float* segment_max,
    float* segment_sum, unsigned prefix_tokens, unsigned tokens, unsigned stride,
    const unsigned char* exp2_table) {
    const unsigned head = blockIdx.x, segment = blockIdx.y, column = threadIdx.x;
    const unsigned width = tiles_per_segment(tokens);
    const unsigned begin = segment*width, end = min(begin+width, (tokens+15u)/16u);
    const size_t scalar = size_t(head)*segments+segment;
    if (begin >= end) {
        segment_output[scalar*dimension+column] = 0.0f;
        if (!column) { segment_max[scalar] = -INFINITY; segment_sum[scalar] = 0.0f; }
        return;
    }
    __shared__ float score[tile_tokens], probability[tile_tokens];
    __shared__ uint16_t probability_bf16[tile_tokens];
    __shared__ float maximum, denominator, next_maximum, alpha;
    float accumulator = 0.0f;
    if (!column) { maximum = -INFINITY; denominator = 1.0f; }
    __syncthreads();
    for (unsigned tile = begin; tile < end; ++tile) {
        if (column < tile_tokens) {
            const unsigned token = tile*tile_tokens+column;
            score[column] = token < tokens ? scores[size_t(head)*stride+token] : -INFINITY;
        }
        __syncthreads();
        if (!column) {
            next_maximum = maximum;
            for (unsigned i = 0u; i < tile_tokens; ++i) next_maximum = fmaxf(next_maximum,score[i]);
            alpha = exponential(maximum-next_maximum, exp2_table);
        }
        __syncthreads();
        if (column < tile_tokens) {
            const unsigned token = tile*tile_tokens+column;
            probability[column] = token < tokens ? exponential(score[column]-next_maximum,exp2_table) : 0.0f;
            probability_bf16[column] = qrt_sm121_q1::bf16(probability[column]);
        }
        __syncthreads();
        if (!column) {
            denominator = fmaf(denominator,alpha,tile_sum(probability));
            maximum = next_maximum;
        }
        auto carry = qrt_q1_moe_hawkeye::value_from_float(
            qrt_sm121_q1::multiply(accumulator,alpha),-133);
        uint32_t products[tile_tokens];
#pragma unroll
        for (unsigned i = 0u; i < tile_tokens; ++i) {
            const unsigned token = tile*tile_tokens+i;
            uint16_t value = 0u;
            if (token < tokens) {
                const bool tail = token >= prefix_tokens;
                const unsigned local = tail ? token-prefix_tokens : token;
                value = (tail ? tail_value : prefix_value)[size_t(local)*512u+(head/8u)*256u+column];
            }
            products[i] = qrt_sm121_group16::pack_product(
                qrt_q1_moe_hawkeye::multiply_bf16(probability_bf16[i],value,-133));
        }
        const auto sum = qrt_sm121_group16::sum_packed(carry,products);
        carry = qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
        accumulator = qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
        __syncthreads();
    }
    segment_output[scalar*dimension+column] = accumulator;
    if (!column) { segment_max[scalar] = maximum; segment_sum[scalar] = denominator; }
}

__global__ void merge_kernel(const float* segment_output, const float* segment_max,
    const float* segment_sum, float* output, const unsigned char* exp2_table,
    const unsigned char* rcp_table) {
    const unsigned head = blockIdx.x, column = threadIdx.x;
    __shared__ float scales[segments], inverse[2];
    if (!column) {
        float maximum = -INFINITY;
        for (unsigned i = 0u; i < segments; ++i) maximum = fmaxf(maximum,segment_max[head*segments+i]);
        for (unsigned i = 0u; i < segments; ++i)
            scales[i] = exponential(segment_max[head*segments+i]-maximum,exp2_table);
        const float low = merge_denominator(segment_sum+head*segments,scales);
        const float high = merge_denominator_high(segment_sum+head*segments,scales);
        inverse[0] = low == 0.0f ? 0.0f : qrt_sm121_attention_rcp::evaluate(rcp_table,low);
        inverse[1] = high == 0.0f ? 0.0f : qrt_sm121_attention_rcp::evaluate(rcp_table,high);
    }
    __syncthreads();
    output[head*dimension+column] = qrt_sm121_q1::multiply(
        merge_numerator(segment_output+size_t(head)*segments*dimension+column,scales),
        inverse[(column >> 5u) & 1u]);
}

inline hipError_t launch(const float* scores, const uint16_t* prefix_value,
    const uint16_t* tail_value, float* segment_output, float* segment_max,
    float* segment_sum, float* output, unsigned prefix_tokens, unsigned tokens,
    unsigned stride, const unsigned char* exp2_table, const unsigned char* rcp_table,
    hipStream_t stream) {
    if (!scores || !prefix_value || !tail_value || !segment_output || !segment_max ||
        !segment_sum || !output || !exp2_table || !rcp_table || !tokens || tokens > 263680u ||
        prefix_tokens >= tokens || stride < tokens) return hipErrorInvalidValue;
    hipLaunchKernelGGL(segment_kernel,dim3(heads,segments),dim3(dimension),0u,stream,
        scores,prefix_value,tail_value,segment_output,segment_max,segment_sum,
        prefix_tokens,tokens,stride,exp2_table);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(merge_kernel,dim3(heads),dim3(dimension),0u,stream,
        segment_output,segment_max,segment_sum,output,exp2_table,rcp_table);
    return hipGetLastError();
}
} // namespace qrt_sm121_q1_segmented_attention
