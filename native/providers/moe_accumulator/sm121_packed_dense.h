#pragma once
#include "../gdn/sm121_q1_math.h"
#include <type_traits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PACKED_DENSE_HD __host__ __device__
#else
#define QRT_PACKED_DENSE_HD
#endif

namespace qrt_sm121_packed_dense {
// Original non-speculative single-row cuBLAS projections. The 2048/4096
// reduction widths use sixteen strided FMA chains, except the 32-row gates
// which require thirty-two. Shared-down (K=512) uses thirty-two contiguous
// sixteen-element chains. Reduce by successive halves, then round to BF16.
template<unsigned K> constexpr unsigned lanes = K == 512u ? 32u : 16u;
QRT_PACKED_DENSE_HD inline float operand(uint16_t value) {
    return qrt_sm121_q1::widen(value);
}
QRT_PACKED_DENSE_HD inline float operand(float value) {
    return qrt_sm121_q1::widen(qrt_sm121_q1::bf16(value));
}
QRT_PACKED_DENSE_HD inline float fma(float x, float w, float sum) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __fmaf_rn(x, w, sum);
#else
    return std::fma(x, w, sum);
#endif
}

template<unsigned K, class Input>
QRT_PACKED_DENSE_HD inline float lane_dot(const Input* input,
    const uint16_t* weight, unsigned lane) {
    static_assert(K == 512u || K == 2048u || K == 4096u, "qualified original dense shape");
    static_assert(std::is_same<Input,uint16_t>::value || std::is_same<Input,float>::value,
                  "BF16 input or its widened resident carrier");
    float sum = 0.0f;
    if constexpr (K == 512u) {
        for (unsigned i = lane * 16u; i < (lane + 1u) * 16u; ++i)
            sum = fma(operand(input[i]), operand(weight[i]), sum);
    } else {
        for (unsigned i = lane; i < K; i += 16u)
            sum = fma(operand(input[i]), operand(weight[i]), sum);
    }
    return sum;
}

// Two independent chains per physical lane reproduce the 32 logical lanes
// of the original K2048/N32 gate. Their first reduction combines lanes i and
// i+16; the existing 16-lane shuffle performs the remaining halves. Preserve
// the launch geometry and resolve the actual reduction order, including
// non-midpoint errors that a higher-precision dot cannot repair.
template<class Input>
QRT_PACKED_DENSE_HD inline float gate_lane_dot(const Input* input,
    const uint16_t* weight, unsigned lane) {
    static_assert(std::is_same<Input,uint16_t>::value || std::is_same<Input,float>::value,
                  "BF16 input or its widened resident carrier");
    float first = 0.0f, second = 0.0f;
    for (unsigned i = lane; i < 2048u; i += 32u) {
        first = fma(operand(input[i]), operand(weight[i]), first);
        second = fma(operand(input[i + 16u]), operand(weight[i + 16u]), second);
    }
    return qrt_sm121_q1::add(first, second);
}

#if defined(__HIPCC__) || defined(__CUDACC__)
template<unsigned K, class Input, bool Gate = false>
__device__ inline float dot(const Input* input, const uint16_t* weight) {
    static_assert(!Gate || K == 2048u, "original gate input width");
    const unsigned lane = threadIdx.x % lanes<K>;
    float sum;
    if constexpr (Gate) sum = gate_lane_dot(input, weight, lane);
    else sum = lane_dot<K>(input, weight, lane);
    for (unsigned offset = lanes<K> / 2u; offset; offset >>= 1u)
        sum = qrt_sm121_q1::add(sum, __shfl_down(sum, offset, lanes<K>));
    return sum;
}

// Each query is independently evaluated as an original one-row projection.
// A two-query component replay does not imply speculative GEMM arithmetic.
template<unsigned K, class Input, class Output>
__global__ void projection(const Input* input, const uint16_t* weights,
    Output* output, unsigned columns, unsigned queries) {
    static_assert(std::is_same<Output,uint16_t>::value || std::is_same<Output,float>::value,
                  "BF16 endpoint or its widened resident carrier");
    const size_t cell = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / lanes<K>;
    if (cell >= size_t(columns) * queries) return;
    const unsigned column = static_cast<unsigned>(cell % columns);
    const size_t query = cell / columns;
    float sum;
    if constexpr (K == 2048u) {
        sum = columns == 32u
            ? dot<K,Input,true>(input + query * K, weights + size_t(column) * K)
            : dot<K>(input + query * K, weights + size_t(column) * K);
    } else sum = dot<K>(input + query * K, weights + size_t(column) * K);
    if (threadIdx.x % lanes<K> == 0u) {
        const uint16_t rounded = qrt_sm121_q1::bf16(sum);
        if constexpr (std::is_same<Output,uint16_t>::value) output[cell] = rounded;
        else output[cell] = qrt_sm121_q1::widen(rounded);
    }
}

__global__ void shared_activation(const uint16_t* input, const uint16_t* gate,
    const uint16_t* up, const uint16_t* silu, uint16_t* output) {
    const unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / 16u;
    if (row >= 512u) return;
    const float g = dot<2048u>(input, gate + size_t(row) * 2048u);
    const float u = dot<2048u>(input, up + size_t(row) * 2048u);
    if ((threadIdx.x & 15u) == 0u) output[row] = qrt_sm121_q1::bf16(
        qrt_sm121_q1::multiply(qrt_sm121_q1::widen(silu[qrt_sm121_q1::bf16(g)]),
                             qrt_sm121_q1::widen(qrt_sm121_q1::bf16(u))));
}
#endif
} // namespace qrt_sm121_packed_dense

#undef QRT_PACKED_DENSE_HD
