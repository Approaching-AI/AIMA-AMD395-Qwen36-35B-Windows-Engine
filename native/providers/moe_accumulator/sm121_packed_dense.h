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
// reduction widths use sixteen strided FMA chains. Shared-down (K=512)
// instead uses thirty-two contiguous sixteen-element chains. Both reduce
// lane partials by successive halves, then round once to BF16.
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

QRT_PACKED_DENSE_HD inline double midpoint_add(double a, double b) {
#if defined(__HIP_DEVICE_COMPILE__)
    double result;
    asm("v_add_f64 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
#else
    volatile double result = a + b;
    return result;
#endif
}

// A32-row gate's FP32 reduction can lose a small product and land exactly
// on a BF16 midpoint. Resolve that ambiguity before BF16 rounding, rather
// than rounding the already-rounded FP32 midpoint a second time. Other
// endpoints retain the original reduction. This depends only on shape and
// arithmetic, never on layer, head, position, token or reference values.
template<class Input>
QRT_PACKED_DENSE_HD inline uint16_t gate_midpoint(float sum,
    const Input* input, const uint16_t* weight) {
    const uint32_t bits = qrt_sm121_exp2::bits(sum);
    if ((bits & 0xffffu) != 0x8000u || (bits & 0x7f800000u) == 0x7f800000u)
        return qrt_sm121_q1::bf16(sum);
    double precise = 0.0, correction = 0.0;
    for (unsigned i = 0; i < 2048u; ++i) {
        const double product = double(operand(input[i])) * double(operand(weight[i]));
        const double next = midpoint_add(precise, product);
        const double residual = std::abs(precise) >= std::abs(product)
            ? midpoint_add(midpoint_add(precise, -next), product)
            : midpoint_add(midpoint_add(product, -next), precise);
        correction = midpoint_add(correction, residual);
        precise = next;
    }
    precise = midpoint_add(precise, correction);
    if (precise == double(sum)) return qrt_sm121_q1::bf16(sum);
    const uint16_t base = static_cast<uint16_t>(bits >> 16u);
    const bool next = (bits >> 31u) ? precise < double(sum) : precise > double(sum);
    return static_cast<uint16_t>(base + unsigned(next));
}

#if defined(__HIPCC__) || defined(__CUDACC__)
template<unsigned K, class Input>
__device__ inline float dot(const Input* input, const uint16_t* weight) {
    const unsigned lane = threadIdx.x % lanes<K>;
    float sum = lane_dot<K>(input, weight, lane);
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
    const float sum = dot<K>(input + query * K, weights + size_t(column) * K);
    if (threadIdx.x % lanes<K> == 0u) {
        uint16_t rounded = qrt_sm121_q1::bf16(sum);
        if constexpr (K == 2048u) {
            if (columns == 32u)
                rounded = gate_midpoint(sum, input + query * K, weights + size_t(column) * K);
        }
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
