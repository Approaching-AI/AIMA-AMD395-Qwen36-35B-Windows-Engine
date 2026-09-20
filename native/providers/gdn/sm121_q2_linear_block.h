#pragma once
#include "sm121_q2_linear.h"
#include "sm121_q2_linear_block_layout.h"
#include "sm121_q2_gated_math.h"
#include "../moe_accumulator/sm121_q1_moe.h"

namespace qrt_sm121_q2 {
namespace linear_block_detail {
__global__ void gated_norm(const uint16_t* core, const uint16_t* z,
    const uint16_t* weights, const unsigned char* rsqrt, const float* silu,
    uint16_t* output) {
    const unsigned lane = threadIdx.x;
    const size_t first = size_t(blockIdx.x)*128u;
    float sum = gated_lane_sum(core + first, lane);
    for (unsigned offset = 16u; offset; offset >>= 1u)
        sum = qrt_sm121_q1::add(sum, __shfl_down(sum, offset, 32));
    const float inverse = gated_inverse(__shfl(sum, 0, 32), rsqrt);
    for (unsigned item = 0; item < 4u; ++item) {
        const unsigned col = lane*4u + item;
        output[first + col] = gated_value(core[first + col], z[first + col],
                                          weights[col], inverse, silu);
    }
}

template<unsigned K> inline hipError_t project(const uint16_t* input,
    const uint16_t* weights, uint16_t* output, unsigned columns, hipStream_t stream) {
    for (unsigned row = 0; row < scheduled_rows; ++row) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_sm121_q1_moe::projection<K>),
            dim3((columns + 15u)/16u), dim3(256u), 0u, stream,
            input + size_t(row)*K, weights, output + size_t(row)*columns, columns);
        const auto status = hipGetLastError();
        if (status != hipSuccess) return status;
    }
    return hipSuccess;
}
} // namespace linear_block_detail

// Allocation-free, private two-row target computation. Success means all
// producers were submitted, not that they completed or may publish state.
// The enclosing transaction must drain or quarantine borrowers on any failure,
// sample its actual target outputs, then commit only the accepted state/ring.
template<class Element> inline hipError_t launch_linear_block(const LinearBlockViews<Element>& v,
    const LinearBlockTables& t, hipStream_t stream = nullptr) {
    if (!valid_linear_block(v, t)) return hipErrorInvalidValue;
    const uint16_t* weights[] = {v.qkv_weights, v.z_weights, v.a_weights, v.b_weights};
    uint16_t* outputs[] = {v.qkv, v.z, v.a, v.b};
    const unsigned columns[] = {8192u, 4096u, 32u, 32u};
    for (unsigned part = 0; part < 4u; ++part) {
        const auto status = linear_block_detail::project<2048u>(
            v.normalized_input, weights[part], outputs[part], columns[part], stream);
        if (status != hipSuccess) return status;
    }
    auto status = launch_linear(v.convolution, v.recurrent, t.recurrent, stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(linear_block_detail::gated_norm, dim3(64u), dim3(32u), 0u, stream,
        v.recurrent.staged_core, v.z, v.norm_weights, t.recurrent.rsqrt, t.gated_silu, v.gated);
    status = hipGetLastError();
    return status == hipSuccess ? linear_block_detail::project<4096u>(
        v.gated, v.output_weights, v.output, 2048u, stream) : status;
}
} // namespace qrt_sm121_q2
