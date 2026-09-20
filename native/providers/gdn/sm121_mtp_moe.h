#pragma once
#include <hip/hip_runtime.h>
#include "sm121_mtp_projection.h"
#include "sm121_mtp_moe_math.h"
#include "sm121_mtp_moe_layout.h"

namespace qrt_sm121_mtp {
struct MoeWeights {
    const uint16_t* router = nullptr;         // [256,2048]
    const uint16_t* shared_gate = nullptr;    // [1,2048]
    const uint16_t* shared_gate_up = nullptr; // [1024,2048]
    const uint16_t* shared_down = nullptr;    // [2048,512]
    const uint16_t* routed_gate_up = nullptr; // [256,1024,2048]
    const uint16_t* routed_down = nullptr;    // [256,2048,512]
};
struct MoeTables {
    const uint16_t* silu = nullptr; // 65536 BF16 domain entries, after header.
    const uint16_t* sigmoid = nullptr; // 65536 BF16 domain entries.
    const uint32_t* router_exp_fraction = nullptr; // 8388608 FP32 bit patterns.
};

namespace mtp_moe_detail {
__global__ void router(const uint16_t* logits, const uint32_t* fraction,
                       uint32_t* ids, float* weights, uint32_t* invalid) {
    if (threadIdx.x) return;
    const unsigned row = blockIdx.x;
    if (!moe_route(logits + size_t(row) * 256u, fraction, ids + row * 8u, weights + row * 8u)) {
        atomicOr(invalid, 1u);
        for (unsigned route = 0; route < 8u; ++route) {
            ids[row * 8u + route] = 0u;
            weights[row * 8u + route] = 0.0f;
        }
    }
}

__global__ void shared_gate(const uint16_t* input, const uint16_t* weights,
                            uint16_t* output, uint32_t* invalid) {
    const unsigned lane = threadIdx.x;
    if (lane >= 16u) return;
    float value = qrt_sm121_shared_gate::lane_dot(input + size_t(blockIdx.x) * 2048u, weights, lane);
    for (unsigned offset = 8u; offset; offset >>= 1u)
        value = qrt_sm121_q1::add(value, __shfl_down(value, offset, 16));
    if (!lane) {
        if (!moe_finite(value)) atomicOr(invalid, 2u);
        output[blockIdx.x] = qrt_sm121_q1::bf16(value);
    }
}

template<bool Down>
__global__ void routed_projection(const uint16_t* input, const uint16_t* weights,
    const uint32_t* ids, const float* route_weights, uint16_t* output,
    uint32_t* invalid, unsigned first_cell, unsigned end_cell) {
    constexpr unsigned input_features = Down ? 512u : 2048u;
    constexpr unsigned output_features = Down ? 2048u : 1024u;
    const unsigned cell = first_cell + (blockIdx.x * blockDim.x + threadIdx.x) / 4u;
    if (cell >= end_cell) return;
    const unsigned feature = cell % output_features, slot = cell / output_features;
    const unsigned expert = ids[slot];
    if (expert >= 256u) {
        if (!(threadIdx.x & 3u)) { output[cell] = 0u; atomicOr(invalid, 4u); }
        return;
    }
    const unsigned input_row = Down ? slot : slot / 8u;
    float value = qrt_sm121_subgroup::dot<4u, 1u, false>(
        input + size_t(input_row) * input_features,
        weights + (size_t(expert) * output_features + feature) * input_features,
        input_features);
    if (!(threadIdx.x & 3u)) {
        if constexpr (Down) value = qrt_sm121_q1::multiply(value, route_weights[slot]);
        if (!moe_finite(value)) atomicOr(invalid, 8u);
        output[cell] = qrt_sm121_q1::bf16(value);
    }
}

__global__ void activate(MoeBuffers buffers, const uint16_t* silu, unsigned rows) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * 4608u) return;
    const unsigned row = index / 4608u, column = index % 4608u;
    if (column < 512u) {
        const size_t base = size_t(row) * 1024u + column;
        buffers.shared_activated[size_t(row) * 512u + column] =
            moe_activate(buffers.shared_gate_up[base], buffers.shared_gate_up[base + 512u], silu);
    } else {
        const unsigned route_column = column - 512u;
        const size_t base = (size_t(row) * 8u + route_column / 512u) * 1024u + route_column % 512u;
        buffers.routed_activated[size_t(row) * 4096u + route_column] =
            moe_activate(buffers.routed_gate_up[base], buffers.routed_gate_up[base + 512u], silu);
    }
}

__global__ void finish(MoeBuffers buffers, const uint16_t* sigmoid, unsigned rows) {
    const unsigned cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= rows * 2048u) return;
    const unsigned row = cell / 2048u, column = cell % 2048u;
    const uint16_t shared = moe_shared_product(buffers.shared_gate[row], buffers.shared_down[cell], sigmoid);
    const uint16_t routed = moe_routed_sum(buffers.routed_weighted + size_t(row) * 16384u, column);
    const uint16_t output = moe_output(shared, routed);
    buffers.shared[cell] = shared;
    buffers.routed[cell] = routed;
    buffers.output[cell] = output;
    if (!moe_finite(qrt_sm121_q1::widen(output))) atomicOr(buffers.invalid, 16u);
}

inline hipError_t dense(const uint16_t* weights, const uint16_t* input, uint16_t* output,
    unsigned output_features, unsigned input_features, unsigned rows, unsigned maximum_blocks, hipStream_t stream) {
    const unsigned cells = rows * output_features, capacity = maximum_blocks * 64u;
    for (unsigned first = 0; first < cells; first += capacity) {
        const unsigned remaining = cells - first, count = remaining < capacity ? remaining : capacity;
        hipLaunchKernelGGL(projection_kernel, dim3((count + 63u) / 64u), dim3(256u), 0u, stream,
            weights, input, output, output_features, input_features, first, first + count);
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) return status;
    }
    return hipSuccess;
}

template<bool Down>
inline hipError_t routed(const uint16_t* input, const uint16_t* weights, const MoeBuffers& buffers,
                         unsigned rows, unsigned maximum_blocks, hipStream_t stream) {
    const unsigned cells = rows * 8u * (Down ? 2048u : 1024u), capacity = maximum_blocks * 64u;
    for (unsigned first = 0; first < cells; first += capacity) {
        const unsigned remaining = cells - first, count = remaining < capacity ? remaining : capacity;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(routed_projection<Down>), dim3((count + 63u) / 64u), dim3(256u), 0u, stream,
            input, weights, buffers.topk_ids, buffers.topk_weights,
            Down ? buffers.routed_weighted : buffers.routed_gate_up, buffers.invalid, first, first + count);
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) return status;
    }
    return hipSuccess;
}
} // namespace mtp_moe_detail

// Asynchronous original TP1/EP1 MTP MoE, with no allocation or host transfer.
// The caller drains partial submissions on failure, and publishes output only
// after completion and a zero device invalid flag. Workspace must remain live.
inline hipError_t launch_moe(const uint16_t* input, const MoeWeights& weights, const MoeTables& tables,
    void* workspace, size_t workspace_bytes, unsigned rows, unsigned maximum_blocks = 1024u,
    hipStream_t stream = nullptr) {
    MoeBuffers buffers;
    if (!maximum_blocks || maximum_blocks > 4096u ||
        !bind_moe_buffers(workspace, workspace_bytes, rows, &buffers)) return hipErrorInvalidValue;
    const size_t needed = moe_workspace_bytes(rows);
    struct Borrowed { const void* pointer; size_t bytes; };
    const Borrowed borrowed[] = {
        {input, size_t(rows) * 2048u * 2u}, {weights.router, 256u * 2048u * 2u},
        {weights.shared_gate, 2048u * 2u}, {weights.shared_gate_up, 1024u * 2048u * 2u},
        {weights.shared_down, 2048u * 512u * 2u},
        {weights.routed_gate_up, size_t(256u) * 1024u * 2048u * 2u},
        {weights.routed_down, size_t(256u) * 2048u * 512u * 2u},
        {tables.silu, 65536u * 2u}, {tables.sigmoid, 65536u * 2u},
        {tables.router_exp_fraction, 8388608u * 4u}
    };
    for (const auto& item : borrowed)
        if (!moe_disjoint(workspace, needed, item.pointer, item.bytes)) return hipErrorInvalidValue;
    hipError_t status = hipMemsetAsync(buffers.invalid, 0, sizeof(uint32_t), stream);
    if (status != hipSuccess) return status;
    status = mtp_moe_detail::dense(weights.router, input, buffers.router, 256u, 2048u, rows, maximum_blocks, stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(mtp_moe_detail::router, dim3(rows), dim3(32u), 0u, stream,
        buffers.router, tables.router_exp_fraction, buffers.topk_ids, buffers.topk_weights, buffers.invalid);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(mtp_moe_detail::shared_gate, dim3(rows), dim3(32u), 0u, stream,
        input, weights.shared_gate, buffers.shared_gate, buffers.invalid);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    status = mtp_moe_detail::dense(weights.shared_gate_up, input, buffers.shared_gate_up, 1024u, 2048u, rows, maximum_blocks, stream);
    if (status != hipSuccess) return status;
    status = mtp_moe_detail::routed<false>(input, weights.routed_gate_up, buffers, rows, maximum_blocks, stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(mtp_moe_detail::activate, dim3(rows * 18u), dim3(256u), 0u, stream, buffers, tables.silu, rows);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    status = mtp_moe_detail::dense(weights.shared_down, buffers.shared_activated, buffers.shared_down, 2048u, 512u, rows, maximum_blocks, stream);
    if (status != hipSuccess) return status;
    status = mtp_moe_detail::routed<true>(buffers.routed_activated, weights.routed_down, buffers, rows, maximum_blocks, stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(mtp_moe_detail::finish, dim3(rows * 8u), dim3(256u), 0u, stream, buffers, tables.sigmoid, rows);
    return hipGetLastError();
}
} // namespace qrt_sm121_mtp
