#pragma once
#include "expert_candidate_order.h"

// A class-major, then expert-major permutation of the original candidate list.
// The three contiguous class ranges let replay kernels omit ineligible paths
// without rescanning the entire list. Counts and ranges stay on the GPU.
namespace qrt_moe_class_expert_order {
constexpr unsigned experts = 256u, classes = 3u, buckets = classes * experts;
constexpr unsigned threads = 256u, maximum_blocks = 1024u;
constexpr size_t maximum_capacity = qrt_moe_expert_order::maximum_capacity;
constexpr size_t metadata_words = buckets + (buckets + 1u) + buckets;
enum class Projection : unsigned { Gate, Up, Down };
struct Workspace { uint32_t* storage = nullptr; size_t capacity = 0u; };
struct Rows { const uint32_t *input, *weights; Projection projection; };
inline size_t bytes(size_t capacity) {
    return capacity && capacity <= maximum_capacity ?
        (capacity + metadata_words) * sizeof(uint32_t) : 0u;
}
struct Views { uint32_t *indices, *counts, *offsets, *cursors; };
inline Views views(Workspace workspace) {
    auto* counts = workspace.storage + workspace.capacity;
    return {workspace.storage, counts, counts + buckets, counts + buckets + buckets + 1u};
}
__device__ __forceinline__ unsigned bucket(unsigned cell, const int32_t* topk_ids, Rows rows) {
    const bool down = rows.projection == Projection::Down;
    const unsigned columns = down ? 2048u : 512u, route = cell / columns;
    const unsigned expert = unsigned(topk_ids[route]);
    const unsigned weight_row = expert * (down ? 2048u : 1024u) + cell % columns +
        (rows.projection == Projection::Up ? 512u : 0u);
    const unsigned classification = rows.input[down ? route : route / 8u] & rows.weights[weight_row];
    return (classification == 3u ? 2u : classification == 1u ? 1u : 0u) * experts + expert;
}
__global__ void histogram(const uint32_t* indices, const uint32_t* count,
    const int32_t* topk_ids, Rows rows, uint32_t* counts) {
    for (unsigned slot = blockIdx.x * blockDim.x + threadIdx.x; slot < *count;
         slot += gridDim.x * blockDim.x)
        atomicAdd(counts + bucket(indices[slot], topk_ids, rows), 1u);
}
__global__ void prefix(const uint32_t* counts, uint32_t* offsets, uint32_t* cursors) {
    if (threadIdx.x) return;
    unsigned total = 0u;
    for (unsigned index = 0u; index < buckets; ++index) {
        offsets[index] = total; cursors[index] = total; total += counts[index];
    }
    offsets[buckets] = total;
}
__global__ void scatter(const uint32_t* indices, const uint32_t* count,
    const int32_t* topk_ids, Rows rows, uint32_t* cursors, uint32_t* ordered) {
    for (unsigned slot = blockIdx.x * blockDim.x + threadIdx.x; slot < *count;
         slot += gridDim.x * blockDim.x) {
        const unsigned cell = indices[slot];
        const unsigned destination = atomicAdd(cursors + bucket(cell, topk_ids, rows), 1u);
        ordered[destination] = cell;
    }
}
inline hipError_t launch(const uint32_t* indices, const uint32_t* count,
    const int32_t* topk_ids, Rows rows, Workspace workspace, hipStream_t stream) {
    if (!indices || !count || !topk_ids || !rows.input || !rows.weights ||
        unsigned(rows.projection) > unsigned(Projection::Down) || !workspace.storage ||
        !bytes(workspace.capacity) || indices == workspace.storage)
        return hipErrorInvalidValue;
    const auto view = views(workspace);
    const unsigned requested = unsigned((workspace.capacity + threads - 1u) / threads);
    const unsigned blocks = requested < maximum_blocks ? requested : maximum_blocks;
    auto status = hipMemsetAsync(view.counts, 0, buckets * sizeof(uint32_t), stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(histogram, dim3(blocks), dim3(threads), 0u, stream,
        indices, count, topk_ids, rows, view.counts);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(prefix, dim3(1u), dim3(threads), 0u, stream,
        view.counts, view.offsets, view.cursors);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(scatter, dim3(blocks), dim3(threads), 0u, stream,
        indices, count, topk_ids, rows, view.cursors, view.indices);
    return hipGetLastError();
}
} // namespace qrt_moe_class_expert_order
