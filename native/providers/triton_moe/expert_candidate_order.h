#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

// A permutation of an already collected routed candidate list. The input
// count/list and top-k IDs are the existing validated producer's outputs.
// No selection predicate, floating-point operand, or accumulation changes.
namespace qrt_moe_expert_order {
constexpr unsigned experts = 256u, threads = 256u, maximum_blocks = 1024u;
constexpr size_t maximum_capacity = size_t(16384u) * threads;
constexpr size_t metadata_words = experts + (experts + 1u) + experts;
struct Workspace { uint32_t* storage = nullptr; size_t capacity = 0u; };
inline size_t bytes(size_t capacity) {
    return capacity && capacity <= maximum_capacity ? (capacity + metadata_words) * sizeof(uint32_t) : 0u;
}
struct Views { uint32_t *indices, *counts, *offsets, *cursors; };
inline Views views(Workspace workspace) {
    auto* counts = workspace.storage + workspace.capacity;
    return {workspace.storage, counts, counts + experts, counts + experts + (experts + 1u)};
}
__global__ void histogram(const uint32_t* indices, const uint32_t* count,
    const int32_t* topk_ids, unsigned columns, uint32_t* counts) {
    for (unsigned slot = blockIdx.x * blockDim.x + threadIdx.x; slot < *count;
         slot += gridDim.x * blockDim.x) {
        const unsigned expert = unsigned(topk_ids[indices[slot] / columns]);
        atomicAdd(counts + expert, 1u);
    }
}
__global__ void prefix(const uint32_t* counts, uint32_t* offsets, uint32_t* cursors) {
    if (threadIdx.x) return;
    unsigned total = 0u;
    for (unsigned expert = 0u; expert < experts; ++expert) {
        offsets[expert] = total; cursors[expert] = total; total += counts[expert];
    }
    offsets[experts] = total;
}
__global__ void scatter(const uint32_t* indices, const uint32_t* count,
    const int32_t* topk_ids, unsigned columns, uint32_t* cursors, uint32_t* ordered) {
    for (unsigned slot = blockIdx.x * blockDim.x + threadIdx.x; slot < *count;
         slot += gridDim.x * blockDim.x) {
        const unsigned cell = indices[slot], expert = unsigned(topk_ids[cell / columns]);
        const unsigned destination = atomicAdd(cursors + expert, 1u);
        ordered[destination] = cell;
    }
}
inline hipError_t launch(const uint32_t* indices, const uint32_t* count,
    const int32_t* topk_ids, unsigned columns, Workspace workspace, hipStream_t stream) {
    if (!indices || !count || !topk_ids || !workspace.storage || !bytes(workspace.capacity) ||
        (columns != 512u && columns != 2048u) || indices == workspace.storage)
        return hipErrorInvalidValue;
    const auto view = views(workspace);
    const unsigned requested = unsigned((workspace.capacity + threads - 1u) / threads);
    const unsigned blocks = requested < maximum_blocks ? requested : maximum_blocks;
    auto status = hipMemsetAsync(view.counts, 0, experts * sizeof(uint32_t), stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(histogram, dim3(blocks), dim3(threads), 0u, stream,
        indices, count, topk_ids, columns, view.counts);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(prefix, dim3(1u), dim3(threads), 0u, stream,
        view.counts, view.offsets, view.cursors);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(scatter, dim3(blocks), dim3(threads), 0u, stream,
        indices, count, topk_ids, columns, view.cursors, view.indices);
    return hipGetLastError();
}
} // namespace qrt_moe_expert_order
