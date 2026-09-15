#ifndef QRT_REPLAY_WEIGHT_BUCKETS_H
#define QRT_REPLAY_WEIGHT_BUCKETS_H
#include <hip/hip_runtime.h>
#include "replay_weight_bucket_plan.h"
namespace qrt_replay_weight_buckets {
constexpr unsigned threads = 256u;
__global__ void count_kernel(const unsigned* indices, unsigned count, Plan plan, unsigned* histogram, unsigned* status) {
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    const unsigned bin = bucket(plan, indices[slot]);
    if (bin == UINT32_MAX) { atomicOr(status, 1u); return; }
    atomicAdd(histogram + bin, 1u);
}
__global__ void prefix_kernel(const unsigned* histogram, unsigned* starts, unsigned* cursors,
    unsigned bins, unsigned count, unsigned* status) {
    __shared__ unsigned partial[threads];
    const unsigned lane = threadIdx.x, chunk = (bins + threads - 1u) / threads;
    const unsigned first = lane * chunk, end = min(first + chunk, bins);
    unsigned sum = 0u;
    for (unsigned bin = first; bin < end; ++bin) sum += histogram[bin];
    partial[lane] = sum; __syncthreads();
    for (unsigned delta = 1u; delta < threads; delta *= 2u) {
        const unsigned add = lane >= delta ? partial[lane - delta] : 0u;
        __syncthreads(); partial[lane] += add; __syncthreads();
    }
    unsigned offset = partial[lane] - sum;
    for (unsigned bin = first; bin < end; ++bin) {
        starts[bin] = cursors[bin] = offset; offset += histogram[bin];
    }
    if (!lane) {
        starts[bins] = partial[threads - 1u];
        if (partial[threads - 1u] != count) atomicOr(status, 2u);
    }
}
__global__ void scatter_kernel(const unsigned* indices, unsigned count, Plan plan, const unsigned* starts,
    unsigned* cursors, unsigned* ordered, unsigned* status) {
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    // The preceding count/prefix kernels finish on this stream. Bad input
    // suppresses all output writes, so a caller can inspect status before use.
    if (slot >= count || *status) return;
    const unsigned cell = indices[slot], bin = bucket(plan, cell);
    if (bin == UINT32_MAX) { atomicOr(status, 4u); return; }
    const unsigned target = atomicAdd(cursors + bin, 1u);
    if (target >= count || target >= starts[bin + 1u]) { atomicOr(status, 8u); return; }
    ordered[target] = cell;
}
inline bool overlaps(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    if (!a_bytes || !b_bytes) return false;
    const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}
// Isolated reordering component. The caller supplies unique selected indices,
// retains buffers through stream completion, and checks the device status
// before consuming ordered values. Bin order is deterministic; order within
// a bin is unspecified. No numerical value, membership or K16 order changes.
inline hipError_t launch(const unsigned* indices, size_t index_words, unsigned count, Plan plan,
    unsigned* ordered, size_t ordered_words, unsigned* workspace, size_t capacity, hipStream_t stream) {
    const size_t required = workspace_words(plan);
    if (!required || !indices || !ordered || !workspace || count > size_t(plan.rows) * plan.tokens ||
        index_words < count || ordered_words < count || capacity < required ||
        overlaps(indices, size_t(count) * 4u, ordered, size_t(count) * 4u) ||
        overlaps(indices, size_t(count) * 4u, workspace, required * 4u) ||
        overlaps(ordered, size_t(count) * 4u, workspace, required * 4u)) return hipErrorInvalidValue;
    const unsigned bins = bucket_count(plan);
    unsigned* histogram = workspace; unsigned* starts = histogram + bins;
    unsigned* cursors = starts + bins + 1u; unsigned* status = cursors + bins;
    auto result = hipMemsetAsync(workspace, 0, required * 4u, stream);
    if (result != hipSuccess) return result;
    if (count) {
        hipLaunchKernelGGL(count_kernel, dim3((count + threads - 1u) / threads), dim3(threads), 0u, stream, indices, count, plan, histogram, status);
        result = hipGetLastError(); if (result != hipSuccess) return result;
    }
    hipLaunchKernelGGL(prefix_kernel, dim3(1u), dim3(threads), 0u, stream, histogram, starts, cursors, bins, count, status);
    result = hipGetLastError(); if (result != hipSuccess) return result;
    if (count) {
        hipLaunchKernelGGL(scatter_kernel, dim3((count + threads - 1u) / threads), dim3(threads), 0u, stream, indices, count, plan, starts, cursors, ordered, status);
        result = hipGetLastError();
    }
    return result;
}
}
#endif
