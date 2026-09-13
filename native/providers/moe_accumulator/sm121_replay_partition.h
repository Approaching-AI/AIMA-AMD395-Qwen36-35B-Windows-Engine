#ifndef QRT_SM121_REPLAY_PARTITION_H
#define QRT_SM121_REPLAY_PARTITION_H
#include <hip/hip_runtime.h>
#include <cstdint>

namespace qrt_sm121_replay_partition {
// Partition already selected, unique cells without changing selection or dot
// arithmetic. The two counts start at zero and the output has count slots.
// Floating cells reserve from the front, integer cells from the back. Their
// total never exceeds count, so the independently reserved ranges are disjoint.
// Per-CTA aggregation uses only two global atomics, including partial CTAs.
// A collector may have spare capacity between the two classes. Its replay
// maps virtual integer slots to capacity-1-slot, leaving that gap unread.
// Every thread in the CTA participates, including threads without a candidate.
__device__ __forceinline__ void append_block(uint32_t cell, bool valid,
    bool floating, uint32_t* ordered, uint32_t* counts, unsigned capacity) {
    __shared__ uint32_t local_count[2], first[2];
    if (threadIdx.x < 2u) local_count[threadIdx.x] = 0u;
    __syncthreads();
    const unsigned kind = floating ? 0u : 1u;
    unsigned offset = 0u;
    if (valid) {
        offset = atomicAdd(local_count + kind, 1u);
    }
    __syncthreads();
    if (threadIdx.x < 2u) first[threadIdx.x] = atomicAdd(counts + threadIdx.x, local_count[threadIdx.x]);
    __syncthreads();
    if (valid) {
        const unsigned target = kind ? capacity - 1u - (first[kind] + offset) : first[kind] + offset;
        ordered[target] = cell;
    }
}

__global__ void indices_kernel(const uint32_t* indices, uint32_t* ordered,
    uint32_t* counts, const unsigned* weight_flags, const unsigned* input_flags,
    unsigned rows, unsigned count) {
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = slot < count;
    const unsigned cell = valid ? indices[slot] : 0u;
    const bool floating = valid && (weight_flags[cell % rows] & input_flags[cell / rows] & 1u);
    append_block(cell, valid, floating, ordered, counts, count);
}
} // namespace qrt_sm121_replay_partition
#endif
