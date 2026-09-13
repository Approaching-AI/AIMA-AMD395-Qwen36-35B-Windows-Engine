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
__global__ void indices_kernel(const uint32_t* indices, uint32_t* ordered,
    uint32_t* counts, const unsigned* weight_flags, const unsigned* input_flags,
    unsigned rows, unsigned count) {
    __shared__ uint32_t local_count[2], first[2];
    if (threadIdx.x < 2u) local_count[threadIdx.x] = 0u;
    __syncthreads();
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned cell = 0u, kind = 0u, offset = 0u;
    if (slot < count) {
        cell = indices[slot];
        kind = (weight_flags[cell % rows] & input_flags[cell / rows] & 1u) ? 0u : 1u;
        offset = atomicAdd(local_count + kind, 1u);
    }
    __syncthreads();
    if (threadIdx.x < 2u) first[threadIdx.x] = atomicAdd(counts + threadIdx.x, local_count[threadIdx.x]);
    __syncthreads();
    if (slot < count) {
        const unsigned target = kind ? count - 1u - (first[kind] + offset) : first[kind] + offset;
        ordered[target] = cell;
    }
}
} // namespace qrt_sm121_replay_partition
#endif
