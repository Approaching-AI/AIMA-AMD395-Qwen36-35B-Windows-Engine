#ifndef QRT_SM121_PROJECTION_INTERVAL_FILTER_H
#define QRT_SM121_PROJECTION_INTERVAL_FILTER_H
#include <hip/hip_runtime.h>
#include "sm121_projection_interval.h"

// Component-only filter. Original candidate identities enter on the device;
// only a certified common BF16 endpoint replaces one. Every other identity is
// returned for unchanged original K16 replay. No reference tensor is consumed.
namespace qrt_sm121_projection_interval_filter {
namespace interval = qrt_sm121_projection_interval;
namespace bound = qrt_sm121_pv_bound;
using B16 = unsigned short __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));
constexpr unsigned threads = 128u;

__global__ void mark(const unsigned* indices, unsigned count, uint8_t* mask,
    unsigned elements, unsigned* status) {
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    const unsigned cell = indices[slot];
    if (cell >= elements) { atomicOr(status + 1u, 1u); return; }
    mask[cell] = 1u;
}

__global__ void filter(const uint16_t* weights, const uint16_t* inputs,
    uint8_t* mask, float* output, unsigned rows, unsigned tokens, unsigned width) {
    const unsigned lane = threadIdx.x % 32u;
    const unsigned row_tiles = (rows + 15u) / 16u, token_tiles = (tokens + 15u) / 16u;
    const unsigned tile = blockIdx.x * (threads / 32u) + threadIdx.x / 32u;
    if (tile >= row_tiles * token_tiles) return;
    const unsigned row_base = (tile % row_tiles) * 16u, token_base = (tile / row_tiles) * 16u;
    unsigned active = 0u;
    for (unsigned j = 0u; j < 8u; ++j) {
        const unsigned token = token_base + 2u * j + lane / 16u, row = row_base + lane % 16u;
        if (token < tokens && row < rows && mask[size_t(token) * rows + row] == 1u) active |= 1u << j;
    }
    if (!__ballot(active != 0u)) return;
    F8 lower{}, upper{};
#pragma unroll 1
    for (unsigned base = 0u; base < width; base += 16u) {
        B16 a{}, b{}, aa{}, bb{};
        interval::Row ar, br;
        const unsigned token = token_base + lane % 16u, row = row_base + lane % 16u;
#pragma unroll
        for (unsigned k = 0u; k < 16u; ++k) {
            a[k] = token < tokens ? inputs[size_t(token) * width + base + k] : 0u;
            b[k] = row < rows ? weights[size_t(row) * width + base + k] : 0u;
            aa[k] = a[k] & 0x7fffu; bb[k] = b[k] & 0x7fffu;
            interval::include(ar, a[k]); interval::include(br, b[k]);
        }
        const F8 zero{};
        const F8 product = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, zero);
        const F8 absolute = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(aa, bb, zero);
#pragma unroll
        for (unsigned j = 0u; j < 8u; ++j) {
            const unsigned source_lane = 2u * j + lane / 16u;
            // All lanes execute operand shuffles before the candidate mask.
            const interval::Row left{__shfl(ar.minimum, source_lane), __shfl(ar.maximum, source_lane),
                bool(__shfl(int(ar.valid), source_lane))};
            if (active & (1u << j)) {
                const auto next = interval::group({lower[j], upper[j]}, product[j], absolute[j], left, br);
                lower[j] = next.lower; upper[j] = next.upper;
            }
        }
    }
    for (unsigned j = 0u; j < 8u; ++j) if ((active & (1u << j)) && interval::same_bf16({lower[j], upper[j]})) {
        const unsigned token = token_base + 2u * j + lane / 16u, row = row_base + lane % 16u;
        const size_t cell = size_t(token) * rows + row;
        output[cell] = bound::value(uint32_t(bound::bf16(lower[j])) << 16u);
        mask[cell] = 2u;
    }
}

__global__ void compact(const unsigned* indices, unsigned count, const uint8_t* mask,
    unsigned elements, unsigned* remaining, unsigned* status) {
    const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % 32u;
    const unsigned cell = slot < count ? indices[slot] : elements;
    bool replay = false;
    if (slot < count) {
        if (cell >= elements) atomicOr(status + 1u, 1u);
        else {
            const unsigned state = mask[cell];
            if (state != 1u && state != 2u) atomicOr(status + 1u, 2u);
            replay = state == 1u;
        }
    }
    const uint32_t ballot = uint32_t(__ballot(replay));
    unsigned start = 0u;
    if (!lane && ballot) start = atomicAdd(status, unsigned(__popc(ballot)));
    start = __shfl(start, 0u);
    if (replay) remaining[start + unsigned(__popc(ballot & ((1u << lane) - 1u)))] = cell;
}

inline hipError_t launch(const uint16_t* weights, const uint16_t* inputs,
    const unsigned* indices, unsigned count, uint8_t* mask, size_t mask_capacity,
    float* output, unsigned rows, unsigned tokens, unsigned width,
    unsigned* remaining, size_t remaining_capacity, unsigned* status, hipStream_t stream) {
    // The caller supplies unique candidate indices. Out-of-range device
    // identities set status[1] without accessing source or destination memory.
    if (!weights || !inputs || !mask || !output || !remaining || !status || (!indices && count) ||
        !rows || rows > 16384u || !tokens || tokens > 8192u || !width || width > 4096u || width % 16u ||
        count > size_t(rows) * tokens || mask_capacity < size_t(rows) * tokens || remaining_capacity < count)
        return hipErrorInvalidValue;
    const unsigned elements = rows * tokens;
    auto error = hipMemsetAsync(status, 0, 2u * sizeof(unsigned), stream);
    if (error != hipSuccess) return error;
    error = hipMemsetAsync(mask, 0, elements, stream);
    if (error != hipSuccess || !count) return error;
    hipLaunchKernelGGL(mark, dim3((count + 255u) / 256u), dim3(256u), 0u, stream, indices, count, mask, elements, status);
    error = hipGetLastError(); if (error != hipSuccess) return error;
    const unsigned tiles = ((rows + 15u) / 16u) * ((tokens + 15u) / 16u);
    hipLaunchKernelGGL(filter, dim3((tiles + threads / 32u - 1u) / (threads / 32u)), dim3(threads), 0u, stream,
        weights, inputs, mask, output, rows, tokens, width);
    error = hipGetLastError(); if (error != hipSuccess) return error;
    hipLaunchKernelGGL(compact, dim3((count + 255u) / 256u), dim3(256u), 0u, stream,
        indices, count, mask, elements, remaining, status);
    return hipGetLastError();
} // launch
} // namespace qrt_sm121_projection_interval_filter
#endif
