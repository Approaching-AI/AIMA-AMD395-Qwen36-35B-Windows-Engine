#ifndef QRT_DEFERRED_QK_FALLBACK_H
#define QRT_DEFERRED_QK_FALLBACK_H
#include "prepared_decoded_qk.h"

// Isolated scheduling replacement. The fast kernel uses unchanged prepared
// products and ordered FP32 carries, with no call to the extended integer dot.
// Rejected cells receive a NaN marker that cannot be a successful fast result.
// A second kernel replaces every marker with the original complete dot.
namespace qrt_deferred_qk_fallback {
namespace decoded = qrt_sm121_decoded_bf16;
constexpr uint32_t deferred_bits = 0x7ffffffeu;
template<bool TileFlags>
__global__ void scores(const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride,
    unsigned* tile_flags) {
    constexpr unsigned Window = 128u, Rows = 16u, Keys = 16u, prepared_query_start = 0u;
    static_assert(Window && Window % 16u == 0u && qrt_blackwell_attention::kHeadDim % Window == 0u);
    static_assert(Rows * Keys == qrt_blackwell_attention::kThreads);
    constexpr unsigned rows = Rows, keys = Keys;
    __shared__ uint32_t qvalues[rows][Window], kvalues[Window][keys];
    const unsigned head = blockIdx.y, kv_head = head / (qrt_blackwell_attention::kQueryHeads / qrt_blackwell_attention::kKvHeads);
    const unsigned query_tile = blockIdx.z * rows, key_tile = blockIdx.x * keys;
    const unsigned qr = threadIdx.x / keys, kc = threadIdx.x % keys;
    const unsigned row = query_tile + qr, key = key_tile + kc;
    const unsigned packed_start = query_start - prepared_query_start;
    const bool live = row < query_count && key < stride;
    const bool active = live && key <= query_start + row;
    const size_t output_cell = (size_t(row) * qrt_blackwell_attention::kQueryHeads + head) * stride + key;
    const unsigned last_query = query_start + min(query_tile + rows, query_count) - 1u;
    if (key_tile > last_query) {
        if (live) output[output_cell] = -INFINITY;
        return;
    }
    bool fallback = active && (!query_flags[(packed_start + row) * qrt_blackwell_attention::kQueryHeads + head] ||
        !key_flags[key * qrt_blackwell_attention::kKvHeads + kv_head]);
    float float_carry = 0.0f;
    for (unsigned window = 0u; window < qrt_blackwell_attention::kHeadDim; window += Window) {
        for (unsigned cell = threadIdx.x; cell < rows * Window; cell += qrt_blackwell_attention::kThreads) {
            const unsigned r = cell / Window, c = cell % Window;
            qvalues[r][c] = query_tile + r < query_count
                ? packed_query[(size_t(packed_start + query_tile + r) * qrt_blackwell_attention::kQueryHeads + head) * qrt_blackwell_attention::kHeadDim + window + c]
                : decoded::pack(0u);
        }
        for (unsigned cell = threadIdx.x; cell < Window * keys; cell += qrt_blackwell_attention::kThreads) {
            const unsigned r = cell / keys, c = cell % keys;
            kvalues[r][c] = key_tile + c < stride
                ? packed_key[(size_t(kv_head) * qrt_blackwell_attention::kHeadDim + window + r) * key_stride + key_tile + c]
                : decoded::pack(0u);
        }
        __syncthreads();
        if (active && !fallback) {
            for (unsigned base = 0u; base < Window; base += 16u) {
                qrt_sm121_float_alignment::Group group;
#pragma unroll
                for (unsigned i = 0u; i < 16u; ++i)
                    decoded::set_packed(group, i, qvalues[qr][base + i], kvalues[base + i][kc]);
                float next;
                if (!qrt_sm121_f32_carry::accumulate<0u>(float_carry, group, &next)) { fallback = true; break; }
                float_carry = next;
            }
        }
        __syncthreads();
    }
    const bool deferred = active && fallback;
    if constexpr (TileFlags) {
        const unsigned mask = __ballot(deferred);
        if (threadIdx.x % 32u == 0u && mask) {
            const size_t tile = (size_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
            atomicExch(tile_flags + tile, 1u);
        }
    }
    if (live) output[output_cell] = !active ? -INFINITY : deferred
        ? qrt_sm121_float_alignment::from_bits(deferred_bits) : float_carry * qrt_blackwell_attention::kExactScale;
}

__device__ __forceinline__ void replay_cell(const uint16_t* query, const uint16_t* key,
    float* output, size_t cell, unsigned start, unsigned stride, unsigned key_stride,
    unsigned query_origin = 0u) {
    if (qrt_sm121_f32_carry::bits(output[cell]) != deferred_bits) return;
    const unsigned row = unsigned(cell / stride), column = unsigned(cell % stride);
    const unsigned head = row % qrt_blackwell_attention::kQueryHeads;
    const unsigned kv = head / (qrt_blackwell_attention::kQueryHeads / qrt_blackwell_attention::kKvHeads);
    output[cell] = qrt_decoded_window_qk::raw_dot(
        query + (size_t(start + row / qrt_blackwell_attention::kQueryHeads - query_origin) * qrt_blackwell_attention::kQueryHeads + head) * qrt_blackwell_attention::kHeadDim,
        key + size_t(kv) * qrt_blackwell_attention::kHeadDim * key_stride + column, key_stride);
}
__global__ void replay_scan(const uint16_t* query, const uint16_t* key, float* output,
    unsigned start, unsigned queries, unsigned stride, unsigned key_stride) {
    const size_t cell = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell < size_t(queries) * qrt_blackwell_attention::kQueryHeads * stride)
        replay_cell(query, key, output, cell, start, stride, key_stride);
}
__global__ void replay_scan_from_query_origin(const uint16_t* query, const uint16_t* key,
    float* output, unsigned start, unsigned queries, unsigned stride, unsigned key_stride,
    unsigned query_origin) {
    const size_t cell = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell < size_t(queries) * qrt_blackwell_attention::kQueryHeads * stride)
        replay_cell(query,key,output,cell,start,stride,key_stride,query_origin);
}
__global__ void replay_tiles(const uint16_t* query, const uint16_t* key, float* output,
    const unsigned* flags, unsigned start, unsigned queries, unsigned stride, unsigned key_stride) {
    const size_t tile = (size_t(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    if (!flags[tile]) return;
    const unsigned row = blockIdx.z * 16u + threadIdx.x / 16u;
    const unsigned column = blockIdx.x * 16u + threadIdx.x % 16u;
    if (row < queries && column < stride)
        replay_cell(query, key, output, (size_t(row) * qrt_blackwell_attention::kQueryHeads + blockIdx.y) * stride + column,
            start, stride, key_stride);
}
}
#endif
