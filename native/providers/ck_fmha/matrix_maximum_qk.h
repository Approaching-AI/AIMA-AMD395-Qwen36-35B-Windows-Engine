#ifndef QRT_MATRIX_MAXIMUM_QK_H
#define QRT_MATRIX_MAXIMUM_QK_H
#include "prepared_decoded_qk.h"
#include "../moe_accumulator/sm121_matrix_maximum.h"

namespace qrt_matrix_maximum_qk {
using namespace qrt_blackwell_attention;
namespace maximum = qrt_sm121_matrix_maximum;
template<bool Key>
__global__ void prepare(const uint16_t* input, uint32_t* output, unsigned* invalid, unsigned tokens) {
    constexpr unsigned heads = Key ? kKvHeads : kQueryHeads;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(tokens) * heads * kHeadDim) return;
    const unsigned row = unsigned(i / kHeadDim), feature = unsigned(i % kHeadDim);
    const size_t destination = Key ? (size_t(row % heads) * kHeadDim + feature) * tokens + row / heads : i;
    output[destination] = maximum::pack(input[i]);
    if (!qrt_sm121_float_alignment::eligible(input[i])) atomicOr(invalid + row, 1u);
}

template<unsigned Window>
__global__ void scores(const uint16_t* query, const uint16_t* transposed_key,
    const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    static_assert(Window == 64u || Window == 128u || Window == 256u);
    constexpr unsigned rows = 16u, keys = 16u, cells = rows * keys;
    __shared__ int16_t maxima[16u][cells];
    __shared__ uint16_t qvalues[rows][Window], kvalues[Window][keys];
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned query_tile = blockIdx.z * rows, key_tile = blockIdx.x * keys;
    const unsigned qr = threadIdx.x / keys, kc = threadIdx.x % keys;
    const unsigned row = query_tile + qr, key = key_tile + kc;
    const bool live = row < query_count && key < stride;
    const bool active = live && key <= query_start + row;
    const size_t output_cell = (size_t(row) * kQueryHeads + head) * stride + key;
    if (key_tile > query_start + min(query_tile + rows, query_count) - 1u) {
        if (live) output[output_cell] = -INFINITY;
        return;
    }
    // The eight waves cover all sixteen K16 groups before scalar score work.
    // Only recovered integer maxima cross this barrier, not dot/carry values.
    for (unsigned group = wave; group < 16u; group += 8u) {
        const unsigned source = lane % 16u, qrow = query_tile + source, krow = key_tile + source;
        MantissaBf16x16 a{}, b{};
        unsigned qvalid = qrow < query_count, kvalid = krow < stride;
#pragma unroll
        for (unsigned i = 0u; i < 16u; ++i) {
            const uint16_t q = qrow < query_count ? uint16_t(packed_query[
                (size_t(qrow) * kQueryHeads + head) * kHeadDim + group * 16u + i]) : 0u;
            const uint16_t k = krow < stride ? uint16_t(packed_key[
                (size_t(kv_head) * kHeadDim + group * 16u + i) * key_stride + krow]) : 0u;
            qvalid &= unsigned(!(q & maximum::invalid_operand));
            kvalid &= unsigned(!(k & maximum::invalid_operand));
            a[i] = q & maximum::invalid_operand ? 0u : q;
            b[i] = k & maximum::invalid_operand ? 0u : k;
        }
        const MantissaF32x8 zero{};
        const auto product = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, zero);
#pragma unroll
        for (unsigned i = 0u; i < 8u; ++i) {
            const unsigned r = 2u * i + lane / 16u;
            const bool valid = __shfl(qvalid, r, 32u) && kvalid;
            maxima[group][r * keys + source] = int16_t(valid ? maximum::recover(product[i]) : maximum::invalid_maximum);
        }
    }
    __syncthreads();
    bool fallback = active && (query_flags[row * kQueryHeads + head] ||
        key_flags[key * kKvHeads + kv_head]);
    float carry = 0.0f;
    for (unsigned window = 0u; window < kHeadDim; window += Window) {
        for (unsigned i = threadIdx.x; i < rows * Window; i += kThreads) {
            const unsigned r = i / Window, c = i % Window;
            qvalues[r][c] = query_tile + r < query_count ? uint16_t(packed_query[
                (size_t(query_tile + r) * kQueryHeads + head) * kHeadDim + window + c] >> 16u) : 0u;
        }
        for (unsigned i = threadIdx.x; i < Window * keys; i += kThreads) {
            const unsigned r = i / keys, c = i % keys;
            kvalues[r][c] = key_tile + c < stride ? uint16_t(packed_key[
                (size_t(kv_head) * kHeadDim + window + r) * key_stride + key_tile + c] >> 16u) : 0u;
        }
        __syncthreads();
        if (active && !fallback) {
            for (unsigned base = 0u; base < Window; base += 16u) {
                qrt_sm121_float_alignment::Group group;
                const int m = maxima[(window + base) / 16u][threadIdx.x];
                if (m == maximum::invalid_maximum) {
#pragma unroll
                    for (unsigned i = 0u; i < 16u; ++i) group.set(i, qvalues[qr][base + i], kvalues[base + i][kc]);
                } else {
                    group.maximum = m;
                    group.first_negative = ((qvalues[qr][base] ^ kvalues[base][kc]) & 0x8000u) != 0u;
#pragma unroll
                    for (unsigned i = 0u; i < 16u; ++i)
                        group.products[i] = qrt_sm121_decoded_bf16::value(qvalues[qr][base + i]) *
                            qrt_sm121_decoded_bf16::value(kvalues[base + i][kc]);
                }
                float next;
                if (!qrt_sm121_f32_carry::accumulate<0u>(carry, group, &next)) { fallback = true; break; }
                carry = next;
            }
        }
        __syncthreads();
    }
    if (live) output[output_cell] = !active ? -INFINITY : fallback
        ? qrt_decoded_window_qk::raw_dot(query + (size_t(query_start + row) * kQueryHeads + head) * kHeadDim,
            transposed_key + size_t(kv_head) * kHeadDim * key_stride + key, key_stride)
        : carry * kExactScale;
}
}
#endif
