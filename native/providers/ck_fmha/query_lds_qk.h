#pragma once
#include "deferred_qk_fallback.h"
#include "query_lds_layout.h"

// Component-only shared query layouts. Preparation, key storage, K16 order,
// arithmetic, sentinels and complete deferred original replay are unchanged.
namespace qrt_query_lds_qk {
namespace decoded=qrt_sm121_decoded_bf16;
template<unsigned Layout>
__global__ void scores(const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    constexpr unsigned Window = 128u, Rows = 16u, Keys = 16u, prepared_query_start = 0u;
    static_assert(Window && Window % 16u == 0u && qrt_blackwell_attention::kHeadDim % Window == 0u);
    static_assert(Rows * Keys == qrt_blackwell_attention::kThreads);
    constexpr unsigned rows = Rows, keys = Keys;
    static_assert(rows==qrt_query_lds_layout::rows && Window==qrt_query_lds_layout::width);
    __shared__ uint32_t qvalues[qrt_query_lds_layout::words<Layout>()], kvalues[Window][keys];
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
            qvalues[qrt_query_lds_layout::index<Layout>(r,c)] = query_tile + r < query_count
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
                    decoded::set_packed(group, i, qvalues[qrt_query_lds_layout::index<Layout>(qr,base+i)], kvalues[base + i][kc]);
                float next;
                if (!qrt_sm121_f32_carry::accumulate<0u>(float_carry, group, &next)) { fallback = true; break; }
                float_carry = next;
            }
        }
        __syncthreads();
    }
    const bool deferred = active && fallback;
    if (live) output[output_cell] = !active ? -INFINITY : deferred
        ? qrt_sm121_float_alignment::from_bits(qrt_deferred_qk_fallback::deferred_bits) : float_carry * qrt_blackwell_attention::kExactScale;
}

}
