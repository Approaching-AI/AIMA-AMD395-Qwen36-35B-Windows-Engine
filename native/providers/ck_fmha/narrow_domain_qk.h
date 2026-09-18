#pragma once
#include "deferred_qk_fallback.h"
#include "../moe_accumulator/sm121_narrow_f32_carry.h"

// Isolated complete-tile domain split. Narrow blocks use a proved K256
// specialization without per-group exception checks. Other blocks preserve
// original active-cell checks, K16 carries and complete-dot fallback.
namespace qrt_narrow_domain_qk {
namespace decoded = qrt_sm121_decoded_bf16;
__global__ void classify_rows(const uint16_t* values,unsigned* accepted,unsigned rows){
    const unsigned row=blockIdx.x,feature=threadIdx.x;
    if(row>=rows)return;
    __shared__ unsigned invalid;
    if(!feature)invalid=0u;
    __syncthreads();
    if(!qrt_sm121_narrow_f32_carry::eligible(values[size_t(row)*256u+feature]))atomicOr(&invalid,1u);
    __syncthreads();
    if(!feature)accepted[row]=invalid==0u;
}
template<bool Narrow,unsigned QueryCells,unsigned KeyCells,unsigned Window>
__global__ void scores(const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, const unsigned* query_narrow,
    const unsigned* key_narrow,float* output,unsigned* tile_counts,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    static_assert((QueryCells == 2u || QueryCells == 4u) && (KeyCells == 2u || KeyCells == 4u));
    static_assert(Window == 64u || Window == 128u);
    constexpr unsigned window_width = Window, rows = 16u * QueryCells;
    constexpr unsigned columns = 16u * KeyCells, threads = 256u;
    __shared__ uint32_t qvalues[rows][window_width], kvalues[window_width][columns];
    const unsigned head = blockIdx.y, kv_head = head / 8u;
    const unsigned query_tile = blockIdx.z * rows, key_tile = blockIdx.x * columns;
    const unsigned qr = threadIdx.x / 16u, kc = threadIdx.x % 16u;
    const unsigned last_query = start + min(query_tile + rows, count) - 1u;
    const bool interior=query_tile+rows<=count && key_tile+columns<=stride &&
        key_tile+columns-1u<=start+query_tile;
    __shared__ unsigned rejected;
    if(!threadIdx.x)rejected=0u;
    __syncthreads();
    if(interior){
        const unsigned t=threadIdx.x;
        if(t<rows && !query_narrow[(start+query_tile+t)*16u+head])atomicOr(&rejected,1u);
        if(t<columns && !key_narrow[(key_tile+t)*2u+kv_head])atomicOr(&rejected,1u);
    }
    __syncthreads();
    const bool accepted=interior && !rejected;
    if(Narrow!=accepted)return;
    if(!threadIdx.x && tile_counts)atomicAdd(tile_counts+(Narrow?1u:0u),1u);
    if (key_tile > last_query) {
#pragma unroll
        for (unsigned q = 0u; q < QueryCells; ++q) {
#pragma unroll
            for (unsigned k = 0u; k < KeyCells; ++k) {
                const unsigned row = query_tile + qr + q * 16u, key = key_tile + kc + k * 16u;
                if (row < count && key < stride)
                    output[(size_t(row) * 16u + head) * stride + key] = -INFINITY;
            }
        }
        return;
    }
    float carry[QueryCells][KeyCells]{};
    bool active[QueryCells][KeyCells], fallback[QueryCells][KeyCells];
#pragma unroll
    for (unsigned q = 0u; q < QueryCells; ++q) {
#pragma unroll
        for (unsigned k = 0u; k < KeyCells; ++k) {
            const unsigned row = query_tile + qr + q * 16u, key = key_tile + kc + k * 16u;
            active[q][k] = Narrow || (row < count && key < stride && key <= start + row);
            fallback[q][k] = !Narrow && active[q][k] &&
                (!query_flags[(start + row) * 16u + head] || !key_flags[key * 2u + kv_head]);
        }
    }
    for (unsigned window = 0u; window < 256u; window += window_width) {
        for (unsigned cell = threadIdx.x; cell < rows * window_width; cell += threads) {
            const unsigned r = cell / window_width, c = cell % window_width;
            qvalues[r][c] = query_tile + r < count
                ? packed_query[(size_t(start + query_tile + r) * 16u + head) * 256u + window + c]
                : decoded::pack(0u);
        }
        for (unsigned cell = threadIdx.x; cell < window_width * columns; cell += threads) {
            const unsigned r = cell / columns, c = cell % columns;
            kvalues[r][c] = key_tile + c < stride
                ? packed_key[(size_t(kv_head) * 256u + window + r) * key_stride + key_tile + c]
                : decoded::pack(0u);
        }
        __syncthreads();
        for (unsigned base = 0u; base < window_width; base += 16u) {
#pragma unroll
            for (unsigned q = 0u; q < QueryCells; ++q) {
                bool any = false;
#pragma unroll
                for (unsigned k = 0u; k < KeyCells; ++k) any |= active[q][k] && !fallback[q][k];
                if (any) {
                    qrt_sm121_float_alignment::Group groups[KeyCells];
#pragma unroll
                    for (unsigned i = 0u; i < 16u; ++i) {
                        const uint32_t left = qvalues[qr + q * 16u][base + i];
#pragma unroll
                        for (unsigned k = 0u; k < KeyCells; ++k)
                            decoded::set_packed(groups[k], i, left, kvalues[base + i][kc + k * 16u]);
                    }
#pragma unroll
                    for (unsigned k = 0u; k < KeyCells; ++k) if (active[q][k] && !fallback[q][k]) {
                        if constexpr(Narrow)carry[q][k]=qrt_sm121_narrow_f32_carry::accumulate(carry[q][k],groups[k]);
                        else {
                            float next;
                            if(qrt_sm121_f32_carry::accumulate<0u>(carry[q][k],groups[k],&next))carry[q][k]=next;
                            else fallback[q][k]=true;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned q = 0u; q < QueryCells; ++q) {
#pragma unroll
        for (unsigned k = 0u; k < KeyCells; ++k) {
            const unsigned row = query_tile + qr + q * 16u, key = key_tile + kc + k * 16u;
            if (Narrow || (row < count && key < stride))
                output[(size_t(row) * 16u + head) * stride + key] = !active[q][k] ? -INFINITY
                    : fallback[q][k] ? qrt_sm121_float_alignment::from_bits(qrt_deferred_qk_fallback::deferred_bits)
                    : carry[q][k] * qrt_blackwell_attention::kExactScale;
        }
    }
}
} // namespace qrt_narrow_domain_qk
