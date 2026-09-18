#pragma once
#include "narrow_domain_qk.h"
#include "prepared_decoded_qk_range.h"

// Isolated range form of the qualified narrow QK tile. Only query metadata
// addressing changes; complete original Q/K remain the fallback authority.
namespace qrt_long_narrow_qk {
namespace decoded = qrt_sm121_decoded_bf16;
template<bool Narrow,unsigned QueryCells,unsigned KeyCells,unsigned Window>
__global__ void scores(const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, const unsigned* query_narrow,
    const unsigned* key_narrow,float* output,unsigned* tile_counts,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride, unsigned query_origin) {
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
    // Q staging is not live during classification. Reuse its first word so
    // the K128 tile stays at exactly 32 KiB instead of requiring four more
    // shared bytes. Every thread consumes the decision before staging starts.
    auto* rejected=&qvalues[0][0];
    if(!threadIdx.x)*rejected=0u;
    __syncthreads();
    if(interior){
        const unsigned t=threadIdx.x;
        if(t<rows && !query_narrow[(start+query_tile+t-query_origin)*16u+head])atomicOr(rejected,1u);
        if(t<columns && !key_narrow[(key_tile+t)*2u+kv_head])atomicOr(rejected,1u);
    }
    __syncthreads();
    const bool accepted=interior && !*rejected;
    __syncthreads();
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
                (!query_flags[(start + row - query_origin) * 16u + head] || !key_flags[key * 2u + kv_head]);
        }
    }
    for (unsigned window = 0u; window < 256u; window += window_width) {
        for (unsigned cell = threadIdx.x; cell < rows * window_width; cell += threads) {
            const unsigned r = cell / window_width, c = cell % window_width;
            qvalues[r][c] = query_tile + r < count
                ? packed_query[(size_t(start + query_tile + r - query_origin) * 16u + head) * 256u + window + c]
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
struct Workspace {
    qrt_prepared_decoded_qk_range::Workspace decoded;
    unsigned* query_domain=nullptr;
    unsigned* key_domain=nullptr;
    unsigned* tile_counts=nullptr;
};
inline bool valid(const Workspace& w) {
    return qrt_prepared_decoded_qk_range::valid(w.decoded) &&
        w.query_domain && w.key_domain && w.tile_counts;
}
inline int prepare_domain(const uint16_t* query,const uint16_t* key,
    const Workspace& w,hipStream_t stream) {
    if(!query||!key||!valid(w))return int(hipErrorInvalidValue);
    auto status=hipMemsetAsync(w.tile_counts,0,2u*sizeof(unsigned),stream);
    if(status!=hipSuccess)return int(status);
    const auto& d=w.decoded;
    hipLaunchKernelGGL(qrt_narrow_domain_qk::classify_rows,
        dim3(d.query_count*16u),dim3(256u),0u,stream,
        query+size_t(d.query_start)*4096u,w.query_domain,d.query_count*16u);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(qrt_narrow_domain_qk::classify_rows,
        dim3(d.key_tokens*2u),dim3(256u),0u,stream,key,w.key_domain,d.key_tokens*2u);
    return int(hipGetLastError());
}
inline int launch_workspace(const void* state,const uint16_t* query,
    const uint16_t* transposed_key,float* output,hipStream_t stream,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride) {
    if(!state||!query||!transposed_key||!output)return int(hipErrorInvalidValue);
    const auto& w=*static_cast<const Workspace*>(state);const auto& d=w.decoded;
    if(!valid(w)||!count||count>128u||start<d.query_start||
        start-d.query_start>=d.query_count||count>d.query_count-(start-d.query_start)||
        stride!=start+count||key_stride!=d.key_tokens)return int(hipErrorInvalidValue);
    namespace range=qrt_prepared_decoded_qk_range;
    const auto* q=d.words;const auto* k=q+range::query_words;
    const auto* qflags=k+range::key_words(d.key_capacity);
    const auto* kflags=qflags+range::query_flag_words;
    const dim3 grid((stride+63u)/64u,16u,(count+31u)/32u);
    hipLaunchKernelGGL((scores<true,2u,4u,64u>),grid,dim3(256u),0u,stream,
        q,k,qflags,kflags,w.query_domain,w.key_domain,output,w.tile_counts,
        start,count,stride,key_stride,d.query_start);
    auto status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL((scores<false,2u,4u,64u>),grid,dim3(256u),0u,stream,
        q,k,qflags,kflags,w.query_domain,w.key_domain,output,w.tile_counts,
        start,count,stride,key_stride,d.query_start);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,
        dim3((size_t(count)*16u*stride+255u)/256u),dim3(256u),0u,stream,
        query,transposed_key,output,start,count,stride,key_stride);
    return int(hipGetLastError());
}
} // namespace qrt_long_narrow_qk
