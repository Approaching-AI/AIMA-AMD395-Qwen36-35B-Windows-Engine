#pragma once
#include "wave_matrix_qk.h"
#include "../moe_accumulator/sm121_partial_matrix_group.h"

// Isolated wave-owned QK with individually lossless coefficients. Only the
// exceptional original products use scalar alignment; every group still has
// its original ordered carry and complete narrow fallback.
namespace qrt_partial_wave_matrix_qk {
namespace partial = qrt_sm121_partial_matrix_group;
namespace group = qrt_sm121_compact_matrix_group;
namespace original = qrt_compact_matrix_queue_qk;
using Row = partial::Row;
using I4 = original::I4;
using I8 = original::I8;
using F8 = qrt_wave_matrix_qk::F8;
struct Workspace { qrt_narrow_domain_qk::Workspace original; const Row *query, *key; };

template<bool Key>
__global__ void prepare(const uint16_t* input, Row* output, unsigned tokens) {
    constexpr unsigned heads = Key ? 2u : 16u;
    const size_t item = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (item >= size_t(tokens)*heads*16u) return;
    const unsigned row = unsigned(item/16u), g = unsigned(item%16u);
    uint16_t raw[16];
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) raw[i] = input[item*16u+i];
    output[Key ? (size_t(row%heads)*16u+g)*tokens+row/heads : item] = partial::prepare(raw);
}

__device__ __forceinline__ Row broadcast(const Row& row, unsigned source) {
    Row result;
    result.common.encoded.control = __shfl(row.common.encoded.control,source);
    result.exceptions = __shfl(row.exceptions,source);
#pragma unroll
    for (unsigned i = 0u; i < 4u; ++i)
        result.common.encoded.exponents[i] = __shfl(row.common.encoded.exponents[i],source);
#pragma unroll
    for (unsigned i = 0u; i < 2u; ++i)
        result.common.trailing[i] = __shfl(row.common.trailing[i],source);
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i)
        result.common.encoded.pairs[i] = __shfl(row.common.encoded.pairs[i],source);
    return result;
}
__device__ __forceinline__ void digits(const Row& row, I4& high, I4& low) {
#pragma unroll
    for (unsigned i = 0u; i < 4u; ++i) {
        const uint32_t a = partial::coefficient_pair(row,2u*i);
        const uint32_t b = partial::coefficient_pair(row,2u*i+1u);
        low[i] = int((a & 0x00ff00ffu) | ((b & 0x00ff00ffu) << 8u));
        high[i] = int(((a >> 8u) & 0x00ff00ffu) | (b & 0xff00ff00u));
    }
}

template<bool ForceOriginal = false>
__global__ __launch_bounds__(128) void scores(Workspace w, float* output,
    unsigned start, unsigned count, unsigned stride) {
    const unsigned tid = threadIdx.x, lane = tid%32u, wave = tid/32u;
    const unsigned head = blockIdx.y, kv = head/8u;
    const unsigned qt = blockIdx.z*32u, kt = blockIdx.x*32u, tokens = w.original.tokens;
    const bool interior = qt+32u <= count && kt+32u <= stride && kt+31u <= start+qt;
    if (!interior) return;
    __shared__ unsigned admitted;
    if (wave == 0u) {
        const bool bad = !w.original.query_domain[(start+qt+lane)*16u+head] ||
            !w.original.key_domain[(kt+lane)*2u+kv];
        const unsigned mask = __ballot(bad);
        if (!lane) admitted = mask == 0u;
    }
    __syncthreads();
    if (!admitted) return;
    if (!tid) atomicAdd(w.original.tile_counts+1u,1u);
    const unsigned query_row = qt+(wave/2u)*16u+lane%16u;
    const unsigned key_base = kt+(wave%2u)*16u;
    F8 carries{};
#pragma unroll 1
    for (unsigned g = 0u; g < 16u; ++g) {
        const Row a = w.query[(size_t(start+query_row)*16u+head)*16u+g];
        const Row b = w.key[(size_t(kv)*16u+g)*tokens+key_base+lane%16u];
        if constexpr (ForceOriginal) {
            // This safety path needs only the original words. Expand once
            // per input row, retaining the established raw-row shuffle owner.
            uint32_t a_raw[8], b_raw[8];
#pragma unroll
            for (unsigned pair = 0u; pair < 8u; ++pair) {
                a_raw[pair] = uint32_t(partial::original(a,pair*2u)) |
                    (uint32_t(partial::original(a,pair*2u+1u)) << 16u);
                b_raw[pair] = uint32_t(partial::original(b,pair*2u)) |
                    (uint32_t(partial::original(b,pair*2u+1u)) << 16u);
            }
#pragma unroll
            for (unsigned item = 0u; item < 8u; ++item) {
                uint32_t key_raw[8];
#pragma unroll
                for (unsigned pair = 0u; pair < 8u; ++pair)
                    key_raw[pair] = __shfl(b_raw[pair],item*2u+lane/16u);
                carries[item] = original::original_group(carries[item],a_raw,key_raw);
            }
            continue;
        }
        I8 hh{}, hl{}, lh{}, ll{};
        if constexpr (!ForceOriginal) {
            I4 ah{}, al{}, bh{}, bl{};
            digits(a,ah,al); digits(b,bh,bl);
            const I8 zero{};
            hh = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,bh,true,ah,zero,false);
            hl = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,bh,false,al,zero,false);
            lh = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,bl,true,ah,zero,false);
            ll = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,bl,false,al,zero,false);
        }
#pragma unroll
        for (unsigned item = 0u; item < 8u; ++item) {
            // Every lane supplies every source before any sparse correction
            // or fallback diverges. Sources always belong to this wave.
            const Row key = broadcast(b,item*2u+lane/16u);
            bool accepted = false;
            if constexpr (!ForceOriginal) {
                const int64_t mathematical = int64_t(hh[item])*65536 +
                    (int64_t(hl[item])+lh[item])*256 + ll[item];
                float updated;
                accepted = partial::accumulate(carries[item],a,key,mathematical,&updated);
                if (accepted) carries[item] = updated;
            }
            if (!accepted) {
                uint32_t a_raw[8], b_raw[8];
#pragma unroll
                for (unsigned pair = 0u; pair < 8u; ++pair) {
                    a_raw[pair] = uint32_t(partial::original(a,pair*2u)) |
                        (uint32_t(partial::original(a,pair*2u+1u)) << 16u);
                    b_raw[pair] = uint32_t(partial::original(key,pair*2u)) |
                        (uint32_t(partial::original(key,pair*2u+1u)) << 16u);
                }
                carries[item] = original::original_group(carries[item],a_raw,b_raw);
            }
        }
    }
#pragma unroll
    for (unsigned item = 0u; item < 8u; ++item)
        output[(size_t(query_row)*16u+head)*stride+key_base+item*2u+lane/16u] =
            carries[item]*qrt_blackwell_attention::kExactScale;
}

template<bool ForceOriginal = false>
inline int launch(const void* state, const uint16_t* query,
    const uint16_t* transposed_key, float* output, hipStream_t stream,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    if (!state || !query || !transposed_key || !output) return int(hipErrorInvalidValue);
    const auto& w = *static_cast<const Workspace*>(state);
    const auto& o = w.original;
    if (!qrt_narrow_domain_qk::valid(o) || !w.query || !w.key || !count || count > 128u ||
        start >= o.tokens || count > o.tokens-start || stride != start+count || key_stride != o.tokens)
        return int(hipErrorInvalidValue);
    const dim3 grid((stride+31u)/32u,16u,(count+31u)/32u);
    hipLaunchKernelGGL((scores<ForceOriginal>),grid,dim3(128u),0u,stream,w,output,start,count,stride);
    auto status = hipGetLastError();
    if (status != hipSuccess) return int(status);
    hipLaunchKernelGGL((qrt_narrow_domain_qk::scores<false,2u,2u,64u>),grid,dim3(256u),0u,stream,
        o.query,o.key,o.query_flags,o.key_flags,o.query_domain,o.key_domain,output,o.tile_counts,
        start,count,stride,key_stride);
    status = hipGetLastError();
    if (status != hipSuccess) return int(status);
    hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,
        dim3((size_t(count)*16u*stride+255u)/256u),dim3(256u),0u,stream,
        query,transposed_key,output,start,count,stride,key_stride);
    return int(hipGetLastError());
}
} // namespace qrt_partial_wave_matrix_qk
