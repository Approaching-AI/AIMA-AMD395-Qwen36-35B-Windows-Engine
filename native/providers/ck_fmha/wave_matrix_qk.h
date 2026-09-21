#pragma once
#include "compact_matrix_queue_qk.h"
#include "../moe_accumulator/sm121_compact_matrix_metadata.h"
#include "../moe_accumulator/sm121_matrix_remainder_group.h"

// Isolated matrix-QK replacement. Each wave keeps its complete16x16 output
// tile and eight ordered carries per lane. Matrix results never enter LDS;
// only the one-time complete32x32 domain decision uses a CTA barrier.
namespace qrt_wave_matrix_qk {
namespace original = qrt_compact_matrix_queue_qk;
namespace group = qrt_sm121_compact_matrix_group;
namespace metadata = qrt_sm121_compact_matrix_metadata;
using Workspace = original::Workspace;
using I4 = original::I4;
using I8 = original::I8;
using F8 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ metadata::Row broadcast(const metadata::Row& row,
    unsigned source) {
    metadata::Row result;
    result.control = __shfl(row.control, source);
#pragma unroll
    for (unsigned i = 0u; i < 4u; ++i)
        result.exponents[i] = __shfl(row.exponents[i], source);
    result.trailing[0] = __shfl(row.trailing[0], source);
    result.trailing[1] = __shfl(row.trailing[1], source);
    result.first_pair = __shfl(row.first_pair, source);
    return result;
}

__device__ __forceinline__ void digits(const group::Row& row, I4& high, I4& low) {
#pragma unroll
    for (unsigned i = 0u; i < 4u; ++i) {
        const uint32_t a = row.encoded.pairs[2u*i];
        const uint32_t b = row.encoded.pairs[2u*i+1u];
        low[i] = int((a & 0x00ff00ffu) | ((b & 0x00ff00ffu) << 8u));
        high[i] = int(((a >> 8u) & 0x00ff00ffu) | (b & 0xff00ff00u));
    }
}

template<bool ForceOriginal = false, bool CorrectRemainder = false>
__global__ __launch_bounds__(128) void scores(Workspace w, float* output,
    unsigned start, unsigned count, unsigned stride) {
    const unsigned tid = threadIdx.x, lane = tid % 32u, wave = tid / 32u;
    const unsigned head = blockIdx.y, kv = head / 8u;
    const unsigned qt = blockIdx.z * 32u, kt = blockIdx.x * 32u;
    const unsigned tokens = w.original.tokens;
    const bool interior = qt + 32u <= count && kt + 32u <= stride &&
        kt + 31u <= start + qt;
    if (!interior) return;

    __shared__ unsigned admitted;
    if (wave == 0u) {
        const bool bad = !w.original.query_domain[(start + qt + lane)*16u + head] ||
            !w.original.key_domain[(kt + lane)*2u + kv];
        const unsigned mask = __ballot(bad);
        if (!lane) admitted = mask == 0u;
    }
    __syncthreads();
    if (!admitted) return;
    if (!tid) atomicAdd(w.original.tile_counts + 1u, 1u);

    const unsigned query_row = qt + (wave / 2u)*16u + lane % 16u;
    const unsigned key_base = kt + (wave % 2u)*16u;
    F8 carries{};
#pragma unroll 1
    for (unsigned g = 0u; g < 16u; ++g) {
        const group::Row a = w.query[(size_t(start + query_row)*16u + head)*16u + g];
        const group::Row b = w.key[(size_t(kv)*16u + g)*tokens + key_base + lane % 16u];
        const auto a_metadata = metadata::prepare(a);
        const auto b_metadata = metadata::prepare(b);
        I8 hh{}, hl{}, lh{}, ll{};
        if constexpr (!ForceOriginal) {
            I4 ah{}, al{}, bh{}, bl{};
            digits(a, ah, al);
            digits(b, bh, bl);
            const I8 zero{};
            // Transpose the mathematical product tile: this lane retains its
            // own query row and broadcasts only the eight needed key rows.
            hh = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,bh,true,ah,zero,false);
            hl = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,bh,false,al,zero,false);
            lh = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,bl,true,ah,zero,false);
            ll = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,bl,false,al,zero,false);
        }

        uint32_t a_raw[8], b_raw[8];
#pragma unroll
        for (unsigned pair = 0u; pair < 8u; ++pair) {
            a_raw[pair] = uint32_t(group::compact::original(a.encoded, pair*2u)) |
                (uint32_t(group::compact::original(a.encoded, pair*2u+1u)) << 16u);
            if constexpr (!CorrectRemainder || ForceOriginal)
                b_raw[pair] = uint32_t(group::compact::original(b.encoded, pair*2u)) |
                    (uint32_t(group::compact::original(b.encoded, pair*2u+1u)) << 16u);
        }
#pragma unroll
        for (unsigned item = 0u; item < 8u; ++item) {
            const unsigned source = item*2u + lane/16u;
            bool accepted = false;
            if constexpr (!ForceOriginal) {
                const auto key_metadata = broadcast(b_metadata, source);
                const int64_t mathematical = int64_t(hh[item])*65536 +
                    (int64_t(hl[item]) + lh[item])*256 + ll[item];
                float updated;
                if constexpr (CorrectRemainder) {
                    auto key_group = metadata::expand(key_metadata);
#pragma unroll
                    for (unsigned pair = 0u; pair < 8u; ++pair)
                        key_group.encoded.pairs[pair] = __shfl(b.encoded.pairs[pair], source);
                    accepted = qrt_sm121_matrix_remainder_group::accumulate(
                        carries[item], a, key_group, mathematical, &updated);
                    if (accepted) carries[item] = updated;
                    else {
                        // All key shuffles above are uniform. Only the local
                        // fallback consumers expand their original BF16 row.
                        uint32_t key_raw[8];
#pragma unroll
                        for (unsigned pair = 0u; pair < 8u; ++pair)
                            key_raw[pair] = uint32_t(group::compact::original(key_group.encoded, pair*2u)) |
                                (uint32_t(group::compact::original(key_group.encoded, pair*2u+1u)) << 16u);
                        carries[item] = original::original_group(carries[item], a_raw, key_raw);
                    }
                } else {
                    accepted = metadata::accumulate(carries[item], a_metadata,
                        key_metadata, mathematical, &updated);
                    if (accepted) carries[item] = updated;
                }
            }
            // All source lanes participate in every shuffle when any output
            // in the wave rejects. Divergent consumers cannot read inactive
            // source lanes. Rejection keeps the incoming carry untouched.
            if constexpr (!CorrectRemainder || ForceOriginal) if (__ballot(!accepted)) {
                uint32_t key_raw[8];
#pragma unroll
                for (unsigned pair = 0u; pair < 8u; ++pair)
                    key_raw[pair] = __shfl(b_raw[pair], source);
                if (!accepted)
                    carries[item] = original::original_group(carries[item], a_raw, key_raw);
            }
        }
    }
#pragma unroll
    for (unsigned item = 0u; item < 8u; ++item) {
        const unsigned key = key_base + item*2u + lane/16u;
        output[(size_t(query_row)*16u + head)*stride + key] =
            carries[item] * qrt_blackwell_attention::kExactScale;
    }
}

template<bool ForceOriginal = false, bool CorrectRemainder = false>
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
    hipLaunchKernelGGL((scores<ForceOriginal,CorrectRemainder>),grid,dim3(128u),0u,stream,w,output,start,count,stride);
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
} // namespace qrt_wave_matrix_qk
