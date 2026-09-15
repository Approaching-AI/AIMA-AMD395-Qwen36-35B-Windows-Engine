#ifndef QRT_COARSE_SELECTIVE_QK_H
#define QRT_COARSE_SELECTIVE_QK_H
#include "selective_qk.h"
#include "../moe_accumulator/sm121_coarse_projection_matrix.h"

// Isolated producer for the strict selective QK component. Zero-C K16 WMMA
// outputs are summed explicitly, with the unchanged coarse projection bound
// accounting for every original K16 carry. Unsupported rows get infinite
// intervals and therefore original score replay. No product dispatcher uses it.
namespace qrt_coarse_selective_qk {
using namespace qrt_blackwell_attention;
namespace bound = qrt_sm121_coarse_projection_bound;

template<unsigned Chunk>
__global__ void native_scores(const uint16_t* query, const uint16_t* transposed_key,
    const unsigned* query_ok, const unsigned* key_ok, float* scores, float* errors,
    unsigned query_start, unsigned query_count, unsigned score_stride, unsigned key_stride) {
    static_assert(Chunk == 64u || Chunk == 128u || Chunk == 256u);
    __shared__ NativeOperandRow left[16], right[kIntegerMatrixColumns];
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u, head = blockIdx.y;
    const unsigned query_tile = blockIdx.z * 16u, key_tile = blockIdx.x * kIntegerMatrixColumns;
    const unsigned kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned last_query = query_start + min(query_tile + 16u, query_count) - 1u;
    MantissaF32x8 accumulator{}, envelope{};
    if (key_tile <= last_query) {
        for (unsigned coarse = 0u; coarse < kHeadDim; coarse += Chunk) {
            MantissaF32x8 partial{}, positive{};
#pragma unroll 1
            for (unsigned base = coarse; base < coarse + Chunk; base += 16u) {
                if (wave == 0u && lane < 16u) {
                    const unsigned row = query_tile + lane;
                    const bool valid = row < query_count && query_ok[(query_start + row) * kQueryHeads + head];
#pragma unroll
                    for (unsigned i = 0u; i < 16u; ++i)
                        left[lane].original[i] = valid
                            ? query[(size_t(query_start + row) * kQueryHeads + head) * kHeadDim + base + i] : 0u;
                }
                if (lane < 16u) {
                    const unsigned row = wave * 16u + lane, key = key_tile + row;
                    const bool valid = key < score_stride && key_ok[key * kKvHeads + kv_head];
#pragma unroll
                    for (unsigned i = 0u; i < 16u; ++i)
                        right[row].original[i] = valid
                            ? transposed_key[(size_t(kv_head) * kHeadDim + base + i) * key_stride + key] : 0u;
                }
                __syncthreads();
                partial += blackwell_native_mma(left[lane % 16u], right[wave * 16u + lane % 16u], MantissaF32x8{});
                positive += blackwell_native_mma<true>(left[lane % 16u], right[wave * 16u + lane % 16u], MantissaF32x8{});
                __syncthreads();
            }
#pragma unroll
            for (unsigned item = 0u; item < 8u; ++item) {
                const auto next = bound::advance<Chunk / 16u>(
                    {accumulator[item], envelope[item]}, partial[item], positive[item]);
                accumulator[item] = next.center; envelope[item] = next.error;
            }
        }
    }
#pragma unroll
    for (unsigned item = 0u; item < 8u; ++item) {
        const unsigned row = query_tile + 2u * item + lane / 16u, key = key_tile + wave * 16u + lane % 16u;
        if (row < query_count && key < score_stride) {
            const size_t cell = (size_t(row) * kQueryHeads + head) * score_stride + key;
            const bool live = key <= query_start + row;
            const bool valid = query_ok[(query_start + row) * kQueryHeads + head] && key_ok[key * kKvHeads + kv_head];
            scores[cell] = live ? accumulator[item] * kExactScale : -INFINITY;
            errors[cell] = !live ? 0.0f : valid
                ? bound::scalar::finish(envelope[item], accumulator[item], kExactScale)
                : bound::scalar::infinity();
        }
    }
}
}
#endif
