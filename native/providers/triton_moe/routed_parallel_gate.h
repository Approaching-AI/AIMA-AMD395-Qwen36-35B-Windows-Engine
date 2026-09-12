#ifndef QRT_ROUTED_PARALLEL_GATE_H
#define QRT_ROUTED_PARALLEL_GATE_H
#include <cstdint>
#include <cstddef>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define QRT_GATE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_GATE_INLINE inline
#endif

namespace qrt_routed_parallel_gate {
constexpr unsigned kM = 64, kN = 64, kK = 64, kStride = kK + 8;
constexpr unsigned kHidden = 2048, kIntermediate = 512, kExperts = 256;
constexpr unsigned kTopK = 8, kMaximumRoutes = 8192 * kTopK;
constexpr unsigned kColumnBlocks = 2 * kIntermediate / kN;
constexpr size_t kProjectionStride = size_t(kMaximumRoutes) * kIntermediate;

// The original layout merges an expert's final M64 and <=M32 blocks. This
// schedule visits both descriptors independently and keeps their real rows.
QRT_GATE_INLINE int expert_from_descriptor(int encoded) {
    if (encoded >= 0) return encoded < int(kExperts) ? encoded : -1;
    if (encoded < -2 * int(kExperts)) return -1;
    return encoded < -int(kExperts) ? -encoded - int(kExperts) - 1 : -encoded - 1;
}
struct Cell { unsigned row, column; };
QRT_GATE_INLINE Cell cell(unsigned wave, unsigned lane, unsigned element, unsigned fragment) {
    return {(wave % 4) * 16 + 2 * element + lane / 16,
            (wave / 4) * 32 + fragment * 16 + lane % 16};
}

#if defined(__HIPCC__)
using Bf16x16 = uint16_t __attribute__((ext_vector_type(16)));
using F32x8 = float __attribute__((ext_vector_type(8)));
using U32x4 = uint32_t __attribute__((ext_vector_type(4)));
__device__ __forceinline__ Bf16x16 fragment(const uint16_t *values) {
    Bf16x16 result;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) result[i] = values[i];
    return result;
}

// Each CTA computes one M64/N64 projection tile instead of traversing K four
// times serially for gate/up N32 passes. Original BF16 operands and ascending
// WMMA K16 accumulation are unchanged. Exact endpoint correction follows on
// the same stream, using the original FP32 projection layout.
__global__ __launch_bounds__(256)
void matrix(const uint16_t *input, const uint16_t *weights,
            const int32_t *sorted_routes, const int32_t *block_experts,
            const int32_t *total_padded, float *output, unsigned logical_routes) {
    const unsigned block = blockIdx.x / kColumnBlocks;
    const unsigned column_block = blockIdx.x % kColumnBlocks;
    const int padded = *total_padded;
    if (padded <= 0 || block * kM >= unsigned(padded)) return;
    const int expert = expert_from_descriptor(block_experts[block]);
    if (expert < 0) return;
    __shared__ __align__(16) uint16_t a[kM * kStride];
    __shared__ __align__(16) uint16_t b[kN * kStride];
    __shared__ int32_t routes[kM];
    const unsigned thread = threadIdx.x, wave = thread / 32, lane = thread % 32;
    if (thread < kM) routes[thread] = sorted_routes[block * kM + thread];
    __syncthreads();
    F32x8 accumulators[2]{};
#pragma unroll 1
    for (unsigned base = 0; base < kHidden; base += kK) {
#pragma unroll
        for (unsigned chunk = thread; chunk < kM * kK / 8; chunk += 256) {
            const unsigned row = chunk / (kK / 8), offset = (chunk % (kK / 8)) * 8;
            U32x4 value{};
            const int route = routes[row];
            if (route >= 0 && unsigned(route) < logical_routes)
                __builtin_memcpy(&value, input + size_t(unsigned(route) / kTopK) * kHidden + base + offset, sizeof(value));
            __builtin_memcpy(a + row * kStride + offset, &value, sizeof(value));
            const size_t weight_row = size_t(expert) * (2 * kIntermediate) + column_block * kN + row;
            __builtin_memcpy(&value, weights + weight_row * kHidden + base + offset, sizeof(value));
            __builtin_memcpy(b + row * kStride + offset, &value, sizeof(value));
        }
        __syncthreads();
#pragma unroll
        for (unsigned sub = 0; sub < kK; sub += 16) {
            const Bf16x16 av = fragment(a + ((wave % 4) * 16 + lane % 16) * kStride + sub);
#pragma unroll
            for (unsigned n = 0; n < 2; ++n) {
                const Bf16x16 bv = fragment(b + ((wave / 4) * 32 + n * 16 + lane % 16) * kStride + sub);
                accumulators[n] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(av, bv, accumulators[n]);
            }
        }
        if (base + kK < kHidden) __syncthreads();
    }
#pragma unroll
    for (unsigned n = 0; n < 2; ++n) {
#pragma unroll
        for (unsigned element = 0; element < 8; ++element) {
            const Cell position = cell(wave, lane, element, n);
            const int route = routes[position.row];
            if (route >= 0 && unsigned(route) < logical_routes) {
                const unsigned column = column_block * kN + position.column;
                output[size_t(column / kIntermediate) * kProjectionStride +
                       size_t(route) * kIntermediate + column % kIntermediate] = accumulators[n][element];
            }
        }
    }
}
#endif
}  // namespace qrt_routed_parallel_gate
#undef QRT_GATE_INLINE
#endif
