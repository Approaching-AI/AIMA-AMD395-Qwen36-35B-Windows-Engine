#ifndef QRT_FLA_BLACKWELL_KKT_H
#define QRT_FLA_BLACKWELL_KKT_H
#include "blackwell_accumulator.h"
#include "sm121_exp2_table.h"
#include "../moe_accumulator/sm121_float_alignment.h"
#include <cstdlib>
#include <cstring>
namespace qrt_fla_blackwell {

__global__ void dot_kernel(const uint16_t* k, const uint16_t* beta, float* a,
                           unsigned int chunk_index) {
    chunk_index += blockIdx.z;
    const unsigned int cell = blockIdx.x * (kThreads / kGroup) + threadIdx.x / kGroup;
    const unsigned int row = cell / kChunk, column = cell % kChunk;
    const unsigned int head = blockIdx.y, lane = threadIdx.x % kGroup;
    const unsigned int token = chunk_index * kChunk + row;
    const unsigned int index = (token * 32 + head) * kChunk + column;
    if (row >= kChunk) return;
    if (column >= row) { if (lane == 0) a[index] = 0.0f; return; }
    const unsigned int other = chunk_index * kChunk + column;
    const float beta_value = from_bf16(beta[token * 32 + head]);
    qrt_q1_moe_hawkeye::Value accumulator{0u, kZeroExponent, false};
    for (unsigned int base = 0; base < 128; base += kGroup) {
        const unsigned int dimension = base + lane;
        const uint16_t left = to_bf16(from_bf16(k[(token * 16 + head / 2) * 128 + dimension]) * beta_value);
        const uint16_t right = k[(other * 16 + head / 2) * 128 + dimension];
        accumulator = accumulate(accumulator, left, right, lane);
    }
    if (lane == 0) {
        accumulator = qrt_sm121_group16::finish_accumulator(accumulator);
        a[index] = qrt_q1_moe_hawkeye::value_to_float(accumulator);
    }
}

// Reuse rounded beta*K query rows and original K columns across 256 scalar
// cells. The separate gate kernel still applies the original exponential.
template<bool FloatProducts>
__global__ void tiled_dot_kernel(const uint16_t* k, const uint16_t* beta, float* a,
                                unsigned first_chunk) {
    constexpr unsigned rows = 8u, columns = 32u, pairs = 64u;
    __shared__ uint32_t left[rows][pairs], right[pairs][columns];
    __shared__ unsigned fallback;
    const unsigned chunk = first_chunk + blockIdx.z / (64u / rows);
    const unsigned first_row = (blockIdx.z % (64u / rows)) * rows;
    const unsigned first_column = blockIdx.x * columns;
    const unsigned row = first_row + threadIdx.x / columns;
    const unsigned column = first_column + threadIdx.x % columns;
    const unsigned head = blockIdx.y;
    const size_t index = ((size_t(chunk) * 64u + row) * 32u + head) * 64u + column;
    if (first_column >= first_row + rows - 1u) { a[index] = 0.0f; return; }
    if (!threadIdx.x) fallback = 0u;
    __syncthreads();
    bool invalid = false;
    for (unsigned cell = threadIdx.x; cell < rows * pairs; cell += kThreads) {
        const unsigned r = cell / pairs, pair = cell % pairs;
        const size_t token = size_t(chunk) * 64u + first_row + r;
        const float scale = from_bf16(beta[token * 32u + head]);
        const size_t source = (token * 16u + head / 2u) * 128u + pair * 2u;
        const uint16_t x = to_bf16(from_bf16(k[source]) * scale);
        const uint16_t y = to_bf16(from_bf16(k[source + 1u]) * scale);
        left[r][pair] = uint32_t(x) | (uint32_t(y) << 16u);
        if constexpr (FloatProducts)
            invalid |= !qrt_sm121_float_alignment::eligible(x) || !qrt_sm121_float_alignment::eligible(y);
    }
    for (unsigned cell = threadIdx.x; cell < pairs * columns; cell += kThreads) {
        const unsigned pair = cell / columns, c = cell % columns;
        const size_t source = ((size_t(chunk) * 64u + first_column + c) * 16u + head / 2u) * 128u + pair * 2u;
        const uint16_t x = k[source], y = k[source + 1u];
        right[pair][c] = uint32_t(x) | (uint32_t(y) << 16u);
        if constexpr (FloatProducts)
            invalid |= !qrt_sm121_float_alignment::eligible(x) || !qrt_sm121_float_alignment::eligible(y);
    }
    if constexpr (FloatProducts) { if (invalid) atomicOr(&fallback, 1u); }
    __syncthreads();
    if (column >= row) { a[index] = 0.0f; return; }
    qrt_q1_moe_hawkeye::Value carry{0u, kZeroExponent, false};
    for (unsigned base = 0u; base < 128u; base += 16u) {
        qrt_sm121_group16::AlignedSum sum;
        bool accepted = false;
        if constexpr (FloatProducts) {
            if (!fallback) {
                qrt_sm121_float_alignment::Group group;
#pragma unroll
                for (unsigned i = 0u; i < 16u; i += 2u) {
                    const uint32_t x = left[threadIdx.x / columns][(base + i) / 2u];
                    const uint32_t y = right[(base + i) / 2u][threadIdx.x % columns];
                    group.set(i, uint16_t(x), uint16_t(y));
                    group.set(i + 1u, uint16_t(x >> 16u), uint16_t(y >> 16u));
                }
                accepted = qrt_sm121_float_alignment::sum(carry, group, &sum);
            }
        }
        if (!accepted) {
            uint32_t products[16];
#pragma unroll
            for (unsigned i = 0u; i < 16u; i += 2u) {
                const uint32_t x = left[threadIdx.x / columns][(base + i) / 2u];
                const uint32_t y = right[(base + i) / 2u][threadIdx.x % columns];
                products[i] = qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(
                    uint16_t(x), uint16_t(y), kZeroExponent));
                products[i + 1u] = qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(
                    uint16_t(x >> 16u), uint16_t(y >> 16u), kZeroExponent));
            }
            sum = qrt_sm121_group16::sum_packed(carry, products);
        }
        carry = qrt_sm121_wave16::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
    }
    a[index] = qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}

inline int tiled_kkt_mode() {
    const char* value = std::getenv("QRT_FLA_GDN_TILED_KKT");
    if (!value || !*value || !std::strcmp(value, "0")) return 0;
    if (!std::strcmp(value, "1")) return 1;
    if (!std::strcmp(value, "2")) return 2;
    return -1;
}

inline hipError_t launch_dot_chunks(const uint16_t* k, const uint16_t* beta, float* a,
                                    unsigned first_chunk, unsigned chunks,
                                    unsigned mode, hipStream_t stream) {
    if (!k || !beta || !a || mode > 2u || !chunks || chunks > 16u ||
        first_chunk > 16u - chunks) return hipErrorInvalidValue;
    if (!mode) {
        hipLaunchKernelGGL(dot_kernel, dim3(kChunk * kChunk / (kThreads / kGroup), 32u, chunks),
            dim3(kThreads), 0u, stream, k, beta, a, first_chunk);
    } else if (mode == 1u) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(tiled_dot_kernel<false>), dim3(2u, 32u, chunks * 8u),
            dim3(kThreads), 0u, stream, k, beta, a, first_chunk);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(tiled_dot_kernel<true>), dim3(2u, 32u, chunks * 8u),
            dim3(kThreads), 0u, stream, k, beta, a, first_chunk);
    }
    return hipGetLastError();
}

__global__ void gate_kernel(float* a, const float* g, unsigned int tokens,
                            unsigned int valid_tokens, const unsigned char* table) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * 32 * kChunk) return;
    const unsigned int token = index / (32 * kChunk);
    // A padded scan endpoint can round above the last real prefix sum. Its
    // positive exponent is outside the negative lookup domain, even though
    // the padded dot is zero. Do not let 0 * NaN enter the triangular solve.
    if (token >= valid_tokens) { a[index] = 0.0f; return; }
    const unsigned int head = index / kChunk % 32, column = index % kChunk;
    if (column >= token % kChunk) return;  // Upper triangle is already zero.
    const float difference = g[token * 32 + head] - g[(token / kChunk * kChunk + column) * 32 + head];
    const float argument = difference * 1.4426950408889634074f;
    a[index] *= table ? qrt_sm121_exp2::evaluate(table, argument) : exp2f(argument);
}
}  // namespace qrt_fla_blackwell
#endif
