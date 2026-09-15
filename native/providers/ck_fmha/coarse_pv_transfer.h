#pragma once
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_pv_transfer_bound.h"

// Isolated component: independent zero-entry chunks followed by an ordered
// transfer consumer. Original P, K32 alphas, denominator and exact replay are
// retained. No runtime dispatcher includes this header.
namespace qrt_coarse_pv_transfer {
using namespace qrt_blackwell_attention;
namespace bound = qrt_sm121_pv_transfer_bound;
struct Part { float center, absolute; };
static_assert(sizeof(Part) == 8u);

template<unsigned Chunk>
__global__ void produce(const uint16_t* values, const uint16_t* probabilities,
    const float* scales, Part* parts, unsigned start, unsigned queries, unsigned stride) {
    static_assert(Chunk == 128u || Chunk == 256u || Chunk == 512u);
    const unsigned chunks = (stride + Chunk - 1u) / Chunk;
    const unsigned chunk = blockIdx.z % chunks, query_tile = blockIdx.z / chunks * 16u;
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned head = blockIdx.y, kv = head / (kQueryHeads / kKvHeads);
    const unsigned column = blockIdx.x * kIntegerMatrixColumns + wave * 16u + lane % 16u;
    const unsigned input_row = query_tile + lane % 16u;
    const unsigned last_tokens = start + min(query_tile + 16u, queries);
    const unsigned tiles = (stride + 31u) / 32u, begin = chunk * Chunk;
    const size_t cells = size_t(queries) * kQueryHeads * kHeadDim;
    MantissaF32x8 center{}, absolute{};
    unsigned valid = 0xffu;
    if (begin < last_tokens) {
        for (unsigned base = begin; base < begin + Chunk; base += 16u) {
            NativeOperandRow left{}, right{};
            bool left_ok = true, right_ok = true;
#pragma unroll
            for (unsigned i = 0u; i < 16u; ++i) {
                const unsigned key = base + i;
                const uint16_t p = input_row < queries && key < start + input_row + 1u
                    ? probabilities[(size_t(input_row) * kQueryHeads + head) * stride + key] : 0u;
                const uint16_t v = key < last_tokens
                    ? values[(size_t(key) * kKvHeads + kv) * kHeadDim + column] : 0u;
                const bool p_ok = bound::coarse::eligible(p), v_ok = bound::coarse::eligible(v);
                left_ok &= p_ok; right_ok &= v_ok;
                left.original[i] = p_ok ? p : 0u; right.original[i] = v_ok ? v : 0u;
            }
            if (!(base % 32u)) {
#pragma unroll
                for (unsigned item = 0u; item < 8u; ++item) {
                    const unsigned row = query_tile + 2u * item + lane / 16u;
                    if (row < queries && base / 32u < (start + row + 32u) / 32u) {
                        const float alpha = scales[(size_t(row) * kQueryHeads + head) * (tiles + 1u) + base / 32u];
                        if (!bound::alpha_valid(alpha)) valid &= ~(1u << item);
                        center[item] = bound::product(center[item], alpha);
                    }
                }
            }
            const auto signed_dot = blackwell_native_mma(left, right, MantissaF32x8{});
            const auto positive_dot = blackwell_native_mma<true>(left, right, MantissaF32x8{});
#pragma unroll
            for (unsigned item = 0u; item < 8u; ++item) {
                const unsigned row = query_tile + 2u * item + lane / 16u;
                const bool p_ok = __shfl(unsigned(left_ok), 2u * item + lane / 16u, 32u) != 0u;
                if (row < queries && base / 32u < (start + row + 32u) / 32u) {
                    center[item] += signed_dot[item]; absolute[item] += positive_dot[item];
                    if (!p_ok || !right_ok) valid &= ~(1u << item);
                }
            }
        }
    }
#pragma unroll
    for (unsigned item = 0u; item < 8u; ++item) {
        const unsigned row = query_tile + 2u * item + lane / 16u;
        if (row < queries) {
            const size_t cell = (size_t(row) * kQueryHeads + head) * kHeadDim + column;
            parts[size_t(chunk) * cells + cell] = {
                center[item], valid & (1u << item) ? absolute[item] : bound::scalar::infinity()};
        }
    }
}

template<unsigned Chunk>
__global__ void consume(const Part* parts, const float* scales, float* output,
    float* errors, float* raw_accumulator, float* raw_denominator,
    unsigned start, unsigned queries, unsigned output_start, unsigned stride,
    const unsigned char* rcp_table) {
    const unsigned cell = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t cells = size_t(queries) * kQueryHeads * kHeadDim;
    if (cell >= cells) return;
    const unsigned row = cell / kHeadDim, query = row / kQueryHeads;
    const unsigned tokens = start + query + 1u, tiles = (stride + 31u) / 32u;
    const float* row_scales = scales + size_t(row) * (tiles + 1u);
    bound::State state{};
    for (unsigned chunk = 0u; chunk < (tokens + Chunk - 1u) / Chunk; ++chunk) {
        float entry = state.center; bool alpha_ok = true, erased = false;
        for (unsigned tile = chunk * (Chunk / 32u);
             tile < (chunk + 1u) * (Chunk / 32u) && tile < (tokens + 31u) / 32u; ++tile) {
            const float alpha = row_scales[tile];
            alpha_ok &= bound::alpha_valid(alpha); erased |= alpha == 0.0f;
            entry = bound::product(entry, alpha);
        }
        const auto part = parts[size_t(chunk) * cells + cell];
        state = bound::advance<Chunk / 16u>(state, entry, part.center, part.absolute, alpha_ok, erased);
    }
    const float denominator = row_scales[tiles];
    const float reciprocal = rcp_table ? qrt_sm121_attention_rcp::evaluate(rcp_table, denominator) : 1.0f / denominator;
    const size_t destination = size_t(output_start) * kQueryHeads * kHeadDim + cell;
    output[destination] = state.center * reciprocal;
    // Only the existing captured SM121 reciprocal establishes this envelope;
    // table-free safety cases conservatively replay every output.
    errors[cell] = rcp_table ? bound::scalar::finish(state.error, state.center, reciprocal) : bound::scalar::infinity();
    if (raw_accumulator) raw_accumulator[destination] = state.center;
    if (raw_denominator && !(cell % kHeadDim))
        raw_denominator[size_t(output_start) * kQueryHeads + row] = denominator;
}
template<unsigned Chunk>
inline int launch(const uint16_t* values, const uint16_t* probabilities, const float* scales,
    Part* parts, size_t capacity, float* output, float* errors, float* raw_accumulator,
    float* raw_denominator, unsigned start, unsigned queries, unsigned output_start,
    unsigned stride, const unsigned char* rcp_table, hipStream_t stream) {
    if (!values || !probabilities || !scales || !parts || !output || !errors ||
        !queries || queries > 128u || stride > 8192u || start >= stride || queries > stride - start ||
        output_start > 8192u || queries > 8192u - output_start) return int(hipErrorInvalidValue);
    const size_t cells = size_t(queries) * kQueryHeads * kHeadDim;
    const unsigned chunks = (stride + Chunk - 1u) / Chunk;
    if (capacity < cells * chunks) return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(produce<Chunk>), dim3(2u, kQueryHeads, (queries + 15u) / 16u * chunks),
        dim3(kThreads), 0u, stream, values, probabilities, scales, parts, start, queries, stride);
    auto status = hipGetLastError(); if (status != hipSuccess) return int(status);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(consume<Chunk>), dim3(unsigned((cells + 255u) / 256u)), dim3(kThreads),
        0u, stream, parts, scales, output, errors, raw_accumulator, raw_denominator,
        start, queries, output_start, stride, rcp_table);
    return int(hipGetLastError());
}
} // namespace qrt_coarse_pv_transfer
