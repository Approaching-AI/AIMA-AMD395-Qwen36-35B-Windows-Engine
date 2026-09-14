#pragma once
#include "decoded_window_qk.h"
#include "../moe_accumulator/sm121_f32_carry.h"
#include "prepared_decoded_qk_workspace.h"

// Original BF16 operands remain available for every exceptional row or
// unsupported carried value. The provider enables this only for selected
// cold prefill calls through q8192, refreshing metadata on every layer/call.
namespace qrt_prepared_decoded_qk {
using namespace qrt_blackwell_attention;
namespace decoded = qrt_sm121_decoded_bf16;

template<bool Key>
__global__ void prepare(const uint16_t* input, uint32_t* packed,
    unsigned* flags, uint16_t* transposed, unsigned tokens) {
    constexpr unsigned heads = Key ? kKvHeads : kQueryHeads;
    const unsigned row = blockIdx.x, feature = threadIdx.x;
    if (row >= tokens * heads) return;
    __shared__ unsigned invalid;
    if (!feature) invalid = 0u;
    __syncthreads();
    const unsigned token = row / heads, head = row % heads;
    const uint16_t x = input[size_t(row) * kHeadDim + feature];
    const size_t destination = Key ? (size_t(head) * kHeadDim + feature) * tokens + token
                                  : size_t(row) * kHeadDim + feature;
    packed[destination] = decoded::pack(x);
    if constexpr (Key) transposed[destination] = x;
    if (!qrt_sm121_float_alignment::eligible(x)) atomicOr(&invalid, 1u);
    __syncthreads();
    if (!feature) flags[row] = invalid == 0u;
}

template<unsigned Window, bool FloatCarry, unsigned Rows = 16u, unsigned Keys = 16u>
__global__ void scores(const uint16_t* query, const uint16_t* transposed_key,
    const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    static_assert(Window && Window % 16u == 0u && kHeadDim % Window == 0u);
    static_assert(Rows * Keys == kThreads);
    constexpr unsigned rows = Rows, keys = Keys;
    __shared__ uint32_t qvalues[rows][Window], kvalues[Window][keys];
    const unsigned head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned query_tile = blockIdx.z * rows, key_tile = blockIdx.x * keys;
    const unsigned qr = threadIdx.x / keys, kc = threadIdx.x % keys;
    const unsigned row = query_tile + qr, key = key_tile + kc;
    const bool live = row < query_count && key < stride;
    const bool active = live && key <= query_start + row;
    const size_t output_cell = (size_t(row) * kQueryHeads + head) * stride + key;
    const unsigned last_query = query_start + min(query_tile + rows, query_count) - 1u;
    if (key_tile > last_query) {
        if (live) output[output_cell] = -INFINITY;
        return;
    }
    bool fallback = active && (!query_flags[(query_start + row) * kQueryHeads + head] ||
        !key_flags[key * kKvHeads + kv_head]);
    qrt_q1_moe_hawkeye::Value carry{0u, kBlackwellZeroExponent, false};
    float float_carry = 0.0f;
    for (unsigned window = 0u; window < kHeadDim; window += Window) {
        for (unsigned cell = threadIdx.x; cell < rows * Window; cell += kThreads) {
            const unsigned r = cell / Window, c = cell % Window;
            qvalues[r][c] = query_tile + r < query_count
                ? packed_query[(size_t(query_start + query_tile + r) * kQueryHeads + head) * kHeadDim + window + c]
                : decoded::pack(0u);
        }
        for (unsigned cell = threadIdx.x; cell < Window * keys; cell += kThreads) {
            const unsigned r = cell / keys, c = cell % keys;
            kvalues[r][c] = key_tile + c < stride
                ? packed_key[(size_t(kv_head) * kHeadDim + window + r) * key_stride + key_tile + c]
                : decoded::pack(0u);
        }
        __syncthreads();
        if (active && !fallback) {
            for (unsigned base = 0u; base < Window; base += 16u) {
                qrt_sm121_float_alignment::Group group;
#pragma unroll
                for (unsigned i = 0u; i < 16u; ++i)
                    decoded::set_packed(group, i, qvalues[qr][base + i], kvalues[base + i][kc]);
                if constexpr (FloatCarry) {
                    float next;
                    if (!qrt_sm121_f32_carry::accumulate<0u>(float_carry, group, &next)) { fallback = true; break; }
                    float_carry = next;
                } else {
                    qrt_sm121_group16::AlignedSum sum;
                    if (!qrt_sm121_float_alignment::sum(carry, group, &sum)) { fallback = true; break; }
                    carry = qrt_sm121_wave16::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
                }
            }
        }
        __syncthreads();
    }
    if (live) {
        float result = -INFINITY;
        if (active) {
            if (fallback) result = qrt_decoded_window_qk::raw_dot(
                query + (size_t(query_start + row) * kQueryHeads + head) * kHeadDim,
                transposed_key + size_t(kv_head) * kHeadDim * key_stride + key, key_stride);
            else if constexpr (FloatCarry) result = float_carry * kExactScale;
            else result = qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry)) * kExactScale;
        }
        output[output_cell] = result;
    }
}
inline int prepare_workspace(const uint16_t* query, const uint16_t* key,
    uint16_t* transposed, const Workspace& workspace, hipStream_t stream) {
    if (!query || !key || !transposed || !valid(workspace)) return int(hipErrorInvalidValue);
    auto* q = workspace.words;
    auto* k = q + query_words;
    auto* qflags = k + key_words;
    auto* kflags = qflags + query_flag_words;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare<false>), dim3(workspace.tokens * kQueryHeads),
        dim3(kHeadDim), 0u, stream, query, q, qflags, nullptr, workspace.tokens);
    auto status = hipGetLastError();
    if (status != hipSuccess) return int(status);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare<true>), dim3(workspace.tokens * kKvHeads),
        dim3(kHeadDim), 0u, stream, key, k, kflags, transposed, workspace.tokens);
    return int(hipGetLastError());
}
inline int launch_workspace(const void* state, const uint16_t* query,
    const uint16_t* transposed_key, float* output, hipStream_t stream,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    if (!state || !query || !transposed_key || !output) return int(hipErrorInvalidValue);
    const auto& workspace = *static_cast<const Workspace*>(state);
    if (!valid(workspace) || !count || count > 128u || start >= workspace.tokens ||
        count > workspace.tokens - start || stride != start + count || key_stride != workspace.tokens)
        return int(hipErrorInvalidValue);
    const auto* q = workspace.words;
    const auto* k = q + query_words;
    const auto* qflags = k + key_words;
    const auto* kflags = qflags + query_flag_words;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scores<128u, true>),
        dim3((stride + 15u) / 16u, kQueryHeads, (count + 15u) / 16u), dim3(kThreads),
        0u, stream, query, transposed_key, q, k, qflags, kflags, output,
        start, count, stride, key_stride);
    return int(hipGetLastError());
}
} // namespace qrt_prepared_decoded_qk
