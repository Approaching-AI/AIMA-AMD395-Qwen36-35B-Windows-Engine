#pragma once
#include "decoded_window_qk.h"
#include "../moe_accumulator/sm121_f32_carry.h"
#include "../moe_accumulator/sm121_exponent_mask.h"
#include "prepared_decoded_qk_workspace.h"

// Opt-in experiment. K16 position masks certify exact exponent
// maxima without changing products, carry order, or exceptional-row replay.
// Preparation and its consumer must be selected together on every call.
namespace qrt_exponent_mask_qk {
namespace decoded = qrt_sm121_decoded_bf16;

template<bool Key>
__global__ void prepare(const uint16_t* input, uint32_t* packed,
    unsigned* flags, uint16_t* transposed, unsigned tokens) {
    constexpr unsigned heads = Key ? qrt_blackwell_attention::kKvHeads : qrt_blackwell_attention::kQueryHeads;
    const unsigned row = blockIdx.x, feature = threadIdx.x;
    if (row >= tokens * heads) return;
    __shared__ unsigned invalid;
    if (!feature) invalid = 0u;
    __syncthreads();
    const unsigned token = row / heads, head = row % heads;
    const uint16_t x = input[size_t(row) * qrt_blackwell_attention::kHeadDim + feature];
    const size_t destination = Key ? (size_t(head) * qrt_blackwell_attention::kHeadDim + feature) * tokens + token
                                  : size_t(row) * qrt_blackwell_attention::kHeadDim + feature;
    int maximum = decoded::exponent(x);
#pragma unroll
    for (unsigned distance = 8u; distance; distance >>= 1u)
        maximum = max(maximum, __shfl_xor(maximum, distance, 16));
    // The fixture compiles wave32 explicitly. Each ballot contains two K16
    // rows, all lanes participate, and each subgroup keeps its own 16 bits.
    const unsigned shift = feature & 16u;
    const unsigned rank0 = unsigned(__ballot((x & 0x7fffu) && decoded::exponent(x) == maximum)) >> shift;
    const unsigned rank1 = unsigned(__ballot((x & 0x7fffu) && decoded::exponent(x) == maximum - 1)) >> shift;
    const unsigned rank2 = unsigned(__ballot((x & 0x7fffu) && decoded::exponent(x) == maximum - 2)) >> shift;
    const unsigned slot = feature & 15u;
    const uint16_t low = slot == 0u ? uint16_t(maximum) : slot == 1u ? uint16_t(rank0) :
        slot == 2u ? uint16_t(rank1) : slot == 3u ? uint16_t(rank2) : uint16_t(decoded::exponent(x));
    packed[destination] = (uint32_t(x) << 16u) | low;
    if constexpr (Key) transposed[destination] = x;
    if (!qrt_sm121_float_alignment::eligible(x)) atomicOr(&invalid, 1u);
    __syncthreads();
    if (!feature) flags[row] = invalid == 0u;
}

template<unsigned Window, bool CarryAware, unsigned Rows = 16u, unsigned Keys = 16u>
__device__ __forceinline__ void scores_body(const uint16_t* query, const uint16_t* transposed_key,
    const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride,
    unsigned prepared_query_start) {
    static_assert(Window && Window % 16u == 0u && qrt_blackwell_attention::kHeadDim % Window == 0u);
    static_assert(Rows * Keys == qrt_blackwell_attention::kThreads);
    constexpr unsigned rows = Rows, keys = Keys;
    __shared__ uint32_t qvalues[rows][Window], kvalues[Window][keys];
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
            qvalues[r][c] = query_tile + r < query_count
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
                qrt_sm121_exponent_mask::group<1u, keys, CarryAware>(group,
                    &qvalues[qr][base], &kvalues[base][kc], float_carry);
                float next;
                if (!qrt_sm121_f32_carry::accumulate<0u>(float_carry, group, &next)) { fallback = true; break; }
                float_carry = next;
            }
        }
        __syncthreads();
    }
    if (live) {
        float result = -INFINITY;
        if (active) {
            if (fallback) result = qrt_decoded_window_qk::raw_dot(
                query + (size_t(query_start + row) * qrt_blackwell_attention::kQueryHeads + head) * qrt_blackwell_attention::kHeadDim,
                transposed_key + size_t(kv_head) * qrt_blackwell_attention::kHeadDim * key_stride + key, key_stride);
            else result = float_carry * qrt_blackwell_attention::kExactScale;
        }
        output[output_cell] = result;
    }
}
template<unsigned Window, bool CarryAware, unsigned Rows = 16u, unsigned Keys = 16u>
__global__ void scores(const uint16_t* query, const uint16_t* transposed_key,
    const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    scores_body<Window, CarryAware, Rows, Keys>(query, transposed_key, packed_query,
        packed_key, query_flags, key_flags, output, query_start, query_count, stride, key_stride, 0u);
}
inline int prepare_workspace(const uint16_t* query, const uint16_t* key,
    uint16_t* transposed, const qrt_prepared_decoded_qk::Workspace& workspace, hipStream_t stream) {
    namespace arena = qrt_prepared_decoded_qk;
    if (!query || !key || !transposed || !arena::valid(workspace)) return int(hipErrorInvalidValue);
    auto* q = workspace.words;
    auto* k = q + arena::query_words;
    auto* qflags = k + arena::key_words;
    auto* kflags = qflags + arena::query_flag_words;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare<false>), dim3(workspace.tokens * qrt_blackwell_attention::kQueryHeads),
        dim3(qrt_blackwell_attention::kHeadDim), 0u, stream, query, q, qflags, nullptr, workspace.tokens);
    auto status = hipGetLastError();
    if (status != hipSuccess) return int(status);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare<true>), dim3(workspace.tokens * qrt_blackwell_attention::kKvHeads),
        dim3(qrt_blackwell_attention::kHeadDim), 0u, stream, key, k, kflags, transposed, workspace.tokens);
    return int(hipGetLastError());
}
inline int launch_workspace(const void* state, const uint16_t* query,
    const uint16_t* transposed_key, float* output, hipStream_t stream,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    namespace arena = qrt_prepared_decoded_qk;
    if (!state || !query || !transposed_key || !output) return int(hipErrorInvalidValue);
    const auto& workspace = *static_cast<const arena::Workspace*>(state);
    if (!arena::valid(workspace) || !count || count > 128u || start >= workspace.tokens ||
        count > workspace.tokens - start || stride != start + count || key_stride != workspace.tokens)
        return int(hipErrorInvalidValue);
    const auto* q = workspace.words;
    const auto* k = q + arena::query_words;
    const auto* qflags = k + arena::key_words;
    const auto* kflags = qflags + arena::query_flag_words;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scores<128u, true>),
        dim3((stride + 15u) / 16u, qrt_blackwell_attention::kQueryHeads, (count + 15u) / 16u), dim3(qrt_blackwell_attention::kThreads),
        0u, stream, query, transposed_key, q, k, qflags, kflags, output,
        start, count, stride, key_stride);
    return int(hipGetLastError());
}
} // namespace qrt_exponent_mask_qk
