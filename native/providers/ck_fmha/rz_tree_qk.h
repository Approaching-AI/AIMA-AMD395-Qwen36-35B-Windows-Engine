#pragma once
#include "microtile_exact_qk.h"
#include "../moe_accumulator/sm121_native_rz_carry.h"
#include "../moe_accumulator/sm121_rz_tree_spec.h"

// Experimental arithmetic, not the original width26 accumulator. The only
// acceptance authority is the attached original GB10 product boundary.
namespace qrt_rz_tree_qk {
namespace native = qrt_sm121_native_rz_carry;
namespace original = qrt_q1_moe_hawkeye;
using Workspace = qrt_prepared_decoded_qk::Workspace;

using qrt_sm121_rz_tree::eligible;
using qrt_sm121_rz_tree::add_spec;

__device__ __forceinline__ float add(float a, float b) {
    float value;
    asm volatile("v_add_f32_e64 %0, %1, %2" : "=v"(value) : "v"(a), "v"(b) : "memory");
    return value;
}

template<bool Key>
__global__ void prepare(const uint16_t* input, uint32_t* packed,
    unsigned* flags, uint16_t* transposed, unsigned tokens) {
    constexpr unsigned heads = Key ? 2u : 16u;
    const unsigned row = blockIdx.x, feature = threadIdx.x;
    if (row >= tokens * heads) return;
    __shared__ unsigned invalid;
    if (!feature) invalid = 0u;
    __syncthreads();
    const uint16_t value = input[size_t(row) * 256u + feature];
    const size_t destination = Key ? (size_t(row % heads) * 256u + feature) * tokens + row / heads
                                   : size_t(row) * 256u + feature;
    packed[destination] = qrt_sm121_decoded_bf16::pack(value);
    if constexpr (Key) transposed[destination] = value;
    if (!eligible(value)) atomicOr(&invalid, 1u);
    __syncthreads();
    if (!feature) flags[row] = invalid == 0u;
}

template<unsigned Offset>
__device__ __forceinline__ float pair(const uint32_t* query,
    const uint32_t (&key)[128][32], unsigned column, unsigned base) {
    const float a0 = qrt_sm121_float_alignment::from_bits(query[base + Offset] & 0xffff0000u);
    const float a1 = qrt_sm121_float_alignment::from_bits(query[base + Offset + 1u] & 0xffff0000u);
    const float b0 = qrt_sm121_float_alignment::from_bits(key[base + Offset][column] & 0xffff0000u);
    const float b1 = qrt_sm121_float_alignment::from_bits(key[base + Offset + 1u][column] & 0xffff0000u);
    // Both products are exact normal FP32 or signed zero in the admitted range.
    return add(a0 * b0, a1 * b1);
}
__device__ __forceinline__ float group(float carry, const uint32_t* query,
    const uint32_t (&key)[128][32], unsigned column, unsigned base) {
    const float p0 = pair<0u>(query, key, column, base), p1 = pair<2u>(query, key, column, base);
    const float p2 = pair<4u>(query, key, column, base), p3 = pair<6u>(query, key, column, base);
    const float p4 = pair<8u>(query, key, column, base), p5 = pair<10u>(query, key, column, base);
    const float p6 = pair<12u>(query, key, column, base), p7 = pair<14u>(query, key, column, base);
    return add(carry, add(add(add(p0, p1), add(p2, p3)), add(add(p4, p5), add(p6, p7))));
}
__device__ __forceinline__ void store_score(float carry, bool active, bool fallback,
    float* output, unsigned row, unsigned key, unsigned head, unsigned count, unsigned stride) {
    if (row < count && key < stride)
        output[(size_t(row) * 16u + head) * stride + key] = !active ? -INFINITY
            : fallback ? qrt_sm121_float_alignment::from_bits(qrt_deferred_qk_fallback::deferred_bits)
            : native::restored_scale(carry == 0.0f ? 0.0f : carry, qrt_blackwell_attention::kExactScale);
}
template<bool Audit = false>
__global__ __launch_bounds__(256) void scores(const uint32_t* query, const uint32_t* key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride,
    unsigned long long* audit = nullptr) {
    __shared__ uint32_t qvalues[32][128], kvalues[128][32];
    const unsigned head = blockIdx.y, kv_head = head / 8u;
    const unsigned query_tile = blockIdx.z * 32u, key_tile = blockIdx.x * 32u;
    const unsigned qr = threadIdx.x / 16u, kc = threadIdx.x % 16u;
    const unsigned row0 = query_tile + qr, row1 = row0 + 16u, key0 = key_tile + kc, key1 = key0 + 16u;
    const unsigned last = start + min(query_tile + 32u, count) - 1u;
    if (key_tile > last) {
        store_score(0.0f, false, false, output, row0, key0, head, count, stride);
        store_score(0.0f, false, false, output, row0, key1, head, count, stride);
        store_score(0.0f, false, false, output, row1, key0, head, count, stride);
        store_score(0.0f, false, false, output, row1, key1, head, count, stride);
        return;
    }
    const bool a00 = row0 < count && key0 < stride && key0 <= start + row0;
    const bool a01 = row0 < count && key1 < stride && key1 <= start + row0;
    const bool a10 = row1 < count && key0 < stride && key0 <= start + row1;
    const bool a11 = row1 < count && key1 < stride && key1 <= start + row1;
    const bool f00 = a00 && (!query_flags[(start + row0) * 16u + head] || !key_flags[key0 * 2u + kv_head]);
    const bool f01 = a01 && (!query_flags[(start + row0) * 16u + head] || !key_flags[key1 * 2u + kv_head]);
    const bool f10 = a10 && (!query_flags[(start + row1) * 16u + head] || !key_flags[key0 * 2u + kv_head]);
    const bool f11 = a11 && (!query_flags[(start + row1) * 16u + head] || !key_flags[key1 * 2u + kv_head]);
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    const unsigned before = native::enter();
    unsigned during = 0u;
    if constexpr (Audit) asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 2)" : "=s"(during) : : "memory");
    for (unsigned window = 0u; window < 256u; window += 128u) {
        for (unsigned cell = threadIdx.x; cell < 32u * 128u; cell += 256u) {
            const unsigned r = cell / 128u, c = cell % 128u;
            qvalues[r][c] = query_tile + r < count
                ? query[(size_t(start + query_tile + r) * 16u + head) * 256u + window + c] : 0u;
        }
        for (unsigned cell = threadIdx.x; cell < 128u * 32u; cell += 256u) {
            const unsigned r = cell / 32u, c = cell % 32u;
            kvalues[r][c] = key_tile + c < stride
                ? key[(size_t(kv_head) * 256u + window + r) * key_stride + key_tile + c] : 0u;
        }
        __syncthreads();
        for (unsigned base = 0u; base < 128u; base += 16u) {
            if (a00 && !f00) c00 = group(c00, qvalues[qr], kvalues, kc, base);
            if (a01 && !f01) c01 = group(c01, qvalues[qr], kvalues, kc + 16u, base);
            if (a10 && !f10) c10 = group(c10, qvalues[qr + 16u], kvalues, kc, base);
            if (a11 && !f11) c11 = group(c11, qvalues[qr + 16u], kvalues, kc + 16u, base);
        }
        __syncthreads();
    }
    native::leave(before);
    if constexpr (Audit) {
        unsigned after;
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 2)" : "=s"(after) : : "memory");
        if (!(threadIdx.x % 32u)) {
            atomicAdd(audit, 1ull);
            if (during != 3u || after != before) atomicAdd(audit + 1u, 1ull);
        }
        atomicAdd(audit + 2u, (unsigned long long)(unsigned(f00) + unsigned(f01) + unsigned(f10) + unsigned(f11)));
    }
    store_score(c00, a00, f00, output, row0, key0, head, count, stride);
    store_score(c01, a01, f01, output, row0, key1, head, count, stride);
    store_score(c10, a10, f10, output, row1, key0, head, count, stride);
    store_score(c11, a11, f11, output, row1, key1, head, count, stride);
}

inline int prepare_workspace(const uint16_t* query, const uint16_t* key,
    uint16_t* transposed, const Workspace& workspace, hipStream_t stream) {
    if (!query || !key || !transposed || !qrt_prepared_decoded_qk::valid(workspace)) return int(hipErrorInvalidValue);
    auto* q = workspace.words;
    auto* k = q + qrt_prepared_decoded_qk::query_words;
    auto* qflags = k + qrt_prepared_decoded_qk::key_words;
    auto* kflags = qflags + qrt_prepared_decoded_qk::query_flag_words;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare<false>), dim3(workspace.tokens * 16u), dim3(256u), 0u, stream,
        query, q, qflags, nullptr, workspace.tokens);
    const auto status = hipGetLastError();
    if (status != hipSuccess) return int(status);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare<true>), dim3(workspace.tokens * 2u), dim3(256u), 0u, stream,
        key, k, kflags, transposed, workspace.tokens);
    return int(hipGetLastError());
}
inline int launch_workspace(const void* state, const uint16_t* query,
    const uint16_t* transposed_key, float* output, hipStream_t stream,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    if (!state || !query || !transposed_key || !output) return int(hipErrorInvalidValue);
    const auto& workspace = *static_cast<const Workspace*>(state);
    if (!qrt_prepared_decoded_qk::valid(workspace) || !count || count > 128u || start >= workspace.tokens ||
        count > workspace.tokens - start || stride != start + count || key_stride != workspace.tokens)
        return int(hipErrorInvalidValue);
    const auto* q = workspace.words;
    const auto* k = q + qrt_prepared_decoded_qk::query_words;
    const auto* qflags = k + qrt_prepared_decoded_qk::key_words;
    const auto* kflags = qflags + qrt_prepared_decoded_qk::query_flag_words;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scores<false>), dim3((stride + 31u) / 32u, 16u, (count + 31u) / 32u),
        dim3(256u), 0u, stream, q, k, qflags, kflags, output, start, count, stride, key_stride, nullptr);
    const auto status = hipGetLastError();
    if (status != hipSuccess) return int(status);
    hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,
        dim3((size_t(count) * 16u * stride + 255u) / 256u), dim3(256u), 0u, stream,
        query, transposed_key, output, start, count, stride, key_stride);
    return int(hipGetLastError());
}
} // namespace qrt_rz_tree_qk
