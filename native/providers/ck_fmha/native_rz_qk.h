#pragma once
#include "prepared_decoded_qk.h"
#include "../moe_accumulator/sm121_native_rz_carry.h"

// Component only. Keep the prepared operand format, tile layout, ordered K16
// traversal, masks and complete original fallback of the retained QK route.
namespace qrt_native_rz_qk {
namespace decoded = qrt_sm121_decoded_bf16;
namespace native = qrt_sm121_native_rz_carry;
__device__ __forceinline__ unsigned mode() {
    unsigned value;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 8)\n\t" : "=s"(value) : : "memory");
    return value;
}
template<bool NativeRz, bool Audit = false>
__global__ void scores(const uint16_t* query, const uint16_t* transposed_key,
    const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride,
    unsigned long long* counters = nullptr) {
    constexpr unsigned Window = 128u, Rows = 16u, Keys = 16u, prepared_query_start = 0u;
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
    unsigned saved = 0u, before = 0u, during = 0u;
    if constexpr (Audit) before = mode();
    if constexpr (NativeRz) saved = native::enter();
    if constexpr (Audit) during = mode();
    unsigned fast_groups = 0u;
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
#pragma unroll
                for (unsigned i = 0u; i < 16u; ++i)
                    decoded::set_packed(group, i, qvalues[qr][base + i], kvalues[base + i][kc]);
                float next;
                bool accepted;
                if constexpr (NativeRz) accepted = native::accumulate(float_carry, group, &next);
                else accepted = qrt_sm121_f32_carry::accumulate<0u>(float_carry, group, &next);
                if (!accepted) { fallback = true; break; }
                float_carry = next;
                if constexpr (Audit) ++fast_groups;
            }
        }
        __syncthreads();
    }
    // All threads converge after the final barrier. Restore before the
    // noinline original fallback, and explicitly order the final scalar scale.
    if constexpr (NativeRz) native::leave(saved);
    if constexpr (Audit) {
        const unsigned after = mode();
        if (fast_groups) atomicAdd(counters, static_cast<unsigned long long>(fast_groups));
        if (active && fallback) atomicAdd(counters + 1u, 1ull);
        if (!(threadIdx.x % 32u)) {
            if (after != before || during != (NativeRz ? (before & ~3u) | 3u : before)) atomicAdd(counters + 2u, 1ull);
            atomicAdd(counters + 3u, 1ull);
        }
    }
    if (live) {
        float result = -INFINITY;
        if (active) {
            if (fallback) result = qrt_decoded_window_qk::raw_dot(
                query + (size_t(query_start + row) * qrt_blackwell_attention::kQueryHeads + head) * qrt_blackwell_attention::kHeadDim,
                transposed_key + size_t(kv_head) * qrt_blackwell_attention::kHeadDim * key_stride + key, key_stride);
            else result = native::restored_scale(float_carry, qrt_blackwell_attention::kExactScale);
        }
        output[output_cell] = result;
    }
}
} // namespace qrt_native_rz_qk
