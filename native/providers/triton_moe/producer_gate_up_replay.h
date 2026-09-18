#pragma once

// Included inside the provider after its original selectors and exact dot.
// One completed N32 up tile owns the corresponding gate tile too. Reuse the
// now-dead matrix A stage for a local queue; keep all original predicates and
// the original ordered K16 replay. In particular, gate selection must wait
// for native up, and up selection must wait for the corrected BF16 gate.
template<bool Up>
__device__ __forceinline__ void moe_producer_projection_replay(
    const int32_t* routes, uint32_t rows, uint32_t expert, uint32_t first_column,
    uint16_t* queue, uint32_t* count, float* native, uint16_t* activated,
    const uint16_t* silu_lut, uint32_t midpoint_radius,
    uint32_t low_exponent_threshold, const MoeCorrectionBounds& bounds
#if QRT_TRITON_MOE_ROUTED_PROJECTION_DEBUG
    , uint16_t* projection_debug, float* native_debug,
    uint32_t* correction_count_debug, uint32_t debug_token
#endif
) {
    constexpr uint32_t columns = 32u;
    // All matrix loads/outputs or preceding gate replays must be finished
    // before the same arena is reused or its global outputs are consumed.
    __syncthreads();
    if (threadIdx.x == 0u) *count = 0u;
    __syncthreads();
    for (uint32_t local = threadIdx.x; local < rows * columns; local += blockDim.x) {
        const int32_t route = routes[local / columns];
        if (route < 0 || route >= static_cast<int32_t>(kRoutes)) continue;
        const uint32_t column = first_column + local % columns;
        const uint32_t index = uint32_t(route) * kIntermediate + column;
        const uint32_t weight_row = expert * (2u * kIntermediate) +
            (Up ? kIntermediate : 0u) + column;
        const float value = native[(Up ? kActivatedElements : 0u) + index];
        if constexpr (!Up) activated[index] = float_to_bf16(value);
        const uint32_t bits = __float_as_uint(value);
        const uint32_t low = bits & UINT32_C(0xffff);
        const uint32_t distance = low >= UINT32_C(0x8000)
            ? low - UINT32_C(0x8000) : UINT32_C(0x8000) - low;
        const bool absolute_candidate = moe_l2_candidate(value, bounds,
            uint32_t(route) / kTopK, weight_row);
        const bool midpoint_candidate = midpoint_radius != 0u && distance <= midpoint_radius;
        const bool low_candidate = low_exponent_threshold != 0u &&
            ((bits >> 23u) & UINT32_C(0xff)) <= low_exponent_threshold;
        bool selected;
        if constexpr (Up) {
            selected = absolute_candidate || ((midpoint_candidate || low_candidate) &&
                routed_up_projection_needs_hawkeye_replay(value, activated[index], silu_lut));
        } else {
            selected = absolute_candidate || midpoint_candidate ||
                (low_candidate && routed_gate_projection_needs_hawkeye_replay(
                    value, native[kActivatedElements + index], silu_lut));
        }
        if (selected) queue[atomicAdd(count, 1u)] = static_cast<uint16_t>(local);
#if QRT_TRITON_MOE_ROUTED_PROJECTION_DEBUG
        const uint32_t first_route = debug_token * kTopK;
        if (uint32_t(route) >= first_route && uint32_t(route) < first_route + kTopK)
            native_debug[(uint32_t(route) - first_route) * kIntermediate + column] = value;
#endif
    }
    __syncthreads();
#if QRT_TRITON_MOE_ROUTED_PROJECTION_DEBUG
    if (threadIdx.x == 0u && correction_count_debug) atomicAdd(correction_count_debug, *count);
#endif
    for (uint32_t slot = threadIdx.x / 4u; slot < *count; slot += blockDim.x / 4u) {
        const uint32_t local = queue[slot];
        const uint32_t route = static_cast<uint32_t>(routes[local / columns]);
        const uint32_t column = first_column + local % columns;
        const uint32_t index = route * kIntermediate + column;
        const uint32_t weight_row = expert * (2u * kIntermediate) +
            (Up ? kIntermediate : 0u) + column;
        // This producer variant is admitted only with lossless staged replay.
        // Keep its original per-group fallback, without generating unrelated
        // whole-row replay alternatives inside the matrix epilogue.
        const float exact = qrt_sm121_staged_half_projection::dot<2u>(
            reinterpret_cast<const qrt_sm121_staged_half_projection::Row*>(bounds.prepared_input) +
                size_t(route / kTopK) * (kHidden / 16u),
            reinterpret_cast<const qrt_sm121_staged_half_projection::Row*>(bounds.prepared_weights) +
                size_t(weight_row) * (kHidden / 16u), kHidden);
        if ((threadIdx.x & 3u) == 0u) {
            if constexpr (Up) native[kActivatedElements + index] = exact;
            else activated[index] = float_to_bf16(exact);
        }
    }
    __syncthreads();
    for (uint32_t local = threadIdx.x; local < rows * columns; local += blockDim.x) {
        const int32_t route = routes[local / columns];
        if (route < 0 || route >= static_cast<int32_t>(kRoutes)) continue;
        const uint32_t column = first_column + local % columns;
        const uint32_t index = uint32_t(route) * kIntermediate + column;
        const uint16_t projected = Up ? float_to_bf16(native[kActivatedElements + index]) : activated[index];
#if QRT_TRITON_MOE_ROUTED_PROJECTION_DEBUG
        const uint32_t first_route = debug_token * kTopK;
        if (uint32_t(route) >= first_route && uint32_t(route) < first_route + kTopK)
            projection_debug[(uint32_t(route) - first_route) * kIntermediate + column] = projected;
#endif
        if constexpr (Up) activated[index] = float_to_bf16(
            routed_silu_from_gate_bf16(activated[index], silu_lut) * bf16_to_float(projected));
    }
    // Other waves may still be reading the queue or writing activation. The
    // caller may begin a new matrix pass immediately after this helper.
    __syncthreads();
}
