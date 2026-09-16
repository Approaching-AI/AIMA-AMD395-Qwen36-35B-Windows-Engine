#pragma once
#include "down_consumer_interval.h"
#include "../moe_accumulator/bf16_midpoint_selector.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <thread>

namespace qrt_moe_down_consumer_audit {
namespace c = qrt_routed_consumer;
namespace interval = qrt_moe_down_consumer;
constexpr unsigned tokens = 8192u, hidden = 2048u, routes = 8u, guard = 128u;
constexpr size_t cells = size_t(tokens) * hidden;
enum Counter : unsigned {
    SnapshotCells, SnapshotSelected, CandidateCells, ValidCandidateCells,
    Replayed, RangeValid, RangeOutside, InvalidValues, CheckedCells, SumOutside,
    RoutedInvariantCells, RoutedRemovable, CombinedInvariantCells, CombinedRemovable,
    RoutedMismatch, CombinedMismatch, ResidualMismatch, ProductionMismatch, CounterCount
};
constexpr unsigned words = 2u * guard + CounterCount;

__device__ inline void tally(unsigned* stats, unsigned field, unsigned value) {
    for (unsigned offset = 16u; offset; offset >>= 1u) value += __shfl_down(value, offset, 32u);
    if (!(threadIdx.x % 32u) && value) atomicAdd(stats + field, value);
}
__device__ inline bool selected(float native, float contribution, float error,
    unsigned radius, unsigned exponent_threshold) {
    const unsigned low = c::bits(contribution) & 0xffffu;
    const unsigned distance = low >= 0x8000u ? low - 0x8000u : 0x8000u - low;
    return (radius && distance <= radius) ||
        (exponent_threshold && ((c::bits(native) >> 23u) & 255u) <= exponent_threshold) ||
        qrt_bf16_midpoint::within_error(contribution, error);
}

// Observe the original selected replay before its existing store. No operand,
// projection, selector or corrected result is changed by this diagnostic.
__device__ inline void observe(float raw, float exact, float error, unsigned* stats) {
    atomicAdd(stats + Replayed, 1u);
    if (!c::finite(raw) || !c::finite(exact)) { atomicAdd(stats + InvalidValues, 1u); return; }
    const auto r = c::range(raw, error);
    if (r.valid) {
        atomicAdd(stats + RangeValid, 1u);
        if (!c::contains(r, exact)) atomicAdd(stats + RangeOutside, 1u);
    }
}

__global__ void snapshot(const float* native, const float* weights, const int32_t* experts,
    const float* input_l2, const float* weight_l2, float error_scale,
    unsigned radius, unsigned exponent_threshold, interval::Snapshot* snapshots,
    unsigned* stats, unsigned token_count, const float* hidden_input) {
    const size_t count = size_t(token_count) * hidden;
    for (size_t base = size_t(blockIdx.x) * blockDim.x; base < count; base += size_t(gridDim.x) * blockDim.x) {
        const size_t index = base + threadIdx.x;
        unsigned candidates = 0u, candidate_cell = 0u, valid = 0u, invalid = 0u;
        if (index < count) {
            const unsigned token = unsigned(index / hidden), column = unsigned(index % hidden);
            float raw[routes], error[routes]; unsigned mask = 0u;
            for (unsigned r = 0u; r < routes; ++r) {
                const unsigned route = token * routes + r;
                const float down = native[size_t(route) * hidden + column], weight = weights[route];
                raw[r] = interval::multiply(weight, down);
                const int32_t expert = experts[route];
                const bool good = expert >= 0 && expert < 256 && c::finite(weight);
                error[r] = good ? input_l2[route] * weight_l2[size_t(expert) * hidden + column] * error_scale * fabsf(weight)
                                : c::value(0x7f800000u);
                invalid += !good || !c::finite(raw[r]);
                if (selected(down, raw[r], error[r], radius, exponent_threshold)) mask |= 1u << r;
            }
            auto s = interval::enclose(raw, error, mask);
            s.residual = c::rounded(hidden_input[index]);
            if (invalid) s.flags &= ~interval::valid_bit;
            snapshots[index] = s;
            candidates = unsigned(__popc(mask)); candidate_cell = mask != 0u;
            valid = candidate_cell && (s.flags & interval::valid_bit);
        }
        tally(stats, SnapshotCells, unsigned(index < count));
        tally(stats, SnapshotSelected, candidates); tally(stats, CandidateCells, candidate_cell);
        tally(stats, ValidCandidateCells, valid); tally(stats, InvalidValues, invalid);
    }
}

// Shared work has completed before this launch. Certifying the combined BF16
// value also fixes the original unrounded residual, including the subsequent
// RMSNorm input. Checking only its rounded residual would be insufficient.
__global__ void inspect(const interval::Snapshot* snapshots, const float* corrected,
    const float* weights, const uint16_t* shared_down, const float* shared_gate,
    const float* production_output, unsigned* stats, unsigned token_count) {
    const size_t count = size_t(token_count) * hidden;
    for (size_t base = size_t(blockIdx.x) * blockDim.x; base < count; base += size_t(gridDim.x) * blockDim.x) {
        const size_t index = base + threadIdx.x;
        unsigned sum_outside = 0u, routed = 0u, combined = 0u, selected_count = 0u;
        unsigned routed_wrong = 0u, combined_wrong = 0u, residual_wrong = 0u, production_wrong = 0u, invalid = 0u;
        if (index < count) {
            const unsigned token = unsigned(index / hidden), column = unsigned(index % hidden);
            const auto s = snapshots[index];
            float contribution[routes];
            for (unsigned r = 0u; r < routes; ++r) {
                const unsigned route = token * routes + r;
                contribution[r] = interval::rounded(interval::multiply(weights[route], corrected[size_t(route) * hidden + column]));
            }
            const float exact = interval::sum(contribution);
            const float shared = interval::multiply(shared_gate[token], c::widen(shared_down[index]));
            const float residual = c::widen(s.residual);
            invalid = !c::finite(exact) || !c::finite(shared) || !c::finite(residual);
            production_wrong = c::bits(interval::residual(exact, shared, residual)) != c::bits(production_output[index]);
            if (s.flags & interval::valid_bit) sum_outside = !(s.low <= exact && exact <= s.high);
            selected_count = unsigned(__popc(s.flags & 255u));
            if (selected_count && !invalid) {
                routed = interval::routed_constant(s);
                combined = interval::combined_constant(s, shared);
                routed_wrong = routed && c::rounded(s.center) != c::rounded(exact);
                combined_wrong = combined && c::bits(interval::combined(s.center, shared)) != c::bits(interval::combined(exact, shared));
                residual_wrong = combined && c::bits(interval::residual(s.center, shared, residual)) != c::bits(interval::residual(exact, shared, residual));
            }
        }
        tally(stats, CheckedCells, unsigned(index < count)); tally(stats, SumOutside, sum_outside);
        tally(stats, RoutedInvariantCells, routed); tally(stats, RoutedRemovable, routed * selected_count);
        tally(stats, CombinedInvariantCells, combined); tally(stats, CombinedRemovable, combined * selected_count);
        tally(stats, RoutedMismatch, routed_wrong); tally(stats, CombinedMismatch, combined_wrong);
        tally(stats, ResidualMismatch, residual_wrong); tally(stats, InvalidValues, invalid);
        tally(stats, ProductionMismatch, production_wrong);
    }
}

class Owner {
    hipStream_t stream_;
    const float* hidden_input_;
    interval::Snapshot* snapshots_ = nullptr;
    unsigned* storage_ = nullptr;
    bool captured_ = false;
public:
    Owner(hipStream_t stream, const float* hidden_input) : stream_(stream), hidden_input_(hidden_input) {}
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;
    ~Owner() {
        if (snapshots_ || storage_) (void)hipStreamSynchronize(stream_);
        if (snapshots_) (void)hipFree(snapshots_);
        if (storage_) (void)hipFree(storage_);
    }
    unsigned* data() const { return storage_ ? storage_ + guard : nullptr; }
    hipError_t initialize(bool enabled, bool compatible) {
        if (!enabled) return hipSuccess;
        if (!compatible || !hidden_input_ || snapshots_ || storage_) return hipErrorInvalidValue;
        auto status = hipMalloc(reinterpret_cast<void**>(&snapshots_), (cells + 2u * guard) * sizeof(interval::Snapshot));
        if (status == hipSuccess) status = hipMalloc(reinterpret_cast<void**>(&storage_), words * sizeof(unsigned));
        if (status == hipSuccess) status = hipMemsetAsync(snapshots_, 0xa5, guard * sizeof(interval::Snapshot), stream_);
        if (status == hipSuccess) status = hipMemsetAsync(snapshots_ + guard + cells, 0xa5, guard * sizeof(interval::Snapshot), stream_);
        if (status == hipSuccess) status = hipMemsetAsync(storage_, 0xa5, words * sizeof(unsigned), stream_);
        if (status == hipSuccess) status = hipMemsetAsync(data(), 0, CounterCount * sizeof(unsigned), stream_);
        return status;
    }
    hipError_t capture(const float* native, const float* weights, const int32_t* experts,
        const float* input_l2, const float* weight_l2, float error_scale, unsigned radius, unsigned exponent_threshold) {
        if (!storage_) return hipSuccess;
        if (captured_ || !native || !weights || !experts || !input_l2 || !weight_l2 || !(error_scale > 0.0f)) return hipErrorInvalidValue;
        captured_ = true;
        hipLaunchKernelGGL(snapshot, dim3(4096u), dim3(256u), 0u, stream_, native, weights, experts,
            input_l2, weight_l2, error_scale, radius, exponent_threshold, snapshots_ + guard, data(), tokens, hidden_input_);
        return hipGetLastError();
    }
    hipError_t finish(const float* corrected, const float* weights, const uint16_t* shared_down,
        const float* shared_gate, const float* production_output) {
        if (!storage_) return hipSuccess;
        if (!captured_ || !corrected || !weights || !shared_down || !shared_gate || !production_output) return hipErrorInvalidValue;
        hipLaunchKernelGGL(inspect, dim3(4096u), dim3(256u), 0u, stream_, snapshots_ + guard,
            corrected, weights, shared_down, shared_gate, production_output, data(), tokens);
        auto status = hipGetLastError(); if (status != hipSuccess) return status;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
            status = hipStreamQuery(stream_);
            if (status == hipSuccess) break;
            if (status != hipErrorNotReady) return status;
            if (std::chrono::steady_clock::now() >= deadline) return hipErrorLaunchTimeOut;
            std::this_thread::yield();
        }
        std::array<unsigned, words> counters{};
        status = hipMemcpy(counters.data(), storage_, sizeof(counters), hipMemcpyDeviceToHost);
        if (status != hipSuccess) return status;
        std::array<unsigned, guard * sizeof(interval::Snapshot) / sizeof(unsigned)> redzone{};
        for (size_t offset : {size_t(0u), guard + cells}) {
            status = hipMemcpy(redzone.data(), snapshots_ + offset, sizeof(redzone), hipMemcpyDeviceToHost);
            if (status != hipSuccess) return status;
            for (auto word : redzone) if (word != 0xa5a5a5a5u) return hipErrorInvalidValue;
        }
        for (unsigned i = 0u; i < guard; ++i)
            if (counters[i] != 0xa5a5a5a5u || counters[guard + CounterCount + i] != 0xa5a5a5a5u) return hipErrorInvalidValue;
        const auto* s = counters.data() + guard;
        if (s[SnapshotCells] != cells || s[CheckedCells] != cells || s[SnapshotSelected] != s[Replayed] ||
            s[InvalidValues] || s[CombinedRemovable] > s[Replayed] || s[RoutedRemovable] > s[CombinedRemovable]) return hipErrorInvalidValue;
        std::fprintf(stderr,
            "BATCH_MARK moe_down_consumer_audit tokens=8192 cells=%u selected=%u candidate_cells=%u valid_candidate_cells=%u "
            "replayed=%u valid_ranges=%u corrected_endpoint_outside_range=%u invalid_values=%u checked_cells=%u routed_sum_outside=%u "
            "routed_invariant_cells=%u routed_removable=%u combined_invariant_cells=%u combined_removable=%u "
            "routed_invariant_errors=%u combined_invariant_errors=%u unrounded_residual_errors=%u production_output_errors=%u "
            "workspace_bytes=%zu original_replay_unchanged=1 diagnostic_only=1 redzones_pass=1 performance_acceptance=0\n",
            s[SnapshotCells], s[SnapshotSelected], s[CandidateCells], s[ValidCandidateCells], s[Replayed], s[RangeValid], s[RangeOutside],
            s[InvalidValues], s[CheckedCells], s[SumOutside], s[RoutedInvariantCells], s[RoutedRemovable], s[CombinedInvariantCells],
            s[CombinedRemovable], s[RoutedMismatch], s[CombinedMismatch], s[ResidualMismatch], s[ProductionMismatch],
            (cells + 2u * guard) * sizeof(interval::Snapshot) + words * sizeof(unsigned));
        std::fflush(stderr);
        return hipSuccess;
    }
};
} // namespace qrt_moe_down_consumer_audit
