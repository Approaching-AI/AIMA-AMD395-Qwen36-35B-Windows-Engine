#pragma once
#include "down_consumer_audit.h"

namespace qrt_moe_down_consumer_filter {
namespace c = qrt_routed_consumer;
namespace interval = qrt_moe_down_consumer;
namespace audit = qrt_moe_down_consumer_audit;
constexpr unsigned tokens = 8192u, hidden = 2048u, routes = 8u, guard = 128u;
constexpr size_t cells = size_t(tokens) * hidden;
enum Counter : unsigned {
    Cells, Candidates, Removable, InvariantCells, Collected, Omitted, Replayed, InvalidValues, CounterCount
};
constexpr unsigned words = 2u * guard + CounterCount;
constexpr size_t workspace_bytes = cells + 2u * guard + words * sizeof(unsigned);

__device__ inline bool omitted(const uint8_t* masks, unsigned index) {
    const unsigned route = index / hidden;
    return (masks[size_t(route / routes) * hidden + index % hidden] & (1u << (route % routes))) != 0u;
}
__device__ inline void collected(unsigned* stats, unsigned selected, unsigned skipped) {
    audit::tally(stats, Collected, selected);
    audit::tally(stats, Omitted, skipped);
}
__device__ inline void replayed(unsigned* stats, unsigned count) { audit::tally(stats, Replayed, count); }

// Certify all selected contributions to a token/channel together. An invalid
// or nonconstant interval retains every original selected K16 replay. A fixed
// combined BF16 value fixes the complete original unrounded F32 residual.
__global__ void certify(const float* native, const float* weights, const int32_t* experts,
    const float* input_l2, const float* weight_l2, float error_scale,
    unsigned radius, unsigned exponent_threshold, const uint16_t* shared_down,
    const float* shared_gate, uint8_t* masks, unsigned* stats, unsigned token_count) {
    const size_t count = size_t(token_count) * hidden;
    for (size_t base = size_t(blockIdx.x) * blockDim.x; base < count; base += size_t(gridDim.x) * blockDim.x) {
        const size_t index = base + threadIdx.x;
        unsigned candidates = 0u, removed = 0u, invariant = 0u, invalid = 0u;
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
                if (audit::selected(down, raw[r], error[r], radius, exponent_threshold)) mask |= 1u << r;
            }
            const auto s = interval::enclose(raw, error, mask);
            const float shared = interval::multiply(shared_gate[token], c::widen(shared_down[index]));
            invalid += !c::finite(shared);
            invariant = mask && !invalid && interval::combined_constant(s, shared);
            masks[index] = invariant ? uint8_t(mask) : 0u;
            candidates = unsigned(__popc(mask)); removed = invariant * candidates;
        }
        audit::tally(stats, Cells, unsigned(index < count));
        audit::tally(stats, Candidates, candidates); audit::tally(stats, Removable, removed);
        audit::tally(stats, InvariantCells, invariant); audit::tally(stats, InvalidValues, invalid);
    }
}

class Owner {
    hipStream_t stream_;
    uint8_t* masks_ = nullptr;
    unsigned* storage_ = nullptr;
    bool prepared_ = false;
public:
    explicit Owner(hipStream_t stream) : stream_(stream) {}
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;
    ~Owner() {
        if (masks_ || storage_) (void)hipStreamSynchronize(stream_);
        if (masks_) (void)hipFree(masks_);
        if (storage_) (void)hipFree(storage_);
    }
    const uint8_t* data() const { return masks_ ? masks_ + guard : nullptr; }
    unsigned* stats() const { return storage_ ? storage_ + guard : nullptr; }
    hipError_t initialize(bool enabled, bool compatible) {
        if (!enabled) return hipSuccess;
        if (!compatible || masks_ || storage_) return hipErrorInvalidValue;
        auto status = hipMalloc(reinterpret_cast<void**>(&masks_), cells + 2u * guard);
        if (status == hipSuccess) status = hipMalloc(reinterpret_cast<void**>(&storage_), words * sizeof(unsigned));
        if (status == hipSuccess) status = hipMemsetAsync(masks_, 0xa5, guard, stream_);
        if (status == hipSuccess) status = hipMemsetAsync(masks_ + guard + cells, 0xa5, guard, stream_);
        if (status == hipSuccess) status = hipMemsetAsync(storage_, 0xa5, words * sizeof(unsigned), stream_);
        if (status == hipSuccess) status = hipMemsetAsync(stats(), 0, CounterCount * sizeof(unsigned), stream_);
        return status;
    }
    hipError_t prepare(const float* native, const float* weights, const int32_t* experts,
        const float* input_l2, const float* weight_l2, float error_scale, unsigned radius,
        unsigned exponent_threshold, const uint16_t* shared_down, const float* shared_gate,
        hipEvent_t shared_done) {
        if (!storage_) return hipSuccess;
        if (prepared_ || !native || !weights || !experts || !input_l2 || !weight_l2 ||
            !shared_down || !shared_gate || !shared_done || !(error_scale > 0.0f)) return hipErrorInvalidValue;
        // The caller records this same invocation's event before routed work.
        auto status = hipStreamWaitEvent(stream_, shared_done, 0u);
        if (status != hipSuccess) return status;
        hipLaunchKernelGGL(certify, dim3(4096u), dim3(256u), 0u, stream_, native, weights, experts,
            input_l2, weight_l2, error_scale, radius, exponent_threshold, shared_down, shared_gate,
            masks_ + guard, stats(), tokens);
        status = hipGetLastError(); prepared_ = status == hipSuccess;
        return status;
    }
    hipError_t finish() {
        if (!storage_) return hipSuccess;
        if (!prepared_) return hipErrorInvalidValue;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        hipError_t status;
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
        std::array<uint8_t, guard> redzone{};
        for (size_t offset : {size_t(0u), guard + cells}) {
            status = hipMemcpy(redzone.data(), masks_ + offset, sizeof(redzone), hipMemcpyDeviceToHost);
            if (status != hipSuccess) return status;
            for (auto byte : redzone) if (byte != 0xa5u) return hipErrorInvalidValue;
        }
        for (unsigned i = 0u; i < guard; ++i)
            if (counters[i] != 0xa5a5a5a5u || counters[guard + CounterCount + i] != 0xa5a5a5a5u) return hipErrorInvalidValue;
        const auto* s = counters.data() + guard;
        if (s[Cells] != cells || s[InvalidValues] || s[Candidates] != s[Collected] ||
            s[Removable] != s[Omitted] || s[Omitted] > s[Collected] ||
            s[Replayed] != s[Collected] - s[Omitted] || s[InvariantCells] > cells) return hipErrorInvalidValue;
        std::fprintf(stderr,
            "BATCH_MARK moe_down_consumer_filter tokens=8192 cells=%u selected=%u skipped=%u replayed=%u invariant_cells=%u "
            "invalid_values=%u workspace_bytes=%zu full_unrounded_residual_certificate=1 original_selected_replay=1 "
            "shared_event_dependency=1 counter_identities_pass=1 redzones_pass=1 completed=1\n",
            s[Cells], s[Collected], s[Omitted], s[Replayed], s[InvariantCells], s[InvalidValues], workspace_bytes);
        std::fflush(stderr);
        return hipSuccess;
    }
};
} // namespace qrt_moe_down_consumer_filter
