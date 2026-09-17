#pragma once
#include "down_consumer_filter.h"

// Isolated certificate/collector replacement. A window owns complete tokens,
// so its eight-route certificate can directly publish the remaining original
// indices into the existing bounded queue. No full-shape omission mask or
// second projection/selector scan is needed. Replay and combine stay unchanged.
namespace qrt_moe_down_consumer_compact {
namespace f = qrt_moe_down_consumer_filter;
namespace c = qrt_routed_consumer;
namespace interval = qrt_moe_down_consumer;
namespace audit = qrt_moe_down_consumer_audit;
struct View {
    const float* native;
    const float* weights;
    const int32_t* experts;
    const float* input_l2;
    const float* weight_l2;
    const uint16_t* shared_down;
    const float* shared_gate;
    float error_scale;
    unsigned radius, exponent_threshold, tokens;
};

__global__ __launch_bounds__(256) void collect(View v, unsigned first_token,
    unsigned token_count, unsigned* indices, unsigned* count, unsigned* stats) {
    constexpr unsigned threads = 256u, routes = 8u, hidden = 2048u;
    const unsigned local = blockIdx.x * threads + threadIdx.x;
    const bool active = local < token_count * hidden;
    const unsigned index = first_token * hidden + local;
    const unsigned token = index / hidden, column = index % hidden;
    __shared__ unsigned queue[threads * routes], queue_count, global_base;
    __shared__ unsigned totals[8u][4u];
    if (!threadIdx.x) queue_count = 0u;
    __syncthreads();
    unsigned mask = 0u, invariant = 0u, invalid = 0u;
    if (active) {
        float raw[routes], error[routes];
#pragma unroll
        for (unsigned r = 0u; r < routes; ++r) {
            const unsigned route = token * routes + r;
            const float down = v.native[size_t(route) * hidden + column], weight = v.weights[route];
            raw[r] = interval::multiply(weight, down);
            const int32_t expert = v.experts[route];
            const bool good = expert >= 0 && expert < 256 && c::finite(weight);
            error[r] = good ? v.input_l2[route] * v.weight_l2[size_t(expert) * hidden + column] * v.error_scale * fabsf(weight)
                            : c::value(0x7f800000u);
            invalid += !good || !c::finite(raw[r]);
            if (audit::selected(down, raw[r], error[r], v.radius, v.exponent_threshold)) mask |= 1u << r;
        }
        const auto s = interval::enclose(raw, error, mask);
        const float shared = interval::multiply(v.shared_gate[token], c::widen(v.shared_down[index]));
        invalid += !c::finite(shared);
        invariant = mask && !invalid && interval::combined_constant(s, shared);
    }
    const unsigned lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
#pragma unroll
    for (unsigned r = 0u; r < routes; ++r) {
        const bool selected = active && !invariant && (mask & (1u << r));
        const unsigned ballot = __ballot(selected);
        unsigned base = !lane && ballot ? atomicAdd(&queue_count, unsigned(__popc(ballot))) : 0u;
        base = __shfl(base, 0u, 32u);
        if (selected) queue[base + unsigned(__popc(ballot & ((uint32_t(1u) << lane) - 1u)))] =
            (token * routes + r) * hidden + column;
    }
    unsigned selected_count = unsigned(__popc(mask));
    unsigned removed_count = invariant * selected_count;
    for (unsigned offset = 16u; offset; offset >>= 1u) {
        selected_count += __shfl_down(selected_count, offset, 32u);
        removed_count += __shfl_down(removed_count, offset, 32u);
        invariant += __shfl_down(invariant, offset, 32u);
        invalid += __shfl_down(invalid, offset, 32u);
    }
    if (!lane) {
        totals[warp][0] = selected_count; totals[warp][1] = removed_count;
        totals[warp][2] = invariant; totals[warp][3] = invalid;
    }
    __syncthreads();
    if (!threadIdx.x) {
        global_base = queue_count ? atomicAdd(count, queue_count) : 0u;
        unsigned selected = 0u, removed = 0u, constant = 0u, bad = 0u;
#pragma unroll
        for (unsigned i = 0u; i < 8u; ++i) {
            selected += totals[i][0]; removed += totals[i][1];
            constant += totals[i][2]; bad += totals[i][3];
        }
        atomicAdd(stats + f::Cells, min(threads, token_count * hidden - blockIdx.x * threads));
        if (selected) { atomicAdd(stats + f::Candidates, selected); atomicAdd(stats + f::Collected, selected); }
        if (removed) { atomicAdd(stats + f::Removable, removed); atomicAdd(stats + f::Omitted, removed); }
        if (constant) atomicAdd(stats + f::InvariantCells, constant);
        if (bad) atomicAdd(stats + f::InvalidValues, bad);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < queue_count; i += threads) indices[global_base + i] = queue[i];
}

inline hipError_t launch(View v, unsigned first_token, unsigned token_count,
    unsigned* indices, size_t capacity, unsigned* count, unsigned* stats, hipStream_t stream) {
    if (!v.native || !v.weights || !v.experts || !v.input_l2 || !v.weight_l2 || !v.shared_down || !v.shared_gate ||
        !indices || !count || !stats || !v.tokens || v.tokens > 8192u || !token_count || token_count > 256u ||
        first_token >= v.tokens || token_count > v.tokens - first_token ||
        capacity < size_t(token_count) * 8u * 2048u || !(v.error_scale > 0.0f) || !std::isfinite(v.error_scale))
        return hipErrorInvalidValue;
    // Worst-case selected count is all eight routes. The caller clears this
    // device count and drains/reuses the bounded queue in original stream order.
    hipLaunchKernelGGL(collect, dim3((token_count * 2048u + 255u) / 256u), dim3(256u), 0u, stream,
        v, first_token, token_count, indices, count, stats);
    return hipGetLastError();
}

// Per-invocation ownership. The source view is valid only after this call's
// shared completion event. No queue/count is retained between invocations.
class Owner {
    hipStream_t stream_;
    unsigned* storage_ = nullptr;
    View view_{};
    bool prepared_ = false;
public:
    explicit Owner(hipStream_t stream) : stream_(stream) {}
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;
    ~Owner() {
        if (storage_) { (void)hipStreamSynchronize(stream_); (void)hipFree(storage_); }
    }
    static constexpr size_t workspace_bytes = f::words * sizeof(unsigned);
    unsigned* stats() const { return storage_ ? storage_ + f::guard : nullptr; }
    hipError_t initialize(bool enabled, bool compatible) {
        if (!enabled) return hipSuccess;
        if (!compatible || storage_) return hipErrorInvalidValue;
        auto status = hipMalloc(reinterpret_cast<void**>(&storage_), workspace_bytes);
        if (status == hipSuccess) status = hipMemsetAsync(storage_, 0xa5, workspace_bytes, stream_);
        if (status == hipSuccess) status = hipMemsetAsync(stats(), 0, f::CounterCount * sizeof(unsigned), stream_);
        return status;
    }
    hipError_t prepare(View view, hipEvent_t shared_done) {
        if (!storage_) return hipSuccess;
        if (prepared_ || !shared_done || !view.native || !view.weights || !view.experts ||
            !view.input_l2 || !view.weight_l2 || !view.shared_down || !view.shared_gate ||
            !view.tokens || view.tokens > 8192u || !(view.error_scale > 0.0f) || !std::isfinite(view.error_scale))
            return hipErrorInvalidValue;
        const auto status = hipStreamWaitEvent(stream_, shared_done, 0u);
        if (status == hipSuccess) { view_ = view; prepared_ = true; }
        return status;
    }
    hipError_t collect_blocks(unsigned first, unsigned blocks, unsigned* indices,
        size_t capacity, unsigned* count) {
        if (!prepared_ || first % 64u || blocks % 64u) return hipErrorInvalidValue;
        return launch(view_, first / 64u, blocks / 64u, indices, capacity, count, stats(), stream_);
    }
    hipError_t finish() {
        if (!storage_) return hipSuccess;
        if (!prepared_) return hipErrorInvalidValue;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
            const auto status = hipStreamQuery(stream_);
            if (status == hipSuccess) break;
            if (status != hipErrorNotReady) return status;
            if (std::chrono::steady_clock::now() >= deadline) return hipErrorLaunchTimeOut;
            std::this_thread::yield();
        }
        std::array<unsigned, f::words> counters{};
        const auto status = hipMemcpy(counters.data(), storage_, workspace_bytes, hipMemcpyDeviceToHost);
        if (status != hipSuccess) return status;
        for (unsigned i = 0u; i < f::guard; ++i)
            if (counters[i] != 0xa5a5a5a5u || counters[f::guard + f::CounterCount + i] != 0xa5a5a5a5u)
                return hipErrorInvalidValue;
        const auto* s = counters.data() + f::guard;
        if (s[f::Cells] != view_.tokens * 2048u || s[f::InvalidValues] ||
            s[f::Candidates] != s[f::Collected] || s[f::Removable] != s[f::Omitted] ||
            s[f::Omitted] > s[f::Collected] || s[f::Replayed] != s[f::Collected] - s[f::Omitted] ||
            s[f::InvariantCells] > s[f::Cells]) return hipErrorInvalidValue;
        std::fprintf(stderr,
            "BATCH_MARK moe_down_consumer_compact tokens=%u cells=%u selected=%u skipped=%u replayed=%u "
            "invariant_cells=%u invalid_values=%u workspace_bytes=%zu full_unrounded_residual_certificate=1 "
            "original_selected_replay=1 shared_event_dependency=1 fused_collection=1 omission_bitmap_bytes=0 "
            "counter_identities_pass=1 redzones_pass=1 completed=1\n",
            view_.tokens,s[f::Cells],s[f::Collected],s[f::Omitted],s[f::Replayed],s[f::InvariantCells],
            s[f::InvalidValues],workspace_bytes);
        std::fflush(stderr);
        return hipSuccess;
    }
};
} // namespace qrt_moe_down_consumer_compact
