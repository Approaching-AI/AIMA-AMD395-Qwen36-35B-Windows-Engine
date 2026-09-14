#pragma once
#include "routed_consumer_interval.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <thread>

namespace qrt_routed_consumer_audit {
constexpr unsigned guard = 128u, counters = 8u, words = 2u * guard + counters;

// Observe the original replay result before its existing write. No native
// projection, correction, norm, lookup table or activation is modified here.
__device__ inline void observe(bool up, float raw, float exact, float error,
    uint16_t gate, const uint16_t* table, unsigned* stats, unsigned index) {
    namespace c = qrt_routed_consumer;
    atomicAdd(stats, 1u);
    if (!c::finite(raw) || !c::finite(exact) || !table) { atomicAdd(stats + 5u, 1u); return; }
    const auto interval = c::range(raw, error);
    if (!interval.valid) return;
    atomicAdd(stats + 1u, 1u);
    if (!c::contains(interval, exact)) {
        atomicAdd(stats + 4u, 1u); atomicMin(stats + 7u, index);
    }
    const bool invariant = up ? c::up_constant(interval, gate, table) : c::gate_constant(interval, table);
    if (!invariant) return;
    atomicAdd(stats + 2u, 1u);
    const uint16_t original = up ? c::activated(gate, c::rounded(raw), table) : table[c::rounded(raw)];
    const uint16_t corrected = up ? c::activated(gate, c::rounded(exact), table) : table[c::rounded(exact)];
    if (original != corrected) {
        atomicAdd(stats + 3u, 1u); atomicMin(stats + 6u, index);
    }
}

// One private guarded counter surface per complete gate or up projection.
// Early returns drain queued users before release. The process runner also has
// a separate wall-clock timeout covering a stuck driver or failure drain.
class Owner {
    hipStream_t stream_;
    const char* stage_;
    unsigned* storage_ = nullptr;
public:
    Owner(hipStream_t stream, const char* stage) : stream_(stream), stage_(stage) {}
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;
    ~Owner() {
        if (storage_) { (void)hipStreamSynchronize(stream_); (void)hipFree(storage_); }
    }
    unsigned* data() const { return storage_ ? storage_ + guard : nullptr; }
    hipError_t initialize(bool enabled) {
        if (!enabled) return hipSuccess;
        auto status = hipMalloc(reinterpret_cast<void**>(&storage_), words * sizeof(unsigned));
        if (status != hipSuccess) return status;
        status = hipMemsetAsync(storage_, 0xa5, words * sizeof(unsigned), stream_);
        if (status == hipSuccess) status = hipMemsetAsync(data(), 0, counters * sizeof(unsigned), stream_);
        if (status == hipSuccess) status = hipMemsetAsync(data() + 6u, 0xff, 2u * sizeof(unsigned), stream_);
        return status;
    }
    hipError_t finish() {
        if (!storage_) return hipSuccess;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
            const auto status = hipStreamQuery(stream_);
            if (status == hipSuccess) break;
            if (status != hipErrorNotReady) return status;
            if (std::chrono::steady_clock::now() >= deadline) return hipErrorLaunchTimeOut;
            std::this_thread::yield();
        }
        std::array<unsigned, words> host{};
        auto status = hipMemcpy(host.data(), storage_, sizeof(host), hipMemcpyDeviceToHost);
        if (status != hipSuccess) return status;
        for (unsigned i = 0u; i < guard; ++i)
            if (host[i] != 0xa5a5a5a5u || host[guard + counters + i] != 0xa5a5a5a5u) return hipErrorInvalidValue;
        const unsigned* s = host.data() + guard;
        std::fprintf(stderr,
            "BATCH_MARK moe_consumer_interval_audit stage=%s tokens=8192 k=2048 selected=%u valid_ranges=%u invariant_consumers=%u "
            "invariant_endpoint_errors=%u corrected_endpoint_outside_range=%u invalid_values=%u first_endpoint_error=%u first_range_error=%u "
            "maximum_bf16_endpoints=9 workspace_bytes=%zu redzones_pass=1 original_replay_unchanged=1 diagnostic_only=1 performance_acceptance=0\n",
            stage_, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], sizeof(host));
        std::fflush(stderr);
        const bool invalid = s[5] != 0u;
        status = hipFree(storage_); storage_ = nullptr;
        return invalid ? hipErrorInvalidValue : status;
    }
};
} // namespace qrt_routed_consumer_audit
