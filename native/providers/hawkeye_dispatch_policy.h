#ifndef QRT_HAWKEYE_DISPATCH_POLICY_H
#define QRT_HAWKEYE_DISPATCH_POLICY_H

#include <cstdint>
#include <limits>

namespace qrt_hawkeye_dispatch {

// Admission applies to one bounded collection window. Long projections stream
// windows instead of failing because their aggregate candidate count is large.
// Completed intervals are latency diagnostics. External process deadlines
// bound execution; elapsed time cannot invalidate successfully completed math.
// Collection and exact computation have independent geometry. A 64 MiB index
// arena batches selection across long projections without one host roundtrip
// per eight input tokens. Even a completely dense window fits the arena.
constexpr std::uint32_t maximum_window_elements = 16777216u;
constexpr std::uint32_t maximum_candidates = maximum_window_elements;
// Keep the already exercised exact-dot quantum: at most 65536 independent
// dots per dispatch. This work quantum is independent of latency diagnostics.
constexpr std::uint32_t maximum_exact_blocks = 4096u;
// Limit the actual exact-dot CTA, not the density of a pre-compaction source
// block. The compacted kernel dispatches at most 16 candidate subgroups.
constexpr std::uint32_t maximum_candidates_per_block = 64u;
constexpr double maximum_dispatch_ms = 100.0;
constexpr double maximum_correction_ms = 10000.0;
// A queued replay burst retains every original kernel quantum and its stream
// order. Its completed latency uses the same100ms diagnostic threshold.
constexpr std::uint32_t maximum_queued_replay_dispatches = 8u;
// Device-count replay completes one collection window before returning to the
// host. Its 1024 CTAs each process at most 64 dots per four-lane subgroup at
// K <= 4096. Bound this larger unit independently of the old one-dot dispatch.
constexpr std::uint32_t maximum_device_window_elements = 4194304u;
constexpr std::uint32_t maximum_device_replay_blocks = 1024u;
constexpr double maximum_device_window_ms = 250.0;
// A resident matrix window includes cold backend/plan setup plus the matrix
// and outward finishing. Keep a separate completed unit from exact-dot work;
// setup remains in both aggregate correction wall and actual product TTFT.
constexpr double maximum_matrix_window_ms = 250.0;

constexpr std::uint32_t window_elements(
    std::uint64_t remaining, std::uint32_t capacity = maximum_window_elements
) {
    return remaining < capacity
        ? static_cast<std::uint32_t>(remaining) : capacity;
}

constexpr bool admitted(std::uint32_t candidates, std::uint32_t block_candidates) {
    return candidates <= maximum_candidates &&
        block_candidates <= maximum_candidates_per_block;
}

// Each packed candidate occupies one thread instead of a 16-lane subgroup.
// The total candidate bound remains unchanged.
constexpr bool admitted_packed(std::uint32_t candidates, std::uint32_t block_candidates) {
    return candidates <= maximum_candidates && block_candidates <= 256u;
}

// Call only after successful submission and stream completion. Both clocks
// enclose completed work; NaN, infinity and negative intervals still fail.
constexpr bool completed_intervals_valid(double dispatch_ms, double correction_ms) {
    return dispatch_ms >= 0.0 && dispatch_ms <= (std::numeric_limits<double>::max)() &&
        correction_ms >= 0.0 && correction_ms <= (std::numeric_limits<double>::max)();
}

// Preserve the original nominal classifications for diagnostics and tests.
constexpr bool time_remaining(double dispatch_ms, double correction_ms) {
    return dispatch_ms <= maximum_dispatch_ms &&
        correction_ms <= maximum_correction_ms;
}

constexpr bool device_time_remaining(double window_ms, double correction_ms) {
    return window_ms <= maximum_device_window_ms &&
        correction_ms <= maximum_correction_ms;
}

constexpr bool matrix_time_remaining(double window_ms, double correction_ms) {
    return window_ms <= maximum_matrix_window_ms &&
        correction_ms <= maximum_correction_ms;
}

}  // namespace qrt_hawkeye_dispatch

#endif
