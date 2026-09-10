#ifndef QRT_HAWKEYE_DISPATCH_POLICY_H
#define QRT_HAWKEYE_DISPATCH_POLICY_H

#include <cstdint>

namespace qrt_hawkeye_dispatch {

// Admission applies to one bounded collection window. Long projections stream
// windows instead of failing because their aggregate candidate count is large.
// The completed-dispatch and whole-correction deadlines still bound execution.
constexpr std::uint32_t maximum_candidates = 131072u;
// At most 256 lightweight collection blocks, separately from the configured
// compacted exact-dot cap. Scratch also covers a completely dense window.
constexpr std::uint32_t maximum_window_elements = 65536u;
// Limit the actual exact-dot CTA, not the density of a pre-compaction source
// block. The compacted kernel dispatches at most 16 candidate subgroups.
constexpr std::uint32_t maximum_candidates_per_block = 64u;
constexpr double maximum_dispatch_ms = 100.0;
constexpr double maximum_correction_ms = 10000.0;

constexpr std::uint32_t window_elements(std::uint64_t remaining) {
    return remaining < maximum_window_elements
        ? static_cast<std::uint32_t>(remaining) : maximum_window_elements;
}

constexpr bool admitted(std::uint32_t candidates, std::uint32_t block_candidates) {
    return candidates <= maximum_candidates &&
        block_candidates <= maximum_candidates_per_block;
}

constexpr bool time_remaining(double dispatch_ms, double correction_ms) {
    return dispatch_ms <= maximum_dispatch_ms &&
        correction_ms <= maximum_correction_ms;
}

}  // namespace qrt_hawkeye_dispatch

#endif
