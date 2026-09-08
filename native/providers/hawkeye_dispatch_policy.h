#ifndef QRT_HAWKEYE_DISPATCH_POLICY_H
#define QRT_HAWKEYE_DISPATCH_POLICY_H

#include <cstdint>

namespace qrt_hawkeye_dispatch {

// These are admission limits for the expensive diagnostic, not numerical
// approximations. A rejected request must fail before any exact-dot launch.
constexpr std::uint32_t maximum_candidates = 131072u;
constexpr std::uint32_t maximum_candidates_per_block = 64u;
constexpr double maximum_dispatch_ms = 100.0;
constexpr double maximum_correction_ms = 10000.0;

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
