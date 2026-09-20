#ifndef QRT_Q1_TRACE_POLICY_H
#define QRT_Q1_TRACE_POLICY_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace qrt_q1_trace {
constexpr uint32_t kUnselected = (std::numeric_limits<uint32_t>::max)();

inline bool full_layer(uint32_t value, uint32_t selected, bool all_layers) {
    return value < 40 && value % 4 == 3 && (all_layers || value == selected);
}

struct Selection {
    uint32_t first = kUnselected;
    uint32_t second = kUnselected;
    uint32_t count = 1;
    uint32_t layer = kUnselected;
    bool all_linear_layers = false;

    bool valid_positions() const {
        if (first == kUnselected || count == 0 || count > 512) return false;
        if (second == kUnselected) return true;
        const uint32_t distance = first > second ? first - second : second - first;
        return distance != 0 && distance < 512;
    }
    bool position(size_t value) const {
        return valid_positions() &&
            ((value >= first && value - first < count) ||
             (second != kUnselected && value == second));
    }
    bool raw_position(size_t value) const {
        return valid_positions() &&
            (value == first || (second != kUnselected && value == second));
    }
    bool linear_layer(uint32_t value) const {
        return value < 40 && value % 4 != 3 &&
            (all_linear_layers || value == layer);
    }
    size_t file_limit() const { return all_linear_layers ? 2048u : 64u; }
    size_t byte_limit() const { return (all_linear_layers ? 512u : 16u) << 20u; }
    bool may_write(size_t files, size_t total, size_t bytes) const {
        return bytes != 0 && bytes <= (2u << 20u) && files < file_limit() &&
            total <= byte_limit() && bytes <= byte_limit() - total;
    }
};
}  // namespace qrt_q1_trace
#endif
