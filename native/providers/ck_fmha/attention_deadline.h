#pragma once
#include <cmath>
#include <cstdint>

namespace qrt_sm121_attention_deadline {

// Execution supervision, not a product latency target. A later query window
// attends to more original keys even when its query count stays at 8192.
struct Budget {
    static constexpr unsigned window_queries = 8192u;
    static constexpr uint64_t reference_cells = uint64_t(8192u) * 8192u;
    unsigned first, count, maximum;

    constexpr bool valid() const {
        return count && first < maximum && count <= maximum - first;
    }

    constexpr double window_limit_seconds(unsigned completed) const {
        if (!valid() || !completed || completed > count) return 0.0;
        const unsigned offset = ((completed - 1u) / window_queries) * window_queries;
        const unsigned remaining = count - offset;
        const unsigned queries = remaining < window_queries ? remaining : window_queries;
        const uint64_t cells = uint64_t(queries) * (uint64_t(first) + offset + queries);
        // Division before addition also remains safe at unsigned input limits.
        const uint64_t units = (cells - 1u) / reference_cells + 1u;
        return 20.0 * static_cast<double>(units);
    }

    constexpr double call_limit_seconds() const {
        if (!valid()) return 0.0;
        double seconds = 0.0;
        for (uint64_t offset = 0u; offset < count; offset += window_queries) {
            const uint64_t end = offset + window_queries;
            seconds += window_limit_seconds(static_cast<unsigned>(end < count ? end : count));
        }
        return seconds;
    }
};

} // namespace qrt_sm121_attention_deadline
