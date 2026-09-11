#ifndef QRT_PREFIX_CHECKPOINT_POLICY_H
#define QRT_PREFIX_CHECKPOINT_POLICY_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace qrt_prefix_checkpoint {
constexpr size_t kCapacity = 3u;
constexpr size_t kAlignment = 64u;

struct Positions {
    std::array<uint32_t, kCapacity> tokens{};
    size_t count = 0u;
};

// Checkpoints end at actual FLA chunk boundaries. Message boundaries are
// preferred; plain token input gets three bounded positions near its end.
// No prompt digest, expected output, or benchmark fixture selects a position.
inline Positions select(const uint32_t *tokens, size_t count) {
    Positions result;
    if (tokens == nullptr || count <= kAlignment || count > 8192u) {
        return result;
    }
    auto add = [&](size_t position) {
        position -= position % kAlignment;
        if (position == 0u || position >= count ||
            result.count == kCapacity) {
            return;
        }
        for (size_t i = 0u; i < result.count; ++i) {
            if (result.tokens[i] == position) {
                return;
            }
        }
        result.tokens[result.count++] = static_cast<uint32_t>(position);
    };
    size_t first_end = 0u;
    size_t last_end = 0u;
    for (size_t i = 0u; i < count; ++i) {
        if (tokens[i] == 248046u) {
            if (first_end == 0u) {
                first_end = i + 1u;
            }
            last_end = i + 1u;
        }
    }
    add(first_end);
    add(last_end);
    for (size_t i = last_end; i + 2u < count; ++i) {
        if (tokens[i] == 248045u && tokens[i + 1u] == 74455u &&
            tokens[i + 2u] == 198u) {
            add(i + 3u);
        }
    }
    const size_t earliest = count > 1024u
        ? ((count - 1024u + kAlignment - 1u) / kAlignment) * kAlignment
        : kAlignment;
    add(earliest);
    add(count > 512u ? count - 512u : count / 2u);
    add(count - 1u);
    std::sort(result.tokens.begin(), result.tokens.begin() + result.count);
    return result;
}

inline size_t common_prefix(const uint32_t *a, size_t na,
                            const uint32_t *b, size_t nb) {
    size_t i = 0u;
    if (a != nullptr && b != nullptr) {
        while (i < na && i < nb && a[i] == b[i]) {
            ++i;
        }
    }
    return i;
}
}  // namespace qrt_prefix_checkpoint
#endif
