#ifndef QRT_FLA_CHECKPOINT_H
#define QRT_FLA_CHECKPOINT_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Optional FLA export. These are unrounded recurrent states, not the BF16 H
// tiles used by attention output. The owner must also capture convolution,
// KV and hidden state before publishing a reusable model checkpoint.
namespace qrt_fla_checkpoint {
constexpr uint32_t kVersion = 1u;
constexpr uint32_t kCapacity = 3u;
constexpr uint32_t kChunk = 64u;
constexpr uint64_t kStateBytes = 32u * 128u * 128u * sizeof(float);
struct Plan {
    uint32_t struct_size, abi_version, count, reserved;
    uint32_t prefix_tokens[kCapacity], reserved_tail;
    float* states[kCapacity];
    uint64_t state_bytes[kCapacity];
};
static_assert(std::is_standard_layout<Plan>::value && std::is_trivially_copyable<Plan>::value);
static_assert(sizeof(void*) != 8u || sizeof(Plan) == 80u);

inline bool overlaps(const void* left, uint64_t left_bytes, const void* right, uint64_t right_bytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(left), b = reinterpret_cast<uintptr_t>(right);
    if (!a || !b || left_bytes > UINTPTR_MAX - a || right_bytes > UINTPTR_MAX - b) return true;
    return a < b + right_bytes && b < a + left_bytes;
}
inline bool valid(const Plan* plan, uint32_t tokens) {
    if (!plan || plan->struct_size != sizeof(Plan) || plan->abi_version != kVersion ||
        !plan->count || plan->count > kCapacity || plan->reserved || plan->reserved_tail ||
        !tokens || tokens > 65536u) return false;
    uint32_t previous = 0u;
    for (uint32_t i = 0u; i < kCapacity; ++i) {
        if (i >= plan->count) {
            if (plan->prefix_tokens[i] || plan->states[i] || plan->state_bytes[i]) return false;
            continue;
        }
        const auto address = reinterpret_cast<uintptr_t>(plan->states[i]);
        if (plan->prefix_tokens[i] <= previous || plan->prefix_tokens[i] >= tokens ||
            plan->prefix_tokens[i] % kChunk || !address || address % alignof(float) ||
            plan->state_bytes[i] < kStateBytes || kStateBytes > UINTPTR_MAX - address) return false;
        for (uint32_t j = 0u; j < i; ++j)
            if (overlaps(plan->states[i], kStateBytes, plan->states[j], kStateBytes)) return false;
        previous = plan->prefix_tokens[i];
    }
    return true;
}
inline bool disjoint(const Plan& plan, const void* buffer, uint64_t bytes) {
    for (uint32_t i = 0u; i < plan.count; ++i)
        if (overlaps(plan.states[i], kStateBytes, buffer, bytes)) return false;
    return true;
}

// This value is copied into the kernel argument block. Prefix positions are
// relative to one bounded segment; only its actual selected slots are passed.
struct Segment {
    uint32_t count = 0u;
    uint32_t prefix_tokens[kCapacity]{};
    float* states[kCapacity]{};
};
inline Segment segment(const Plan* plan, uint32_t offset, uint32_t count) {
    Segment result;
    if (!plan) return result;
    for (uint32_t i = 0u; i < plan->count; ++i) {
        const uint32_t position = plan->prefix_tokens[i];
        if (position > offset && position - offset <= count) {
            result.prefix_tokens[result.count] = position - offset;
            result.states[result.count++] = plan->states[i];
        }
    }
    return result;
}
using Launch = int (*)(const float*, const float*, float*, float*, int, void*, int32_t, const Plan*);
} // namespace qrt_fla_checkpoint
#endif
