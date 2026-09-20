#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <type_traits>

namespace qrt_sm121_mtp {
struct MoeBuffers {
    uint16_t* router = nullptr;
    uint16_t* shared_gate = nullptr;
    uint16_t* shared_gate_up = nullptr;
    uint16_t* shared_activated = nullptr;
    uint16_t* shared_down = nullptr;
    uint16_t* shared = nullptr;
    uint16_t* routed_gate_up = nullptr;
    uint16_t* routed_activated = nullptr;
    uint16_t* routed_weighted = nullptr;
    uint16_t* routed = nullptr;
    uint16_t* output = nullptr;
    uint32_t* topk_ids = nullptr;
    float* topk_weights = nullptr;
    uint32_t* invalid = nullptr;
};

inline size_t moe_aligned_bytes(size_t bytes) { return (bytes + 255u) & ~size_t(255u); }

inline size_t moe_workspace_bytes(unsigned rows) {
    if (!rows || rows > 2u) return 0u;
    size_t bytes = 0u;
    for (unsigned width : {256u, 1u, 1024u, 512u, 2048u, 2048u, 8192u, 4096u, 16384u, 2048u, 2048u})
        bytes += moe_aligned_bytes(size_t(rows) * width * sizeof(uint16_t));
    return bytes + 2u * moe_aligned_bytes(size_t(rows) * 8u * sizeof(uint32_t)) + 256u;
}

// Caller-owned scratch; every stage is retained for independent numerical
// qualification. A completed fence and invalid==0 are required before publish.
inline bool bind_moe_buffers(void* workspace, size_t bytes, unsigned rows, MoeBuffers* result) {
    const size_t needed = moe_workspace_bytes(rows);
    const uintptr_t base = reinterpret_cast<uintptr_t>(workspace);
    if (!result || !workspace || !needed || bytes < needed || base % 256u ||
        base > std::numeric_limits<uintptr_t>::max() - needed) return false;
    MoeBuffers next;
    auto* cursor = static_cast<unsigned char*>(workspace);
    const auto take = [&](auto** pointer, unsigned width) {
        *pointer = reinterpret_cast<std::remove_reference_t<decltype(*pointer)>>(cursor);
        cursor += moe_aligned_bytes(size_t(rows) * width * sizeof(**pointer));
    };
    take(&next.router, 256u); take(&next.shared_gate, 1u);
    take(&next.shared_gate_up, 1024u); take(&next.shared_activated, 512u);
    take(&next.shared_down, 2048u); take(&next.shared, 2048u);
    take(&next.routed_gate_up, 8192u); take(&next.routed_activated, 4096u);
    take(&next.routed_weighted, 16384u); take(&next.routed, 2048u); take(&next.output, 2048u);
    take(&next.topk_ids, 8u); take(&next.topk_weights, 8u);
    next.invalid = reinterpret_cast<uint32_t*>(cursor);
    *result = next;
    return true;
}

inline bool moe_disjoint(const void* left, size_t left_bytes, const void* right, size_t right_bytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(left), b = reinterpret_cast<uintptr_t>(right);
    if (!a || !b || !left_bytes || !right_bytes ||
        a > std::numeric_limits<uintptr_t>::max() - left_bytes ||
        b > std::numeric_limits<uintptr_t>::max() - right_bytes) return false;
    return a + left_bytes <= b || b + right_bytes <= a;
}
} // namespace qrt_sm121_mtp
