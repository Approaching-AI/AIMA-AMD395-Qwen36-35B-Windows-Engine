#pragma once
#include "sm121_q2_recurrent_layout.h"
#include <array>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_Q2_CACHE_HD __host__ __device__ __forceinline__
#else
#define QRT_Q2_CACHE_HD inline
#endif

namespace qrt_sm121_q2 {
constexpr unsigned target_context_limit = 263680u;
// The ordinary target owns separate K/V planes, with an immutable prefix and
// an append-only decode tail. Interleaved component histories use stride1024.
// Borrowing these views requires the owner to retain every allocation through
// completion. Element kinds may differ between prefix and committed tail.
struct CachePlane {
    const void* keys = nullptr;
    const void* values = nullptr;
    unsigned tokens = 0;
    unsigned capacity = 0;
    unsigned stride = 512u;
    unsigned element_bytes = 2u;
};
struct CacheView {
    CachePlane prefix;
    CachePlane decoded;
    const uint16_t* staged = nullptr; // [2,K512+V512], private candidates.

    QRT_Q2_CACHE_HD unsigned committed_tokens() const { return prefix.tokens + decoded.tokens; }
    QRT_Q2_CACHE_HD uint16_t read(unsigned token, unsigned head, unsigned channel, bool value) const {
        // Invalid coordinates cannot silently read an unrelated owner. Causal
        // masking is still performed by the attention producer before access.
        if (head >= 2u || channel >= 256u || token >= committed_tokens() + 2u) return 0x7fc0u;
        const CachePlane* plane = &prefix;
        unsigned local = token;
        if (local >= prefix.tokens) { local -= prefix.tokens; plane = &decoded; }
        if (plane == &decoded && local >= decoded.tokens)
            return staged[size_t(local - decoded.tokens)*1024u + (value ? 512u : 0u) + head*256u + channel];
        const void* data = value ? plane->values : plane->keys;
        const size_t index = size_t(local)*plane->stride + head*256u + channel;
        return plane->element_bytes == 2u ? static_cast<const uint16_t*>(data)[index]
            : qrt_sm121_q1::bf16(static_cast<const float*>(data)[index]);
    }
    QRT_Q2_CACHE_HD uint16_t key(unsigned token, unsigned head, unsigned channel) const {
        return read(token, head, channel, false);
    }
    QRT_Q2_CACHE_HD uint16_t value(unsigned token, unsigned head, unsigned channel) const {
        return read(token, head, channel, true);
    }
};

namespace cache_view_detail {
inline bool valid_plane(const CachePlane& plane) {
    if (!plane.capacity) return !plane.tokens && !plane.keys && !plane.values;
    if (plane.capacity > target_context_limit || plane.tokens > plane.capacity ||
        (plane.stride != 512u && plane.stride != 1024u) ||
        (plane.element_bytes != 2u && plane.element_bytes != 4u)) return false;
    const size_t bytes = (size_t(plane.capacity - 1u)*plane.stride + 512u)*plane.element_bytes;
    return recurrent_detail::valid_span({plane.keys, bytes, plane.element_bytes}) &&
           recurrent_detail::valid_span({plane.values, bytes, plane.element_bytes});
}
inline std::array<recurrent_detail::Span,4> history_spans(const CacheView& view) {
    const auto span = [](const CachePlane& p, bool values) {
        return recurrent_detail::Span{values ? p.values : p.keys,
            p.capacity ? (size_t(p.capacity - 1u)*p.stride + 512u)*p.element_bytes : 0u,
            p.element_bytes};
    };
    return {{span(view.prefix, false), span(view.prefix, true), span(view.decoded, false), span(view.decoded, true)}};
}
} // namespace cache_view_detail

inline bool valid_cache_view(const CacheView& view) {
    if (!cache_view_detail::valid_plane(view.prefix) || !cache_view_detail::valid_plane(view.decoded) ||
        view.prefix.tokens > target_context_limit - 2u ||
        view.decoded.tokens > target_context_limit - 2u - view.prefix.tokens) return false;
    const recurrent_detail::Span staged{view.staged, 4096u, 2u};
    if (!recurrent_detail::valid_span(staged)) return false;
    for (const auto& span : cache_view_detail::history_spans(view))
        if (span.bytes && !recurrent_detail::disjoint(span, staged)) return false;
    return true;
}
} // namespace qrt_sm121_q2
#undef QRT_Q2_CACHE_HD
