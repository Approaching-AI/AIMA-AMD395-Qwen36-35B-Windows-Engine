#ifndef QRT_REPLAY_WEIGHT_BUCKET_PLAN_H
#define QRT_REPLAY_WEIGHT_BUCKET_PLAN_H
#include <cstddef>
#include <cstdint>
#if defined(__HIPCC__)
#define QRT_BUCKET_INLINE __host__ __device__ __forceinline__
#else
#define QRT_BUCKET_INLINE inline
#endif
namespace qrt_replay_weight_buckets {
struct Plan { unsigned rows, tokens, weight_rows, token_rows; };
QRT_BUCKET_INLINE bool valid(Plan p) {
    return p.rows && p.rows <= 16384u && p.tokens && p.tokens <= 8192u &&
        ((p.weight_rows == 1u && p.token_rows == 8192u) ||
         (p.weight_rows == 16u && p.token_rows == 256u) ||
         (p.weight_rows == 64u && p.token_rows == 64u) ||
         (p.weight_rows == p.rows && p.token_rows == 1u));
}
QRT_BUCKET_INLINE unsigned columns(Plan p) { return (p.rows + p.weight_rows - 1u) / p.weight_rows; }
QRT_BUCKET_INLINE unsigned bucket_count(Plan p) {
    return valid(p) ? columns(p) * ((p.tokens + p.token_rows - 1u) / p.token_rows) : 0u;
}
QRT_BUCKET_INLINE unsigned bucket(Plan p, unsigned cell) {
    if (!valid(p) || size_t(cell) >= size_t(p.rows) * p.tokens) return UINT32_MAX;
    return (cell / p.rows / p.token_rows) * columns(p) + (cell % p.rows) / p.weight_rows;
}
// Histogram, immutable bin starts, independently advanced scatter cursors,
// and one device status word. Original indices and candidate count are inputs.
QRT_BUCKET_INLINE size_t workspace_words(Plan p) {
    const unsigned bins = bucket_count(p);
    return bins ? size_t(bins) * 3u + 2u : 0u;
}
}
#undef QRT_BUCKET_INLINE
#endif
