#pragma once
#include <cstdint>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_SPATIAL_INLINE __host__ __device__ __forceinline__
#else
#define QRT_SPATIAL_INLINE inline
#endif

// Permute work identifiers only. Values, membership and each dot's K order
// are unaffected. Padding returns a sentinel and is never a candidate.
namespace qrt_sm121_spatial_indices {
QRT_SPATIAL_INLINE constexpr bool valid(unsigned rows, unsigned tokens, unsigned tile_rows, unsigned tile_tokens) {
    return rows && rows <= 16384u && tokens && tokens <= 8192u &&
        ((tile_rows == 64u && tile_tokens == 32u) ||
         (tile_rows == 128u && tile_tokens == 64u) ||
         (tile_rows == 256u && tile_tokens == 128u));
}
QRT_SPATIAL_INLINE constexpr unsigned extent(unsigned rows, unsigned tokens, unsigned tile_rows, unsigned tile_tokens) {
    return valid(rows, tokens, tile_rows, tile_tokens)
        ? ((rows + tile_rows - 1u) / tile_rows) * ((tokens + tile_tokens - 1u) / tile_tokens) * tile_rows * tile_tokens : 0u;
}
QRT_SPATIAL_INLINE constexpr unsigned index(unsigned virtual_index, unsigned rows, unsigned tokens, unsigned tile_rows, unsigned tile_tokens) {
    if (virtual_index >= extent(rows, tokens, tile_rows, tile_tokens)) return UINT32_MAX;
    const unsigned columns = (rows + tile_rows - 1u) / tile_rows;
    const unsigned tile = virtual_index / (tile_rows * tile_tokens);
    const unsigned local = virtual_index % (tile_rows * tile_tokens);
    const unsigned row = tile % columns * tile_rows + local % tile_rows;
    const unsigned token = tile / columns * tile_tokens + local / tile_rows;
    return row < rows && token < tokens ? token * rows + row : UINT32_MAX;
}
} // namespace qrt_sm121_spatial_indices
#undef QRT_SPATIAL_INLINE
