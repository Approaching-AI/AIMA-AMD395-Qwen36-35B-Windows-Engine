#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_BLOCKED_HALF_INLINE __host__ __device__ constexpr inline
#else
#define QRT_BLOCKED_HALF_INLINE constexpr inline
#endif

// Keep each lane's two adjacent K16 records contiguous, and place neighboring
// weight rows in the same small block. This is a component layout; the current
// provider continues to use its qualified row-major prepared operands.
namespace qrt_sm121_blocked_half_layout {
constexpr unsigned tile_rows = 32u, tile_groups = 2u;
constexpr unsigned maximum_rows = 524288u, maximum_width = 8192u;
constexpr size_t tile_records = size_t(tile_rows) * tile_groups;
constexpr size_t invalid_offset = size_t(-1);

QRT_BLOCKED_HALF_INLINE bool valid(unsigned rows, unsigned width) {
    return rows && rows <= maximum_rows && width && width <= maximum_width && width % 16u == 0u;
}
QRT_BLOCKED_HALF_INLINE size_t records(unsigned rows, unsigned width) {
    return valid(rows, width) ? size_t((rows + tile_rows - 1u) / tile_rows) *
        ((width / 16u + tile_groups - 1u) / tile_groups) * tile_records : 0u;
}
QRT_BLOCKED_HALF_INLINE size_t offset(unsigned rows, unsigned width, unsigned row, unsigned group) {
    if (!valid(rows, width) || row >= rows || group >= width / 16u) return invalid_offset;
    const unsigned blocks = (width / 16u + tile_groups - 1u) / tile_groups;
    return ((size_t(row / tile_rows) * blocks + group / tile_groups) * tile_rows +
        row % tile_rows) * tile_groups + group % tile_groups;
}
} // namespace qrt_sm121_blocked_half_layout
#undef QRT_BLOCKED_HALF_INLINE
