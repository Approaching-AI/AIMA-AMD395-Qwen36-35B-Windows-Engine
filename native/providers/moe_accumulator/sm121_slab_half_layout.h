#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#define QRT_SLAB_INLINE __host__ __device__ constexpr inline
#else
#define QRT_SLAB_INLINE constexpr inline
#endif

// Two K16 groups per row occupy one aligned 64-byte payload. Controls use a
// separate plane. A small row slab is the unit of locality for candidate bins.
// This is a lossless address permutation, including the original fallback tag.
namespace qrt_sm121_slab_half_layout {
constexpr unsigned maximum_rows = 524288u, maximum_width = 8192u;
constexpr size_t invalid = size_t(-1);
template<unsigned Rows> QRT_SLAB_INLINE bool valid(unsigned rows, unsigned width) {
    static_assert(Rows == 16u || Rows == 64u || Rows == 256u);
    return rows && rows <= maximum_rows && width && width <= maximum_width && !(width & 15u);
}
template<unsigned Rows> QRT_SLAB_INLINE size_t records(unsigned rows, unsigned width) {
    return valid<Rows>(rows,width) ? size_t((rows+Rows-1u)/Rows) *
        ((width/16u+1u)/2u) * Rows * 2u : 0u;
}
template<unsigned Rows> QRT_SLAB_INLINE size_t words(unsigned rows, unsigned width) {
    return records<Rows>(rows,width)*9u;
}
template<unsigned Rows> QRT_SLAB_INLINE size_t offset(unsigned rows, unsigned width,
    unsigned row, unsigned group) {
    if (!valid<Rows>(rows,width) || row >= rows || group >= width/16u) return invalid;
    return ((size_t(row/Rows)*((width/16u+1u)/2u)+group/2u)*Rows+row%Rows)*2u+group%2u;
}
template<unsigned Rows> QRT_SLAB_INLINE unsigned row(unsigned width, size_t record) {
    const size_t slab = record/(Rows*2u), slabs = (width/16u+1u)/2u;
    return unsigned(slab/slabs)*Rows+unsigned(record/2u)%Rows;
}
template<unsigned Rows> QRT_SLAB_INLINE unsigned group(unsigned width, size_t record) {
    return unsigned(record/(Rows*2u)%((width/16u+1u)/2u))*2u+unsigned(record%2u);
}
} // namespace qrt_sm121_slab_half_layout
#undef QRT_SLAB_INLINE
