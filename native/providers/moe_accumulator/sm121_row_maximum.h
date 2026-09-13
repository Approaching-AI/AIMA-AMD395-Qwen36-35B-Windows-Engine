#ifndef QRT_SM121_ROW_MAXIMUM_H
#define QRT_SM121_ROW_MAXIMUM_H
#include "sm121_float_alignment.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_ROW_MAX_INLINE __host__ __device__ __forceinline__
#else
#define QRT_ROW_MAX_INLINE inline
#endif
namespace qrt_sm121_row_maximum {
// One word per complete row, exactly the old row-flag allocation. The
// invalid sentinel dominates every finite exponent during a maximum reduce.
constexpr uint32_t invalid = 256u;
QRT_ROW_MAX_INLINE uint32_t word(uint16_t value) {
    return !qrt_sm121_float_alignment::eligible(value) ? invalid :
        !(value & 0x7fffu) ? 0u : (value >> 7u) & 255u;
}
QRT_ROW_MAX_INLINE uint32_t append(uint32_t current, uint16_t value) {
    const uint32_t next = word(value);
    return next > current ? next : current;
}
QRT_ROW_MAX_INLINE bool eligible(uint32_t left, uint32_t right) {
    return left < invalid && right < invalid;
}
QRT_ROW_MAX_INLINE int product_upper(uint32_t left, uint32_t right) {
    return int(left + right) - 254;
}
QRT_ROW_MAX_INLINE bool carry_dominates(int exponent, int product_upper) {
    // The same scale range as validated scalar replay. It also puts the
    // carried exponent above the original -133 exponent of zero products.
    return exponent >= -101 && exponent <= 151 && exponent >= product_upper;
}
} // namespace qrt_sm121_row_maximum
#undef QRT_ROW_MAX_INLINE
#endif
