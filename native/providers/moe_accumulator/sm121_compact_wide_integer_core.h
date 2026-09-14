#ifndef QRT_SM121_COMPACT_WIDE_INTEGER_CORE_H
#define QRT_SM121_COMPACT_WIDE_INTEGER_CORE_H
#include "sm121_wide_integer_core.h"
#include <cstddef>

#if defined(__HIPCC__)
#define QRT_COMPACT_WIDE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_COMPACT_WIDE_INLINE inline
#endif

namespace qrt_sm121_compact_wide_core {
// The FP16 high/low digits are derived from original BF16 and unit. Retain
// every other bit used by the established wide integer carry and fallback.
struct Row {
    uint16_t original[18];
    uint32_t trailing[4];
    int unit, maximum;
    uint32_t exceptions, nonzero;
};
static_assert(sizeof(Row) == 68u);
static_assert(offsetof(Row, trailing) == 36u);
static_assert(offsetof(qrt_sm121_wide_core::Row, trailing) == 100u);
static_assert(sizeof(qrt_sm121_wide_core::Row) - sizeof(Row) == 64u);

QRT_COMPACT_WIDE_INLINE unsigned expanded_word(unsigned word) {
    return word >= 9u ? word + 16u : word;
}

QRT_COMPACT_WIDE_INLINE Row pack(const qrt_sm121_wide_core::Row& source) {
    Row row{};
    for (unsigned i = 0u; i < 18u; ++i) row.original[i] = source.original[i];
    for (unsigned i = 0u; i < 4u; ++i) row.trailing[i] = source.trailing[i];
    row.unit = source.unit; row.maximum = source.maximum;
    row.exceptions = source.exceptions; row.nonzero = source.nonzero;
    return row;
}

QRT_COMPACT_WIDE_INLINE void expand_pair(qrt_sm121_wide_core::Row& row, unsigned pair) {
    for (unsigned half = 0u; half < 2u; ++half) {
        const unsigned i = pair * 2u + half;
        const uint32_t positive = qrt_sm121_wide_core::encode(row.original[i], row.unit) ^
            qrt_sm121_wide_core::center;
        row.high[i] = qrt_sm121_positive_karatsuba::positive_half_bits(positive >> 9u);
        row.low[i] = qrt_sm121_positive_karatsuba::positive_half_bits(positive & 511u);
    }
}

QRT_COMPACT_WIDE_INLINE qrt_sm121_wide_core::Row expand(const Row& source) {
    qrt_sm121_wide_core::Row row{};
    for (unsigned i = 0u; i < 18u; ++i) row.original[i] = source.original[i];
    for (unsigned i = 0u; i < 4u; ++i) row.trailing[i] = source.trailing[i];
    row.unit = source.unit; row.maximum = source.maximum;
    row.exceptions = source.exceptions; row.nonzero = source.nonzero;
    for (unsigned pair = 0u; pair < 8u; ++pair) expand_pair(row, pair);
    return row;
}
}  // namespace qrt_sm121_compact_wide_core
#undef QRT_COMPACT_WIDE_INLINE
#endif
