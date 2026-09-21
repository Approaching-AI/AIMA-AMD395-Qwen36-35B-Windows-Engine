#pragma once
#include "sm121_compact_matrix_group.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MATRIX_METADATA_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MATRIX_METADATA_INLINE inline
#endif

// Matrix products already contain every signed16 coefficient. The carry
// certificate needs only these eight words; raw operands remain separately
// available to the unchanged original-group fallback.
namespace qrt_sm121_compact_matrix_metadata {
namespace group = qrt_sm121_compact_matrix_group;
struct Row {
    uint32_t control;
    uint32_t exponents[4];
    uint32_t trailing[2];
    uint32_t first_pair;
};
static_assert(sizeof(Row) == 32u);

QRT_MATRIX_METADATA_INLINE Row prepare(const group::Row& input) {
    return {input.encoded.control,
        {input.encoded.exponents[0], input.encoded.exponents[1],
         input.encoded.exponents[2], input.encoded.exponents[3]},
        {input.trailing[0], input.trailing[1]}, input.encoded.pairs[0]};
}

QRT_MATRIX_METADATA_INLINE group::Row expand(const Row& input) {
    group::Row result{};
    result.encoded.control = input.control;
    for (unsigned i = 0u; i < 4u; ++i)
        result.encoded.exponents[i] = input.exponents[i];
    result.trailing[0] = input.trailing[0];
    result.trailing[1] = input.trailing[1];
    result.encoded.pairs[0] = input.first_pair;
    return result;
}

QRT_MATRIX_METADATA_INLINE bool accumulate(float carry, const Row& left,
    const Row& right, int64_t mathematical, float* output) {
    // Use the original certificate and carry arithmetic. expand supplies all
    // fields read by that function, without reconstructing unused coefficients.
    return group::accumulate(carry, expand(left), expand(right), mathematical, output);
}
} // namespace qrt_sm121_compact_matrix_metadata
#undef QRT_MATRIX_METADATA_INLINE
