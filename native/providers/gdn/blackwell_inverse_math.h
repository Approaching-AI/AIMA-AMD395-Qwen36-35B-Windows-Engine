#ifndef QRT_FLA_BLACKWELL_INVERSE_MATH_H
#define QRT_FLA_BLACKWELL_INVERSE_MATH_H
#include <cmath>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_INVERSE_HD __host__ __device__ __forceinline__
#else
#define QRT_INVERSE_HD inline
#endif
namespace qrt_fla_blackwell_inverse {
// SM121's matching two-warp inverse reduces rows within each eight-row
// half by XOR 4/2/1, then adds the halves. The first pair is an explicit FMA
// of the local product and the separately rounded partner product.
QRT_INVERSE_HD float row_update(const float* a, const float* inverse, unsigned row, unsigned col) {
    float pair[8];
    for (unsigned half = 0; half < 2; ++half) for (unsigned i = 0; i < 4; ++i) {
        const unsigned j = half * 8u + i;
        const float left = 0.0f - a[row * 16u + j];
        const float other = 0.0f - a[row * 16u + j + 4u];
        pair[half * 4u + i] = ::fmaf(left, inverse[j * 16u + col], other * inverse[(j + 4u) * 16u + col]);
    }
    const float low = (pair[0] + pair[2]) + (pair[1] + pair[3]);
    const float high = (pair[4] + pair[6]) + (pair[5] + pair[7]);
    return (0.0f - a[row * 16u + col]) + (low + high);
}
QRT_INVERSE_HD float dot16(const float* a, const float* b, unsigned row, unsigned col, float accumulator) {
    for (unsigned k = 0; k < 16u; ++k) accumulator = ::fmaf(a[row * 16u + k], b[k * 16u + col], accumulator);
    return accumulator;
}
}
#undef QRT_INVERSE_HD
#endif
