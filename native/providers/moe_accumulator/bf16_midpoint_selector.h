#ifndef QRT_BF16_MIDPOINT_SELECTOR_H
#define QRT_BF16_MIDPOINT_SELECTOR_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MIDPOINT_HD __host__ __device__
#else
#define QRT_MIDPOINT_HD
#endif

namespace qrt_bf16_midpoint {

// Admission only: the caller supplies an independently justified absolute
// error estimate, then recomputes admitted dots from live inputs and weights.
// Adjacent midpoints matter at binade boundaries: immediately above 1.0 the
// lower rounding boundary is closer than the midpoint in the current binade.
QRT_MIDPOINT_HD inline bool within_error(float value, float error) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits &= UINT32_C(0x7fffffff);
    if (bits >= UINT32_C(0x7f800000)) {
        return true;
    }
    if (!(error >= 0.0f)) {
        return true;
    }
    float magnitude;
    memcpy(&magnitude, &bits, sizeof(magnitude));
    const uint32_t middle = (bits & UINT32_C(0xffff0000)) | UINT32_C(0x8000);
    for (int offset = -1; offset <= 1; ++offset) {
        if (offset == -1 && middle < UINT32_C(0x10000)) {
            continue;
        }
        const uint32_t candidate = middle + offset * UINT32_C(0x10000);
        if (candidate >= UINT32_C(0x7f800000)) {
            continue;
        }
        float midpoint;
        memcpy(&midpoint, &candidate, sizeof(midpoint));
        if (fabsf(magnitude - midpoint) <= error) {
            return true;
        }
    }
    return false;
}

}  // namespace qrt_bf16_midpoint

#undef QRT_MIDPOINT_HD
#endif
