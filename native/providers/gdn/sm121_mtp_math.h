#pragma once
#include "sm121_q1_math.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MTP_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MTP_INLINE inline
#endif

namespace qrt_sm121_mtp {
// Original selected GCEZ... PTX d880f540... owns four stride-512 BF16
// values per lane. It squares them independently before ordered addition.
// This differs from the eight adjacent values in target embedding RMSNorm.
QRT_MTP_INLINE float lane_sumsq(const uint16_t* row, unsigned int lane) {
    using namespace qrt_sm121_q1;
    float value = widen(row[lane]);
    float sum = multiply(value, value);
    for (unsigned int item = 1; item < 4; ++item) {
        value = widen(row[lane + item * 512u]);
        sum = add(sum, multiply(value, value));
    }
    return sum;
}

// The separately observed S4L4... launcher uses R0_BLOCK=1024. Each lane
// keeps two partials across the two reduction blocks, then adds them before
// the same warp reduction. PTX d248c8... uses FMA for each partial update.
QRT_MTP_INLINE float lane_sumsq_split1024(const uint16_t* row, unsigned int lane) {
    using namespace qrt_sm121_q1;
    const float a = widen(row[lane]), b = widen(row[lane + 512u]);
    const float c = widen(row[lane + 1024u]), d = widen(row[lane + 1536u]);
    return add(fmaf(c, c, fmaf(a, a, 0.0f)), fmaf(d, d, fmaf(b, b, 0.0f)));
}

QRT_MTP_INLINE float sum_warps(const float* warps) {
    float partial[16];
    for (unsigned int i = 0; i < 16; ++i) partial[i] = warps[i];
    for (unsigned int offset = 8; offset; offset >>= 1)
        for (unsigned int i = 0; i < offset; ++i)
            partial[i] = qrt_sm121_q1::add(partial[i], partial[i + offset]);
    return partial[0];
}

QRT_MTP_INLINE float inverse(float sumsq, const unsigned char* rsqrt_table) {
    using namespace qrt_sm121_q1;
    return qrt_sm121_rsqrt::evaluate(rsqrt_table,
        add(multiply(sumsq, 1.0f / 2048.0f), 1.0e-6f));
}

QRT_MTP_INLINE uint16_t normalized(float value, float inverse, uint16_t weight) {
    using namespace qrt_sm121_q1;
    return bf16(multiply(multiply(value, inverse), add(1.0f, widen(weight))));
}
} // namespace qrt_sm121_mtp
#undef QRT_MTP_INLINE
