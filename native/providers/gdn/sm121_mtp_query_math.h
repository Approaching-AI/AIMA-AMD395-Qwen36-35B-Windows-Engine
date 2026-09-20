#pragma once
#include "sm121_mtp_kv_math.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MTP_QUERY_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MTP_QUERY_INLINE inline
#endif

namespace qrt_sm121_mtp {
QRT_MTP_QUERY_INLINE float query_inverse(const float* warps, const unsigned char* table) {
    using namespace qrt_sm121_q1;
    return qrt_sm121_rsqrt::evaluate(table,
        add(multiply(head_norm_warp_sum(warps, false), 1.0f / 256.0f), 1.0e-6f));
}
} // namespace qrt_sm121_mtp
#undef QRT_MTP_QUERY_INLINE
