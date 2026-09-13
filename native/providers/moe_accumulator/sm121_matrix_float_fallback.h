#ifndef QRT_SM121_MATRIX_FLOAT_FALLBACK_H
#define QRT_SM121_MATRIX_FLOAT_FALLBACK_H
#include "sm121_integer_core.h"
#include "sm121_float_alignment.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MATRIX_FLOAT_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MATRIX_FLOAT_INLINE inline
#endif

namespace qrt_sm121_matrix_float_fallback {
enum Path : unsigned { ExactInteger=0u, ScalarFloat=1u, OriginalInteger=2u };
QRT_MATRIX_FLOAT_INLINE bool eligible(const uint16_t* row) {
    bool valid=true;
    for (unsigned i=0u;i<16u;++i)valid=valid&&qrt_sm121_float_alignment::eligible(row[i]);
    return valid;
}

// Matrix partials are accepted only by the existing exact integer certificate.
// Otherwise compute the original aligned K16 from scalar FP32 products, with
// eligibility established once when preparing operands. No general integer
// truncation-compensation loop runs. Exceptional rows/carried values retain
// the original sixteen BF16 products and unsigned modulo sum.
QRT_MATRIX_FLOAT_INLINE Path sum(qrt_q1_moe_hawkeye::Value carry,
    const qrt_sm121_integer_core::Row& left,const qrt_sm121_integer_core::Row& right,
    const int32_t (&partials)[4],bool float_eligible,qrt_sm121_group16::AlignedSum* output) {
    const uint32_t exceptions=(left.exceptions|right.exceptions)&left.nonzero&right.nonzero;
    if (left.unit>=0 && right.unit>=0 && !exceptions) {
        const int64_t product=int64_t(partials[0])*65536+(int64_t(partials[1])+partials[2])*256+partials[3];
        if (qrt_sm121_integer_parts::sum_exact_integer_product(carry,product,
                left.unit,left.maximum,right.unit,right.maximum,output,left.trailing,right.trailing))
            return ExactInteger;
    }
    // A nonzero original FP32 carry never has exponent below -126. Guard the
    // wider internal-Value API before converting such a synthetic carry.
    if (float_eligible && (!carry.significand || carry.exponent>=-126)) {
        qrt_sm121_float_alignment::Group group;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (unsigned i=0u;i<16u;++i)group.set(i,left.original[i],right.original[i]);
        if (qrt_sm121_float_alignment::sum(carry,group,output))return ScalarFloat;
    }
    uint32_t products[16];
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i=0u;i<16u;++i)
        products[i]=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(
            left.original[i],right.original[i],-133));
    *output=qrt_sm121_group16::sum_packed(carry,products);
    return OriginalInteger;
}
} // namespace qrt_sm121_matrix_float_fallback
#undef QRT_MATRIX_FLOAT_INLINE
#endif
