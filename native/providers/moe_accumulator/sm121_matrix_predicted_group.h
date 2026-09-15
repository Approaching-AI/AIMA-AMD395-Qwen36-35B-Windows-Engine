#pragma once
#include "sm121_integer_core.h"
#include "sm121_predicted_group.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MATRIX_PLAN_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MATRIX_PLAN_INLINE inline
#endif

// Exact integer matrix products replace the sixteen scalar product/scaling
// conversions in a predicted plan. Only original exception/remainder pairs
// are replayed. Prediction still changes scheduling, never accepted math.
namespace qrt_sm121_matrix_predicted_group {
namespace core=qrt_sm121_integer_core;
namespace planned=qrt_sm121_predicted_group;
using Row=core::Row;
constexpr int64_t maximum_dot=int64_t(16)*32640*32640;

// Preconditions: immutable rows come from core::prepare, every BF16 operand
// satisfies float_alignment::eligible, and mathematical is their exact signed
// integer-core dot. Four IU8 matrices reconstruct it without floating rounding.
QRT_MATRIX_PLAN_INLINE planned::Plan prepare(const Row& left,const Row& right,
    int64_t mathematical,int predicted_exponent,unsigned* replayed_pairs=nullptr){
    if(replayed_pairs)*replayed_pairs=0u;
    if(predicted_exponent < -133 || predicted_exponent > 511 || left.unit<0 || right.unit<0 ||
        mathematical < -maximum_dot || mathematical > maximum_dot)return {};
    const int product_max=core::paired_maximum(left,right,-133);
    const int alignment=product_max>predicted_exponent?product_max:predicted_exponent;
    const bool first_negative=((left.original[0]^right.original[0])&0x8000u)!=0u;
    if(alignment==-133)return planned::encode(0u,-133,product_max,first_negative);
    if(alignment < -101 || alignment > 127)return {};
    if(!(left.nonzero&right.nonzero))return planned::encode(0u,alignment,product_max,first_negative);
    const int shift=alignment-(left.unit+right.unit-254)-11;
    // Oppositely located large operands can leave every core product zero
    // while original exception products remain nonzero. Such a plan needs no
    // core left shift; preserve the original exception-only compensation.
    if(shift < -25 && mathematical)return {};
    const uint32_t exceptions=(left.exceptions|right.exceptions)&left.nonzero&right.nonzero;
    const uint32_t losses=shift>0&&shift<30?core::remainder_mask(left,right,unsigned(shift)):0u;
    uint32_t pending=losses|exceptions;
    int64_t discarded=0,corrections=0;
    while(pending){
        const unsigned i=core::first_bit(pending);pending&=pending-1u;
        if(replayed_pairs)++*replayed_pairs;
        const uint16_t a=left.original[i],b=right.original[i];
        const int ac=core::signed_core(a,left.unit),bc=core::signed_core(b,right.unit);
        const uint32_t magnitude=uint32_t(ac<0?-ac:ac)*uint32_t(bc<0?-bc:bc);
        const bool negative=((a^b)&0x8000u)!=0u;
        if(losses&(1u<<i)){
            const uint32_t remainder=magnitude&((1u<<unsigned(shift))-1u);
            discarded+=negative?-int64_t(remainder):int64_t(remainder);
        }
        if(exceptions&(1u<<i)){
            const auto original=qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133);
            const unsigned distance=unsigned(alignment-original.exponent);
            const uint32_t aligned=distance>=32u?0u:(original.significand<<2u)>>distance;
            const uint32_t core_aligned=!magnitude||shift>=30?0u:shift>0?
                magnitude>>unsigned(shift):uint32_t(uint64_t(magnitude)<<unsigned(-shift));
            const int64_t difference=int64_t(aligned)-core_aligned;
            corrections+=negative?-difference:difference;
        }
    }
    const int64_t compensated=mathematical-discarded;
    const int64_t products=!compensated||shift>=30?0:shift<=0?
        compensated*(int64_t(1)<<unsigned(-shift)):compensated<0?
        -int64_t(uint64_t(-compensated)>>unsigned(shift)):
        int64_t(uint64_t(compensated)>>unsigned(shift));
    return planned::encode(uint32_t(products+corrections),alignment,product_max,first_negative);
}
} // namespace qrt_sm121_matrix_predicted_group
#undef QRT_MATRIX_PLAN_INLINE
