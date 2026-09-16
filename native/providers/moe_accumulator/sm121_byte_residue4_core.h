#ifndef QRT_SM121_BYTE_RESIDUE4_CORE_H
#define QRT_SM121_BYTE_RESIDUE4_CORE_H
#include "sm121_byte_residue32_core.h"
#if defined(__HIPCC__)
#define QRT_RESIDUE4_INLINE __host__ __device__ __forceinline__
#else
#define QRT_RESIDUE4_INLINE inline
#endif

// Isolated H4 core. Under the observed natural-DOT2 hardware model, eight
// pairs of exact BF16-derived integers bounded by4080 have error<=245/4<128.
// This conditional derivation is not a universal proof of hardware semantics.
// Wider H5/H6 rows do not inherit this bound. No provider uses this header.
namespace qrt_sm121_byte_residue4 {
namespace original=qrt_sm121_byte_residue_core;
using Row=original::Row;
using Value=original::Value;
using AlignedSum=original::AlignedSum;
constexpr int32_t maximum_dot=16*4080*4080;
static_assert(maximum_dot==266342400);
QRT_RESIDUE4_INLINE void prepare(Row& row){original::prepare<4u>(row);}
QRT_RESIDUE4_INLINE bool recover(float approximate,uint32_t residue,int32_t* output){
    if(!(approximate>-float(maximum_dot+512) && approximate<float(maximum_dot+512)))return false;
    const int32_t truncated=int32_t(approximate);
    const uint32_t offset=(residue-uint32_t(truncated))&255u;
    if(offset==128u && approximate==float(truncated))return false;
    int32_t result=truncated+int32_t(offset);
    if(offset>128u || (offset==128u && approximate<float(truncated)))result-=256;
    if(result < -maximum_dot || result > maximum_dot)return false;
    *output=result;return true;
}
QRT_RESIDUE4_INLINE bool sum(Value carry,const Row& left,const Row& right,int32_t mathematical,AlignedSum* output){
    if(mathematical < -maximum_dot || mathematical > maximum_dot)return false;
    // The existing signed32 compensation proof covers the larger H5 range.
    // Its conservative shift>=26 shortcut is also valid for these H4 cores.
    return qrt_sm121_byte_residue32::sum(carry,left,right,mathematical,output);
}
using qrt_sm121_byte_residue32::fallback;
#if defined(__HIPCC__)
using original::products;
#endif
}
#undef QRT_RESIDUE4_INLINE
#endif
